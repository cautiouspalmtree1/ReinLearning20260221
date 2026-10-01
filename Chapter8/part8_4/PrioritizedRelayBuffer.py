import numpy as np


class SumTree:
    """线段树，O(log n) 更新和采样优先级。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)  # 线段树节点

    def update(self, idx, priority):
        """更新叶节点，并向上传播。"""
        pos = idx + self.capacity
        self.tree[pos] = priority
        pos >>= 1
        while pos >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos >>= 1

    def batch_update(self, idxs, priorities):
        """批量更新，减少循环开销。"""
        for idx, p in zip(idxs, priorities):
            self.update(idx, p)

    def sample(self, value):
        """按 value 在 [0, total) 中找对应叶节点索引。"""
        pos = 1
        while pos < self.capacity:
            left = 2 * pos
            if value <= self.tree[left]:
                pos = left
            else:
                value -= self.tree[left]
                pos = left + 1
        return pos - self.capacity  # 返回数据索引 [0, capacity)

    @property
    def total(self):
        return self.tree[1]

    def __getitem__(self, idx):
        return self.tree[idx + self.capacity]


class PrioritizedReplayBuffer:
    """
    优先级经验回放缓冲区。

    核心优化：
    - 分离存储每个字段（states/actions/rewards/...），彻底消除 zip(*batch)
    - states/next_states 用 float16 存储，节省约50%显存
    - get_batch 直接返回 numpy 数组，调用方一次 torch.as_tensor 搞定
    """

    def __init__(
        self,
        buffer_size: int = 50000,
        batch_size: int = 32,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
        frame_shape: tuple = (4, 84, 84),
    ):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon  # 防止优先级为0
        self.frame_count = 0   # 用于 beta 退火

        self.pos = 0    # 写入指针
        self.size = 0   # 当前已存条数

        # ✅ 分离存储，直接 numpy 索引，无需 zip
        self.states      = np.zeros((buffer_size, *frame_shape), dtype=np.float16)
        self.actions     = np.zeros(buffer_size, dtype=np.int64)
        self.rewards     = np.zeros(buffer_size, dtype=np.float32)
        self.next_states = np.zeros((buffer_size, *frame_shape), dtype=np.float16)
        self.dones       = np.zeros(buffer_size, dtype=np.float32)

        # 优先级线段树
        self.tree = SumTree(buffer_size)
        self._max_priority = 1.0  # 新样本初始优先级

    # ------------------------------------------------------------------
    # 兼容旧接口：add((state, action, reward, next_state, done))
    # 同时支持新接口：add(state, action, reward, next_state, done)
    # ------------------------------------------------------------------
    def add(self, *args):
        if len(args) == 1:
            # 旧接口：buffer.add((s, a, r, s', done))
            state, action, reward, next_state, done = args[0]
        else:
            # 新接口：buffer.add(s, a, r, s', done)
            state, action, reward, next_state, done = args

        # state 可以是 numpy array 或 CPU tensor
        if hasattr(state, 'cpu'):
            state = state.cpu().numpy()
        if hasattr(next_state, 'cpu'):
            next_state = next_state.cpu().numpy()

        self.states[self.pos]      = state.astype(np.float16)
        self.actions[self.pos]     = action
        self.rewards[self.pos]     = reward
        self.next_states[self.pos] = next_state.astype(np.float16)
        self.dones[self.pos]       = float(done)

        # 新样本用最大优先级，保证至少被采一次
        self.tree.update(self.pos, self._max_priority ** self.alpha)

        self.pos  = (self.pos + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def get_batch(self):
        """
        返回 (idxs, states, actions, rewards, next_states, dones, weights)
        全部为 numpy array，调用方负责 torch.as_tensor 转换。
        """
        assert self.size >= self.batch_size, "buffer 数据不足，无法采样"

        self.frame_count += 1
        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self.frame_count / self.beta_frames
        )

        idxs = np.zeros(self.batch_size, dtype=np.int64)
        segment = self.tree.total / self.batch_size

        for i in range(self.batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idxs[i] = self.tree.sample(value)

        # ✅ 直接用 numpy fancy index，无任何 Python 循环
        states      = self.states[idxs].astype(np.float32)       # float16 -> float32
        actions     = self.actions[idxs]
        rewards     = self.rewards[idxs]
        next_states = self.next_states[idxs].astype(np.float32)
        dones       = self.dones[idxs]

        # 重要性采样权重
        total = self.tree.total
        probs = np.array([self.tree[i] for i in idxs], dtype=np.float32) / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()  # 归一化，最大权重=1

        return idxs, states, actions, rewards, next_states, dones, weights

    def update_priorities(self, idxs, errors):
        """
        errors: numpy array，shape (batch_size,) 或 (batch_size, 1)
        """
        errors = np.abs(errors).flatten() + self.epsilon
        priorities = errors ** self.alpha
        self._max_priority = max(self._max_priority, priorities.max())
        self.tree.batch_update(idxs, priorities)

    def __len__(self):
        return self.size