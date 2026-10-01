"""
replay_buffer.py — 优先经验回放 (Prioritized Experience Replay, PER)

【普通经验回放的问题】
  普通DQN把所有经验等概率采样，但有些经验比其他更有价值
  （比如遇到罕见的高奖励情况）

【PER的核心思想】
  TD误差(TD error)大 → 这个经验让我们"很惊讶" → 更值得学习
  
  优先级 p_i = |δ_i| + ε  （δ是TD误差，ε防止优先级为0）
  采样概率 P(i) = p_i^α / Σ p_j^α
    α=0：完全均匀采样（退化为普通回放）
    α=1：完全按优先级采样
  
  为了纠正采样偏差，需要重要性采样权重：
  w_i = (1 / N / P(i))^β
  训练时loss乘以w_i来修正偏差

【数据结构：线段树 (Segment Tree)】
  问题：每次采样都要从N个经验中按优先级采样，O(N)太慢
  解决：用线段树，叶节点存优先级，内部节点存子树优先级之和
        采样一个经验只需O(log N)时间
  
  线段树示意（8个叶节点）：
                    [总和]
               /           \
          [左半和]        [右半和]
          /     \          /    \
       [和]    [和]     [和]   [和]
       / \    / \      / \    / \
      p1 p2  p3 p4   p5 p6  p7 p8  ← 叶节点（每个经验的优先级）
"""

import numpy as np


class SumTree:
    """
    线段树：支持O(log N)的优先级采样
    
    用数组实现：
    - 索引0是根节点
    - 索引i的左孩子是 2*i+1，右孩子是 2*i+2
    - 叶节点从索引 n_leaves-1 开始
    
    例如n_leaves=4时，数组长度为7：
    索引: 0    1    2    3    4    5    6
    角色: 根  左1  右1  叶1  叶2  叶3  叶4
    """
    
    def __init__(self, capacity):
        """
        Args:
            capacity: 存储的经验数量（叶节点数）
        """
        self.capacity = capacity  # 叶节点数（经验容量）
        
        # 完全二叉树的数组大小 = 2 * capacity - 1
        # capacity个叶节点 + capacity-1个内部节点
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        
        # 存储实际的经验数据（用列表存字典）
        self.data = [None] * capacity
        
        # 写入指针（循环覆盖旧数据）
        self.write_pos = 0
        
        # 当前存储了多少条经验
        self.size = 0
    
    def _propagate(self, idx, delta):
        """
        更新一个叶节点后，向上更新所有祖先节点的和
        
        Args:
            idx: 树中的节点索引（不是叶节点的偏移量）
            delta: 该节点值的变化量
        """
        parent = (idx - 1) // 2  # 父节点索引
        self.tree[parent] += delta  # 更新父节点的和
        
        # 如果还没到根节点，继续向上传播
        if parent != 0:
            self._propagate(parent, delta)
    
    def _retrieve(self, idx, s):
        """
        从节点idx开始，找到累积优先级为s的叶节点
        这是PER采样的核心：类似"在总优先级上随机选一段"
        
        原理：
        - 生成随机数 s ∈ [0, total_priority]
        - 在线段树上二分找到第一个叶节点，使得前缀和 >= s
        - 这等价于按优先级比例采样
        
        Args:
            idx: 当前节点索引（初始为根节点0）
            s: 要查找的累积优先级值
        Returns:
            叶节点的树索引
        """
        left = 2 * idx + 1   # 左孩子索引
        right = 2 * idx + 2  # 右孩子索引
        
        # 到达叶节点，返回
        if left >= len(self.tree):
            return idx
        
        # 如果s小于等于左子树的和，往左走
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            # 否则往右走，并减去左子树的和
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """返回所有优先级之和（根节点的值）"""
        return self.tree[0]
    
    def add(self, priority, data):
        """
        添加一条新经验
        
        Args:
            priority: 优先级（标量，通常是|TD误差| + epsilon）
            data: 经验数据（元组或字典）
        """
        # 叶节点在树中的索引 = capacity-1 + 叶节点偏移量
        tree_idx = self.write_pos + self.capacity - 1
        
        # 存储数据
        self.data[self.write_pos] = data
        
        # 更新优先级（会同时更新所有祖先节点）
        self.update(tree_idx, priority)
        
        # 移动写入指针（循环覆盖）
        self.write_pos = (self.write_pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def update(self, tree_idx, priority):
        """
        更新某个叶节点的优先级
        
        Args:
            tree_idx: 树中的节点索引
            priority: 新的优先级值
        """
        delta = priority - self.tree[tree_idx]  # 变化量
        self.tree[tree_idx] = priority           # 更新叶节点
        self._propagate(tree_idx, delta)         # 向上更新祖先
    
    def get(self, s):
        """
        按累积优先级s采样一个经验
        
        Args:
            s: 随机数，范围[0, total()]
        Returns:
            (tree_idx, priority, data) 元组
              - tree_idx: 用于后续更新优先级
              - priority: 该经验的优先级
              - data: 实际的经验数据
        """
        tree_idx = self._retrieve(0, s)                        # 找叶节点
        data_idx = tree_idx - self.capacity + 1               # 转为数据索引
        return tree_idx, self.tree[tree_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    使用线段树实现O(log N)的带优先级采样
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            capacity: 最大存储经验数
            alpha: 优先级指数（0=均匀采样，1=完全按优先级）
            beta_start: 重要性采样权重初始值（0=不修正，1=完全修正）
            beta_frames: 多少步后beta线性增长到1.0
                        随训练进行，越来越注重纠正偏差
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0  # 用于计算当前beta值
        
        # 线段树存储经验
        self.tree = SumTree(capacity)
        
        # 初始最大优先级（新经验都赋予当前最大优先级，确保至少被学一次）
        self.max_priority = 1.0
        
        # 防止优先级为0的小常数
        self.epsilon = 1e-6
    
    @property
    def beta(self):
        """
        beta随训练进行从beta_start线性增长到1.0
        
        为什么要这样做？
        - 训练初期：模型差，TD误差不准确，不需要强力纠偏
        - 训练后期：模型好，TD误差更准，需要完全纠偏以保证无偏
        """
        # 线性插值: beta_start → 1.0
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储一条经验
        新经验赋予当前最大优先级（确保每条新经验至少被学习一次）
        
        Args:
            state: 当前状态 (4, 84, 84)
            action: 执行的动作（整数）
            reward: 获得的奖励（标量）
            next_state: 下一状态 (4, 84, 84)
            done: 是否结束（bool）
        """
        # 打包成元组
        experience = (state, action, reward, next_state, done)
        
        # 新经验赋予最大优先级（p^alpha）
        priority = self.max_priority ** self.alpha
        
        self.tree.add(priority, experience)
    
    def sample(self, batch_size):
        """
        按优先级采样一批经验
        
        采样方法：分层采样(stratified sampling)
        - 把总优先级等分为batch_size段
        - 每段内随机采样一个
        - 这样保证样本分布均匀，减少方差
        
        Args:
            batch_size: 批次大小
        Returns:
            states, actions, rewards, next_states, dones: 批次数据
            weights: 重要性采样权重 (batch_size,)
            tree_indices: 用于后续更新优先级
        """
        self.frame += 1
        
        # 分层采样：把总优先级分成batch_size段
        segment = self.tree.total() / batch_size
        
        tree_indices = []
        priorities = []
        experiences = []
        
        for i in range(batch_size):
            # 在第i段内随机采样
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)  # 随机点
            
            tree_idx, priority, data = self.tree.get(s)
            
            tree_indices.append(tree_idx)
            priorities.append(priority)
            experiences.append(data)
        
        # 计算重要性采样权重
        # w_i = (1/N * 1/P(i))^beta
        # P(i) = priority_i / total_priority
        priorities = np.array(priorities, dtype=np.float64)
        N = self.tree.size
        
        # 采样概率
        probs = priorities / self.tree.total()
        
        # 重要性采样权重（归一化到最大权重=1，避免梯度过大）
        weights = (N * probs) ** (-self.beta)
        weights = weights / weights.max()  # 归一化
        weights = weights.astype(np.float32)
        
        # 解包经验元组
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights,
            tree_indices
        )
    
    def update_priorities(self, tree_indices, td_errors):
        """
        用新的TD误差更新经验的优先级
        
        在每次学习后调用，因为模型更新了，TD误差也变了
        
        Args:
            tree_indices: 要更新的经验在线段树中的索引
            td_errors: 对应的TD误差（标量列表）
        """
        for tree_idx, td_error in zip(tree_indices, td_errors):
            # 优先级 = (|TD误差| + epsilon)^alpha
            priority = (abs(td_error) + self.epsilon) ** self.alpha
            
            # 更新线段树
            self.tree.update(tree_idx, priority)
            
            # 更新最大优先级（用于新经验的初始优先级）
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """返回当前存储的经验数量"""
        return self.tree.size


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("测试优先经验回放...")
    
    buffer = PrioritizedReplayBuffer(capacity=1000)
    
    # 添加一些假数据
    for i in range(100):
        state = np.random.rand(4, 84, 84).astype(np.float32)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        done = False
        buffer.push(state, action, reward, next_state, done)
    
    print(f"缓冲区大小: {len(buffer)}")
    
    # 采样
    batch = buffer.sample(32)
    states, actions, rewards, next_states, dones, weights, tree_indices = batch
    
    print(f"采样批次 - states: {states.shape}, weights: {weights.shape}")
    print(f"权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
    
    # 更新优先级
    td_errors = np.random.rand(32)
    buffer.update_priorities(tree_indices, td_errors)
    print("优先级更新成功！")
    print("缓冲区测试通过！")
