"""
agent.py — Rainbow DQN 训练器

这个文件是Rainbow DQN的核心，整合了所有7个改进：
1. ✅ Distributional RL (C51)：损失函数用分布的KL散度
2. ✅ Double DQN：用在线网络选动作，目标网络评估价值
3. ✅ Dueling Network：在model.py中实现
4. ✅ Prioritized Experience Replay：用PER采样，更新优先级
5. ✅ Noisy Nets：NoisyLinear代替epsilon-greedy，在model.py中实现
6. ✅ Multi-step Returns：n步回报，让奖励传播更快
7. ✅ Target Network：稳定训练目标

【Double DQN 详解】
  普通DQN: target = r + γ * max_a Q_target(s', a)
           问题：max操作导致Q值高估（对噪声偏乐观）
  
  Double DQN: 用在线网络选动作，用目标网络评估价值
    a* = argmax_a Q_online(s', a)   ← 在线网络选动作
    target = r + γ * Q_target(s', a*)  ← 目标网络评估
  
  这样两个网络相互制约，减少高估

【Multi-step Returns 详解】
  1步回报: G_1 = r_t + γ * V(s_{t+1})
  n步回报: G_n = r_t + γ*r_{t+1} + γ²*r_{t+2} + ... + γⁿ*V(s_{t+n})
  
  好处：奖励信号传播更快（特别是奖励稀疏时）
  代价：更高的方差（因为用了更多真实奖励，但也用了更少的估计值）

【C51 损失函数详解】
  目标：计算目标分布 m，然后最小化 KL(m || p_θ)
  
  目标分布的计算（Bellman更新投影到支撑点上）：
  z_j → r + γ * z_j （Bellman算子把分布"向右移"）
  然后用线性插值把移动后的分布投影回支撑点 [v_min, v_max]
"""

import numpy as np
import torch
import torch.optim as optim
from collections import deque

from model import RainbowDQN
from replay_buffer import PrioritizedReplayBuffer


class RainbowAgent:
    """
    Rainbow DQN 智能体
    
    包含：网络、优化器、经验回放、损失计算
    """

    def __init__(self, n_actions, config):
        """
        Args:
            n_actions: 动作空间大小
            config: 超参数字典（见下面的默认配置）
        """
        self.n_actions = n_actions
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")

        # === 超参数 ===
        self.gamma = config['gamma']  # 折扣因子
        self.n_steps = config['n_steps']  # multi-step return的步数
        self.batch_size = config['batch_size']  # 训练批次大小
        self.target_update = config['target_update']  # 目标网络更新频率
        self.train_freq = config['train_freq']  # 每多少步训练一次
        self.min_replay_size = config['min_replay_size']  # 开始训练的最小经验量

        # C51分布参数
        self.n_atoms = config['n_atoms']
        self.v_min = config['v_min']
        self.v_max = config['v_max']

        # 支撑点间距
        # 比如v_min=-10, v_max=10, n_atoms=51
        # delta_z = 20/50 = 0.4，即每个原子间隔0.4
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # === 创建两个网络 ===
        # 在线网络：每步都更新，用于选动作和计算损失
        self.online_net = RainbowDQN(
            n_actions=n_actions,
            n_atoms=self.n_atoms,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)

        # 目标网络：每隔target_update步复制在线网络
        # 作用：提供稳定的训练目标（不然目标和预测都在变，训练不稳定）
        self.target_net = RainbowDQN(
            n_actions=n_actions,
            n_atoms=self.n_atoms,
            v_min=self.v_min,
            v_max=self.v_max
        ).to(self.device)

        # 初始时目标网络与在线网络完全相同
        self.target_net.load_state_dict(self.online_net.state_dict())

        # 目标网络不需要梯度（只用于推理）
        for param in self.target_net.parameters():
            param.requires_grad = False

        # === 优化器 ===
        # Adam优化器，论文推荐lr=6.25e-5
        self.optimizer = optim.Adam(
            self.online_net.parameters(),
            lr=config['lr'],
            eps=1.5e-4  # 数值稳定性，避免除以0
        )

        # === 优先经验回放 ===
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config['buffer_size'],
            alpha=config['per_alpha'],
            beta_start=config['per_beta'],
            beta_frames=config['per_beta_frames']
        )

        # === Multi-step Return 缓存 ===
        # 用一个小队列暂存n步内的经验，等积累n步后再存入replay buffer
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # === 训练统计 ===
        self.step_count = 0  # 总步数
        self.episode_count = 0  # 总episode数

    def _compute_n_step_return(self):
        """
        计算n步回报
        
        G_n = r_0 + γ*r_1 + γ²*r_2 + ... + γ^(n-1)*r_{n-1}
        
        同时返回初始状态、动作、n步后的状态和是否结束
        
        Returns:
            (state_0, action_0, G_n, state_n, done_n) 元组
        """
        # n_step_buffer中的数据格式：(state, action, reward, next_state, done)

        # 取最早的经验（要存入回放缓冲区的经验）
        state = self.n_step_buffer[0][0]
        action = self.n_step_buffer[0][1]

        # 计算n步累积回报
        G = 0.0
        for i, (_, _, r, _, done) in enumerate(self.n_step_buffer):
            G += (self.gamma ** i) * r
            if done:
                # 如果中途episode结束，后续步骤不计入
                break

        # n步后的状态和是否结束
        # （如果中途结束，这里应该是终止状态，实际实现中有点简化）
        next_state = self.n_step_buffer[-1][3]
        done = self.n_step_buffer[-1][4]

        return state, action, G, next_state, done

    def push_experience(self, state, action, reward, next_state, done):
        """
        收集经验，实现multi-step return
        
        不直接存入回放缓冲区，先存入n_step_buffer，
        积累n步后计算n步回报，再存入回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        # 加入n步缓存
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # 只有当缓存满了（积累了n步），才计算n步回报并存入replay buffer
        if len(self.n_step_buffer) == self.n_steps:
            s, a, G, s_n, d = self._compute_n_step_return()
            self.replay_buffer.push(s, a, G, s_n, d)

        # 如果episode结束，把缓存里剩余的经验也存入（步数不足n步也存）
        if done:
            # 逐步处理缓存中剩余的经验
            # 比如n=3，缓存有[t, t+1, t+2]，episode在t+1结束
            # 我们还需要处理 [t+1, t+2] 和 [t+2] 这两个子序列
            for i in range(1, len(self.n_step_buffer)):
                sub_buffer = list(self.n_step_buffer)[i:]
                # 临时替换n_step_buffer
                tmp = self.n_step_buffer
                self.n_step_buffer = deque(sub_buffer, maxlen=self.n_steps)
                if len(self.n_step_buffer) > 0:
                    s, a, G, s_n, d = self._compute_n_step_return()
                    self.replay_buffer.push(s, a, G, s_n, d)
                self.n_step_buffer = tmp
            # 清空n步缓存
            self.n_step_buffer.clear()

    def _project_distribution(self, next_probs, rewards, dones):
        """
        C51的核心：将目标分布投影到支撑点上
        
        这是Bellman算子在分布式RL中的实现：
        对每个支撑点 z_j：
          1. 计算Bellman目标: Tz_j = r + γ * z_j（对非终止状态）
          2. 裁剪到[v_min, v_max]
          3. 找到最近的支撑点，用线性插值分配概率
        
        Args:
            next_probs: 下一状态的目标分布, (batch, n_actions, n_atoms)
                        已经用Double DQN选好了动作
            rewards: n步回报, (batch,)
            dones: 是否结束, (batch,)
        Returns:
            m: 投影后的目标分布, (batch, n_atoms)
        """
        batch_size = rewards.size(0)

        # 把rewards和dones调整形状用于广播
        # rewards: (batch,) → (batch, 1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # self.online_net.atoms: (n_atoms,) → (1, n_atoms)（支撑点）
        atoms = self.online_net.atoms.unsqueeze(0)

        # === Bellman算子：把分布"向右移" ===
        # Tz_j = r + γ^n * z_j  （n步折扣）
        # 注意：我们用的是n步回报，所以折扣是 γ^n
        gamma_n = self.gamma ** self.n_steps

        # (1, n_atoms) 广播到 (batch, n_atoms)
        # done=True时，未来价值为0（r + γ^n * 0 = r）
        Tz = rewards + gamma_n * (1 - dones) * atoms

        # === 裁剪到[v_min, v_max] ===
        # 超出范围的直接截断（之后线性插值到边界原子）
        Tz = Tz.clamp(self.v_min, self.v_max)
        # Tz: (batch, n_atoms)

        # === 找到对应的支撑点索引 ===
        # b_j = (Tz_j - v_min) / delta_z  → 浮点索引
        # 比如 Tz_j = -5.0，v_min=-10，delta_z=0.4
        # b_j = (-5 - (-10)) / 0.4 = 12.5
        # → 落在第12和第13个原子之间，权重分别是0.5和0.5
        b = (Tz - self.v_min) / self.delta_z
        # b: (batch, n_atoms)

        # 下界和上界索引
        l = b.floor().long()  # 向下取整（下界原子）
        u = b.ceil().long()  # 向上取整（上界原子）

        # 处理边界情况：l可能=u（当b刚好是整数时）
        # 确保索引在有效范围内
        l = l.clamp(0, self.n_atoms - 1)
        u = u.clamp(0, self.n_atoms - 1)

        # === 线性插值，分配概率到目标分布 m ===
        # m 初始化为0
        m = torch.zeros(batch_size, self.n_atoms, device=self.device)

        # next_probs: (batch, n_atoms)（已经是选定动作的分布）
        # 分配权重：
        #   下界原子 l 获得 (u - b) * p_j 的概率  （距离越远权重越小）
        #   上界原子 u 获得 (b - l) * p_j 的概率

        # 为了用scatter_add，需要把索引展平处理
        # offset: 用于将batch中不同样本的索引分开
        # 形状: (batch, 1) 广播到 (batch, n_atoms)
        offset = (
                     torch.arange(batch_size, device=self.device)
                     .unsqueeze(1)
                     .expand(batch_size, self.n_atoms)
                 ) * self.n_atoms

        # 把m展平为1D，用scatter_add分配概率
        # scatter_add_(dim, index, src): m[index[i]] += src[i]
        m_flat = m.view(-1)

        # 分配给下界原子：权重 = u - b
        l_idx = (l + offset).view(-1)
        m_flat.scatter_add_(0, l_idx, (next_probs * (u.float() - b)).view(-1))

        # 分配给上界原子：权重 = b - l
        u_idx = (u + offset).view(-1)
        m_flat.scatter_add_(0, u_idx, (next_probs * (b - l.float())).view(-1))

        return m

    def _compute_loss(self, batch):
        """
        计算Rainbow DQN的损失
        
        使用C51的KL散度损失（而非普通DQN的MSE损失）
        
        整合了：Double DQN + C51 + PER权重
        
        Args:
            batch: (states, actions, rewards, next_states, dones, weights, tree_indices)
        Returns:
            loss: 标量损失值
            td_errors: 每个样本的TD误差（用于更新PER优先级）
        """
        states, actions, rewards, next_states, dones, weights, tree_indices = batch

        # 转为tensor并移到设备
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        batch_size = states.size(0)

        # ===================================================================
        # 计算目标分布（不需要梯度）
        # ===================================================================
        with torch.no_grad():
            # === Double DQN：用在线网络选动作 ===
            # 在线网络计算下一状态的期望Q值
            online_next_q = self.online_net.get_q_values(next_states)
            # online_next_q: (batch, n_actions)

            # 选择Q值最大的动作（在线网络决策）
            best_actions = online_next_q.argmax(dim=1)
            # best_actions: (batch,)，每个样本选的最优动作索引

            # === 用目标网络评估价值 ===
            # 目标网络计算下一状态的完整分布
            target_next_probs = self.target_net(next_states)
            # target_next_probs: (batch, n_actions, n_atoms)

            # 选择在线网络选定动作的分布
            # best_actions需要扩展以用于索引
            # best_actions: (batch,) → (batch, 1, n_atoms)
            best_actions_idx = best_actions.unsqueeze(1).unsqueeze(2).expand(
                batch_size, 1, self.n_atoms
            )
            # 用gather选出best_action对应的分布
            next_probs = target_next_probs.gather(1, best_actions_idx).squeeze(1)
            # next_probs: (batch, n_atoms) - 选定动作的目标分布

            # === 投影目标分布 ===
            # 这是C51的核心步骤，把 r + γ*Z(s',a*) 投影到支撑点上
            target_dist = self._project_distribution(next_probs, rewards, dones)
            # target_dist: (batch, n_atoms)，投影后的目标分布

        # ===================================================================
        # 计算在线网络的预测分布
        # ===================================================================
        # 在线网络前向传播
        pred_probs = self.online_net(states)
        # pred_probs: (batch, n_actions, n_atoms)

        # 选出实际执行动作对应的分布
        # actions: (batch,) → (batch, 1, n_atoms)
        actions_idx = actions.unsqueeze(1).unsqueeze(2).expand(
            batch_size, 1, self.n_atoms
        )
        pred_dist = pred_probs.gather(1, actions_idx).squeeze(1)
        # pred_dist: (batch, n_atoms) - 执行动作的预测分布

        # ===================================================================
        # 计算KL散度损失
        # ===================================================================
        # KL(target || pred) = Σ target_i * log(target_i / pred_i)
        #                    = Σ target_i * (log(target_i) - log(pred_i))
        #
        # 因为target是固定的，我们只需要最小化 -Σ target_i * log(pred_i)
        # 这就是交叉熵！
        #
        # 注意：pred_dist可能包含0（softmax输出可能极小），用clamp避免log(0)
        log_pred = pred_dist.clamp(min=1e-8).log()
        # cross_entropy: (batch, n_atoms) → sum → (batch,)
        elementwise_loss = -(target_dist * log_pred).sum(dim=1)
        # elementwise_loss: (batch,) - 每个样本的损失

        # === PER加权：重要性采样权重修正 ===
        # 高优先级样本被更频繁采样 → 需要更小的权重来修正偏差
        weighted_loss = (elementwise_loss * weights).mean()

        # === TD误差（用于更新PER优先级）===
        # 这里用cross entropy本身作为TD误差的代理
        td_errors = elementwise_loss.detach().cpu().numpy()

        return weighted_loss, td_errors, tree_indices

    def train_step(self):
        """
        执行一次训练步骤
        
        1. 从回放缓冲区采样
        2. 计算损失
        3. 反向传播更新网络
        4. 更新PER优先级
        5. 定期更新目标网络
        
        Returns:
            loss值（float），如果回放缓冲区不够大则返回None
        """
        # 检查回放缓冲区是否有足够的经验
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        # 只在指定频率训练
        if self.step_count % self.train_freq != 0:
            return None

        # === 重置噪声（Noisy Nets每步重新采样噪声）===
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        # === 采样批次 ===
        batch = self.replay_buffer.sample(self.batch_size)

        # === 计算损失 ===
        loss, td_errors, tree_indices = self._compute_loss(batch)

        # === 反向传播 ===
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度

        # 梯度裁剪：防止梯度爆炸（限制梯度范数最大为10）
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)

        self.optimizer.step()  # 更新参数

        # === 更新PER优先级 ===
        self.replay_buffer.update_priorities(tree_indices, td_errors)

        # === 定期更新目标网络 ===
        if self.step_count % self.target_update == 0:
            # 硬更新：直接复制在线网络的参数
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def select_action(self, state):
        """
        选择动作（Noisy Nets自带探索，无需epsilon-greedy）
        
        Args:
            state: numpy数组, (4, 84, 84)
        Returns:
            action: 整数
        """
        # 增加步数计数
        self.step_count += 1

        return self.online_net.act(state, self.device)

    def save(self, path):
        """保存模型"""
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'episode_count': self.episode_count,
        }, path)
        print(f"模型已保存到 {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint['step_count']
        self.episode_count = checkpoint['episode_count']
        print(f"模型已从 {path} 加载，已训练 {self.step_count} 步")


# 默认超参数配置
DEFAULT_CONFIG = {
    # 网络
    'n_atoms': 51,  # C51的原子数
    'v_min': -10.0,  # Q值分布下界（Pong奖励范围[-1,1]，折扣后约[-10,10]）
    'v_max': 10.0,  # Q值分布上界

    # 训练
    'gamma': 0.99,  # 折扣因子
    'lr': 6.25e-5,  # 学习率（Rainbow论文值）
    'batch_size': 32,  # 批次大小
    'n_steps': 3,  # multi-step return步数
    'train_freq': 4,  # 每4步训练一次（跟env的skip对齐）
    'target_update': 8000,  # 每8000步更新目标网络
    'min_replay_size': 20000,  # 开始训练前收集的经验量

    # 经验回放
    'buffer_size': 500000,  # 回放缓冲区大小（越大越好，受内存限制）
    'per_alpha': 0.5,  # PER优先级指数（0=均匀，1=纯优先）
    'per_beta': 0.4,  # PER重要性采样初始值
    'per_beta_frames': 1000000,  # beta线性增长到1.0所需步数
}
