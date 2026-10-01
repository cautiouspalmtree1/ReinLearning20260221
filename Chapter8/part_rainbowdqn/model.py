"""
model.py — Rainbow DQN 神经网络

融合了以下三个网络改进：
1. Noisy Nets（噪声网络）：用参数化噪声代替epsilon-greedy探索
2. Dueling Network（对决网络）：分别估计状态价值V(s)和动作优势A(s,a)
3. Distributional RL / C51：不学Q值期望，而是学Q值的概率分布

【网络架构】
输入: (4, 84, 84) 堆叠帧
  ↓
卷积层（提取视觉特征）
  ↓
全连接层（NoisyLinear代替普通Linear）
  ↓
Dueling分支:
  ├── 价值流 V(s): 标量，这个状态有多好
  └── 优势流 A(s,a): 每个动作相对平均的优势
  ↓
Q分布: Q(s,a,z) = V(s,z) + A(s,a,z) - mean(A)
  z是支撑点（把Q值范围[-10,10]离散成51个点）

【Noisy Nets详解】
普通线性层: y = Wx + b  （W和b是固定参数）
噪声线性层: y = (μ_w + σ_w ⊙ ε_w)x + (μ_b + σ_b ⊙ ε_b)
            其中 ε 是随机噪声，μ和σ是学习的参数
            
好处：噪声强度(σ)由网络自己学习调整，
      不需要手动调节ε-greedy的衰减计划

【C51/Distributional详解】
普通DQN: Q(s,a) = 期望回报（一个数）
C51 DQN: Q(s,a) = 回报的概率分布（51个概率）

为什么更好？
- 保留了更多信息（分布 vs 期望）
- 训练更稳定（分布的KL散度 vs TD误差的MSE）
- 能区分"期望相同但风险不同"的动作
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    """
    噪声线性层（Noisy Network）
    
    实现方式：Factorized Gaussian Noise（因子化高斯噪声）
    
    标准高斯噪声需要 p*q 个随机数（p输入，q输出）
    因子化方法：只需 p+q 个随机数，然后外积得到 p×q 的噪声矩阵
    
    ε^w_ij = f(ε_i) * f(ε_j)  （两个1D噪声向量的外积）
    ε^b_j  = f(ε_j)
    
    其中 f(x) = sgn(x) * sqrt(|x|)  （保持符号，压缩幅度）
    """
    
    def __init__(self, in_features, out_features, sigma_init=0.5):
        """
        Args:
            in_features: 输入特征数
            out_features: 输出特征数
            sigma_init: 噪声标准差的初始值（论文推荐0.5/sqrt(p)）
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # ---- 可学习参数 ----
        # 均值参数（类似普通线性层的W和b）
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_mu = nn.Parameter(
            torch.empty(out_features)
        )
        
        # 标准差参数（学习噪声的幅度，σ越大=探索越多）
        self.weight_sigma = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias_sigma = nn.Parameter(
            torch.empty(out_features)
        )
        
        # ---- 噪声张量（不参与梯度，每次前向传播重新采样）----
        # register_buffer: 不是参数，但会随模型一起保存/移动到GPU
        self.register_buffer(
            'weight_epsilon', torch.empty(out_features, in_features)
        )
        self.register_buffer(
            'bias_epsilon', torch.empty(out_features)
        )
        
        # 初始化参数
        self._reset_parameters()
        # 初始化噪声
        self.reset_noise()
    
    def _reset_parameters(self):
        """
        初始化均值和标准差参数
        
        均值：均匀分布初始化（类似kaiming初始化）
        标准差：全部初始化为 sigma_init / sqrt(in_features)
        """
        # 均值参数初始化范围
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        # 标准差参数：论文推荐值
        sigma_val = self.sigma_init / np.sqrt(self.in_features)
        self.weight_sigma.data.fill_(sigma_val)
        self.bias_sigma.data.fill_(sigma_val)
    
    @staticmethod
    def _scale_noise(size):
        """
        生成因子化噪声向量
        
        f(x) = sgn(x) * sqrt(|x|)
        
        为什么用这个函数？保持噪声的符号，同时压缩幅度，
        让噪声有更好的统计特性
        
        Args:
            size: 噪声向量长度
        Returns:
            变换后的噪声向量
        """
        x = torch.randn(size)                   # 标准正态分布
        return x.sign() * x.abs().sqrt()        # f(x) = sgn(x)*sqrt(|x|)
    
    def reset_noise(self):
        """
        重新采样噪声（每次前向传播前调用，或每个episode结束时）
        
        因子化方法：只采样 p+q 个数，再算外积得到 p×q 矩阵
        """
        # 为输入和输出各生成一个噪声向量
        epsilon_in  = self._scale_noise(self.in_features)    # 长度 p
        epsilon_out = self._scale_noise(self.out_features)   # 长度 q
        
        # 外积得到权重噪声矩阵 (q, p)
        # ger = outer product（外积）
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        
        # 偏置噪声直接用输出噪声向量
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        """
        前向传播
        
        训练时：使用噪声权重 (μ + σ⊙ε)
        评估时：只用均值 μ（.eval()模式下也可以，但本实现训练评估都用噪声）
        
        Args:
            x: 输入张量 (..., in_features)
        Returns:
            输出张量 (..., out_features)
        """
        # 有效权重 = 均值 + 标准差 × 噪声
        # weight_sigma * weight_epsilon: 元素级乘法 (hadamard product)
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        
        # 普通的线性变换（用的是噪声权重）
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN 网络
    
    融合：Noisy Nets + Dueling Network + Distributional (C51)
    
    输出：(batch, n_actions, n_atoms)
         对每个动作，输出在n_atoms个支撑点上的概率分布
    """
    
    def __init__(self, n_actions, n_atoms=51, v_min=-10.0, v_max=10.0):
        """
        Args:
            n_actions: 动作数量（Pong是6）
            n_atoms: 分布的离散化数量（论文用51，所以叫C51）
            v_min: Q值分布的最小支撑点（预期最低回报）
            v_max: Q值分布的最大支撑点（预期最高回报）
        """
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # 支撑点 z：在[v_min, v_max]间均匀分布n_atoms个点
        # 每个原子代表"Q值等于z_i"的可能性
        # register_buffer: 不是参数，随模型保存
        self.register_buffer(
            'atoms',
            torch.linspace(v_min, v_max, n_atoms)  # 形状: (n_atoms,)
        )
        
        # =========================================
        # 卷积特征提取器（跟Nature DQN一样）
        # 输入: (batch, 4, 84, 84)
        # =========================================
        self.conv = nn.Sequential(
            # 第一层卷积: 4帧 → 32个特征图
            # 8×8的大卷积核用于捕获大范围特征（游戏中的球、球拍）
            # stride=4大步长，快速缩小空间维度
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # 第二层卷积: 32 → 64个特征图
            # 4×4卷积核，继续提取中级特征
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # 第三层卷积: 64 → 64个特征图
            # 3×3卷积核，提取细粒度特征
            # stride=1不再缩小
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 卷积后的特征图大小: (batch, 64, 7, 7)
        # 展平: 64 * 7 * 7 = 3136
        
        conv_out_size = 64 * 7 * 7  # = 3136
        hidden_size = 512
        
        # =========================================
        # Dueling Network 分支（都用NoisyLinear）
        # =========================================
        
        # --- 价值流 (Value Stream) ---
        # 估计 V(s, z)：这个状态有多好，与动作无关
        # 输出: (batch, n_atoms) — 状态价值在各原子上的"分量"
        self.value_hidden = NoisyLinear(conv_out_size, hidden_size)
        self.value_out    = NoisyLinear(hidden_size, n_atoms)
        
        # --- 优势流 (Advantage Stream) ---
        # 估计 A(s, a, z)：动作a相对平均水平的优势
        # 输出: (batch, n_actions, n_atoms) — 每个动作×每个原子
        self.advantage_hidden = NoisyLinear(conv_out_size, hidden_size)
        self.advantage_out    = NoisyLinear(hidden_size, n_actions * n_atoms)
    
    def forward(self, x):
        """
        前向传播：输入状态，输出每个动作的Q值概率分布
        
        Args:
            x: 输入状态, 形状 (batch, 4, 84, 84), 值[0,1]
        Returns:
            probs: 概率分布, 形状 (batch, n_actions, n_atoms)
                   probs[b, a, i] = 在状态x[b]下，执行动作a，
                                    回报等于atoms[i]的概率
        """
        batch_size = x.size(0)
        
        # === Step 1: 卷积提取特征 ===
        # x: (batch, 4, 84, 84)
        conv_out = self.conv(x)
        # conv_out: (batch, 64, 7, 7)
        
        # 展平卷积输出
        # view(-1, ...) 中 -1 会自动推断批次维度
        features = conv_out.view(batch_size, -1)
        # features: (batch, 3136)
        
        # === Step 2: 价值流 ===
        # 经过两层NoisyLinear，输出n_atoms个"价值原子"
        value = F.relu(self.value_hidden(features))
        # value: (batch, 512)
        
        value = self.value_out(value)
        # value: (batch, n_atoms)
        
        # 调整形状为 (batch, 1, n_atoms)，方便后面广播加法
        value = value.view(batch_size, 1, self.n_atoms)
        
        # === Step 3: 优势流 ===
        advantage = F.relu(self.advantage_hidden(features))
        # advantage: (batch, 512)
        
        advantage = self.advantage_out(advantage)
        # advantage: (batch, n_actions * n_atoms)
        
        # 重塑为 (batch, n_actions, n_atoms)
        advantage = advantage.view(batch_size, self.n_actions, self.n_atoms)
        
        # === Step 4: Dueling合并 ===
        # Q(s,a,z) = V(s,z) + A(s,a,z) - mean_a[A(s,a,z)]
        #
        # 减去平均优势的原因：
        # 如果不减，V和A之间没有唯一分解（V+c, A-c给出相同Q值）
        # 减去均值后，A的均值=0，V就是"真正的"状态价值
        #
        # mean(A) 在 dim=1（动作维度）上求平均
        # keepdim=True 保持维度，方便广播
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        # q_atoms: (batch, n_actions, n_atoms)
        
        # === Step 5: Softmax得到概率分布 ===
        # 对每个(状态, 动作)的n_atoms个logit做softmax
        # dim=2表示在原子维度（最后一维）做softmax
        probs = F.softmax(q_atoms, dim=2)
        # probs: (batch, n_actions, n_atoms)
        # 约束: probs[b, a, :].sum() == 1.0
        
        return probs
    
    def get_q_values(self, x):
        """
        从概率分布计算期望Q值（用于选择动作）
        
        E[Q(s,a)] = Σ_i z_i * p_i  （期望值 = 支撑点 × 概率 的加权和）
        
        Args:
            x: 输入状态, 形状 (batch, 4, 84, 84)
        Returns:
            q_values: 期望Q值, 形状 (batch, n_actions)
        """
        # 获取概率分布
        probs = self.forward(x)
        # probs: (batch, n_actions, n_atoms)
        
        # self.atoms: (n_atoms,) → 扩展到 (1, 1, n_atoms) 方便广播
        atoms = self.atoms.unsqueeze(0).unsqueeze(0)
        
        # 期望值 = Σ p_i * z_i，对原子维度(dim=2)求和
        q_values = (probs * atoms).sum(dim=2)
        # q_values: (batch, n_actions)
        
        return q_values
    
    def act(self, state, device):
        """
        根据当前状态选择动作（贪婪地选期望Q值最大的动作）
        
        Noisy Nets自带探索，不需要epsilon-greedy！
        
        Args:
            state: numpy数组, 形状 (4, 84, 84)
            device: torch设备
        Returns:
            action: 整数，选择的动作
        """
        # numpy → tensor，增加batch维度，移到设备
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # 不需要计算梯度（推理模式）
        with torch.no_grad():
            q_values = self.get_q_values(state_tensor)
        
        # 选择Q值最大的动作
        action = q_values.argmax(dim=1).item()
        return action
    
    def reset_noise(self):
        """
        重新采样所有NoisyLinear层的噪声
        通常在每个step开始时调用
        """
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_out.reset_noise()


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("测试 Rainbow DQN 网络...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建网络（Pong有6个动作）
    model = RainbowDQN(n_actions=6, n_atoms=51, v_min=-10, v_max=10).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 测试前向传播
    batch = torch.randn(4, 4, 84, 84).to(device)  # 4个样本
    
    probs = model(batch)
    print(f"概率分布输出形状: {probs.shape}")  # (4, 6, 51)
    print(f"概率和（应该≈1）: {probs[0, 0, :].sum():.4f}")
    
    q_values = model.get_q_values(batch)
    print(f"Q值形状: {q_values.shape}")  # (4, 6)
    print(f"Q值范围: [{q_values.min():.2f}, {q_values.max():.2f}]")
    
    # 测试动作选择
    state = np.random.rand(4, 84, 84).astype(np.float32)
    action = model.act(state, device)
    print(f"选择的动作: {action}")
    
    # 测试噪声重置
    model.reset_noise()
    print("噪声重置成功！")
    
    print("网络测试通过！")
