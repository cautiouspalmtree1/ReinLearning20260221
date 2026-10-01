# Rainbow DQN — Atari Pong 教学实现

## 📁 文件结构

```
rainbow_dqn/
├── utils.py          # Atari环境预处理（灰度、缩放、帧堆叠）
├── replay_buffer.py  # 优先经验回放 + 线段树
├── model.py          # Rainbow网络（NoisyLinear + Dueling + C51）
├── agent.py          # 训练器（Double DQN + 损失计算）
├── train.py          # 主训练脚本
└── requirements.txt  # 依赖
```

## 🌈 Rainbow = DQN + 6个改进

| 改进 | 作用 | 实现位置 |
|------|------|----------|
| Double DQN | 消除Q值高估 | `agent.py` |
| Dueling Network | 分离V(s)和A(s,a) | `model.py` |
| Prioritized Replay | 重要经验多学 | `replay_buffer.py` |
| Multi-step Returns | 奖励传播更快 | `agent.py` |
| Noisy Nets | 参数化噪声探索 | `model.py` |
| Distributional RL | 学Q值分布而非期望 | `model.py` + `agent.py` |

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 开始训练（GPU推荐）
python train.py

# 3. 减小内存用量（RAM < 8GB）
python train.py --buffer 100000

# 4. 观看训练好的模型
python train.py --watch
```

## ⏱️ 训练时间参考

| 硬件 | 训练至打败Pong AI |
|------|------------------|
| RTX 3080 | ~4-6小时 |
| GTX 1080 | ~8-12小时 |
| CPU | ~3-5天（不推荐）|

## 📊 训练指标解读

- **奖励范围**：Pong的奖励为[-1, 1]（裁剪后），每局最高21分
- **目标**：平均奖励 > 18 表示完全打败内置AI
- **早期**（前50万步）：平均奖励约-20（一直输）
- **中期**（50-100万步）：开始学会防守，奖励上升
- **后期**（100-200万步）：稳定打败AI

## 🔧 超参数调节

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_atoms` | 51 | C51原子数，越多分布越精细 |
| `v_min/v_max` | -10/10 | Q值分布范围 |
| `gamma` | 0.99 | 折扣因子 |
| `lr` | 6.25e-5 | 学习率 |
| `n_steps` | 3 | Multi-step步数（3-5都行）|
| `buffer_size` | 500000 | 内存不够可降到100000 |
| `per_alpha` | 0.5 | PER优先级强度 |

## 🧠 学习路径建议

1. 先读 `utils.py`：理解Atari预处理
2. 再读 `replay_buffer.py`：理解线段树和PER
3. 再读 `model.py`：理解NoisyLinear和Dueling+C51网络
4. 最后读 `agent.py`：理解损失函数和训练流程
5. `train.py` 是胶水代码，把以上串联起来

## 📚 参考论文

- Rainbow DQN: [Hessel et al., 2017](https://arxiv.org/abs/1710.02298)
- C51: [Bellemare et al., 2017](https://arxiv.org/abs/1707.06887)
- Noisy Nets: [Fortunato et al., 2017](https://arxiv.org/abs/1706.10295)
- PER: [Schaul et al., 2015](https://arxiv.org/abs/1511.05952)
- Dueling DQN: [Wang et al., 2015](https://arxiv.org/abs/1511.06581)
- Double DQN: [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)
- Multi-step: [Sutton & Barto, RL An Introduction](http://incompleteideas.net/book/the-book.html)
