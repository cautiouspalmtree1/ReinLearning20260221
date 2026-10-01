"""
train.py — Rainbow DQN 训练主脚本

运行方式：
  python train.py              # 开始训练
  python train.py --watch      # 观看已训练的模型玩游戏

训练过程：
  1. 初始化环境和智能体
  2. 收集足够的初始经验（random warm-up）
  3. 循环：
     a. 选择动作（Noisy Nets探索）
     b. 执行动作，收集经验
     c. 存入回放缓冲区（multi-step return）
     d. 采样batch，计算损失，更新网络
     e. 定期更新目标网络
     f. 记录训练统计信息

硬件需求：
  - GPU: 强烈推荐（CPU训练极慢）
  - 内存: 8GB+（回放缓冲区较大）
  - 时间: 在GPU上约需4-10小时才能打败Pong AI
"""

import os
import time
import argparse
import numpy as np
import torch
from collections import deque

from utils import make_env
from agent import RainbowAgent, DEFAULT_CONFIG


def train(config=None):
    """
    主训练循环
    
    Args:
        config: 超参数字典，None则使用默认配置
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    print("=" * 60)
    print("Rainbow DQN — Atari Pong 训练")
    print("=" * 60)
    print("超参数:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    
    # ====================================================
    # 1. 初始化环境
    # ====================================================
    env = make_env("ALE/Pong-v5", render_mode=None)
    n_actions = env.action_space.n
    print(f"动作数量: {n_actions}")
    print(f"观察空间: {env.observation_space.shape}")
    
    # ====================================================
    # 2. 初始化智能体
    # ====================================================
    agent = RainbowAgent(n_actions=n_actions, config=config)
    
    # 创建保存目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # ====================================================
    # 3. 训练统计变量
    # ====================================================
    episode_rewards = []          # 每个episode的总奖励
    recent_rewards = deque(maxlen=100)  # 最近100个episode的奖励（用于计算平均）
    episode_lengths = []          # 每个episode的步数
    losses = []                   # 训练损失
    
    best_mean_reward = -float('inf')  # 最佳平均奖励（用于保存最好的模型）
    
    # 训练开始时间
    start_time = time.time()
    
    # ====================================================
    # 4. 训练主循环
    # ====================================================
    total_steps = config.get('total_steps', 2_000_000)  # 总训练步数
    
    print(f"\n开始训练，目标 {total_steps:,} 步...")
    print(f"首先收集 {config['min_replay_size']:,} 步经验再开始学习...\n")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_num = 0
    
    for step in range(1, total_steps + 1):
        # ------------------------------------------------
        # 4a. 选择并执行动作
        # ------------------------------------------------
        # 回放缓冲区不够时随机探索（warm-up阶段）
        if len(agent.replay_buffer) < config['min_replay_size']:
            action = env.action_space.sample()
            agent.step_count += 1  # 也要计步
        else:
            # 使用Noisy Nets的贪婪策略（网络内部有探索噪声）
            action = agent.select_action(obs)
        
        # 执行动作，获取下一状态
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # ------------------------------------------------
        # 4b. 存储经验（multi-step return处理在这里）
        # ------------------------------------------------
        agent.push_experience(obs, action, reward, next_obs, done)
        
        # 更新当前观察
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        # ------------------------------------------------
        # 4c. 如果episode结束，重置环境
        # ------------------------------------------------
        if done:
            obs, _ = env.reset()
            episode_num += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            recent_rewards.append(episode_reward)
            
            mean_reward = np.mean(recent_rewards)
            
            # 打印每个episode的统计
            elapsed = time.time() - start_time
            fps = step / elapsed
            print(
                f"Episode {episode_num:4d} | "
                f"步数 {step:8,} | "
                f"奖励 {episode_reward:6.1f} | "
                f"平均(100) {mean_reward:6.1f} | "
                f"缓冲区 {len(agent.replay_buffer):6,} | "
                f"FPS {fps:.0f}"
            )
            
            # 保存最好的模型
            if mean_reward > best_mean_reward and len(recent_rewards) >= 10:
                best_mean_reward = mean_reward
                agent.save("checkpoints/rainbow_best.pt")
            
            # 重置episode统计
            episode_reward = 0
            episode_length = 0
        
        # ------------------------------------------------
        # 4d. 训练（采样并更新网络）
        # ------------------------------------------------
        loss = agent.train_step()
        if loss is not None:
            losses.append(loss)
        
        # ------------------------------------------------
        # 4e. 定期详细日志
        # ------------------------------------------------
        if step % 50000 == 0:
            mean_loss = np.mean(losses[-1000:]) if losses else 0
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"【训练进度】步数: {step:,} / {total_steps:,}")
            print(f"  Episode数: {episode_num}")
            print(f"  最近100ep平均奖励: {np.mean(recent_rewards) if recent_rewards else 0:.2f}")
            print(f"  最近1000步平均损失: {mean_loss:.4f}")
            print(f"  已用时间: {elapsed/3600:.1f}小时")
            print(f"  PER beta: {agent.replay_buffer.beta:.3f}")
            print(f"{'='*60}\n")
        
        # ------------------------------------------------
        # 4f. 定期保存检查点
        # ------------------------------------------------
        if step % 200000 == 0:
            agent.save(f"checkpoints/rainbow_step_{step}.pt")
    
    # ====================================================
    # 5. 训练结束
    # ====================================================
    env.close()
    agent.save("checkpoints/rainbow_final.pt")
    
    print("\n训练完成！")
    print(f"最佳平均奖励: {best_mean_reward:.2f}")
    print(f"最终平均奖励(最近100ep): {np.mean(recent_rewards):.2f}")


def watch(model_path="checkpoints/rainbow_best.pt", config=None, n_episodes=5):
    """
    观看训练好的模型玩游戏
    
    Args:
        model_path: 模型文件路径
        config: 超参数配置
        n_episodes: 观看多少局
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    print(f"加载模型: {model_path}")
    
    # 创建带渲染的环境
    env = make_env("ALE/Pong-v5", render_mode="human")
    n_actions = env.action_space.n
    
    agent = RainbowAgent(n_actions=n_actions, config=config)
    agent.load(model_path)
    
    # 评估模式（关闭dropout等，但NoisyNet仍有噪声）
    agent.online_net.eval()
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.online_net.act(obs, agent.device)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            time.sleep(0.02)  # 降速，人眼可以看清楚
        
        print(f"Episode {ep+1}: 奖励 = {episode_reward:.1f}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rainbow DQN - Atari Pong")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="观看模型玩游戏（需要先训练）"
    )
    parser.add_argument(
        "--model",
        default="checkpoints/rainbow_best.pt",
        help="要加载的模型路径"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2_000_000,
        help="训练总步数（默认2M）"
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=500_000,
        help="回放缓冲区大小（内存不足可降低，如100000）"
    )
    args = parser.parse_args()
    
    # 根据命令行参数修改配置
    config = DEFAULT_CONFIG.copy()
    config['total_steps'] = args.steps
    config['buffer_size'] = args.buffer
    
    if args.watch:
        watch(model_path=args.model, config=config)
    else:
        train(config=config)
