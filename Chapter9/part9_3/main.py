import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Chapter9.part9_3.PolicyAgent import PolicyAgent
import torch

"""
Actor Critic  算法
"""

env = gym.make("CartPole-v1")
policy_agent = PolicyAgent(gamma=0.9, alpha=0.1, input_size=4, hidden_size=512, output_size=2)
policy_agent.init_plot()  # 初始化画布

total_steps = 0
episodes = 10000
episode_avg_reward = 0.0
for episode in range(episodes):
    state, info = env.reset()
    total_reward = 0
    total_loss = 0
    done = False

    while not done:
        total_steps += 1
        # 这里建议用 agent.get_action(state)，而不是随机，否则学不到东西
        # action = np.random.choice(0, 1)
        action, prob = policy_agent.get_action(torch.tensor(state, dtype=torch.float32))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 存入并更新
        loss = policy_agent.update_policy_net(state, action, reward, next_state, done, prob)

        total_reward += reward
        state = next_state
        if loss is not None:
            total_loss += loss

    # Episode 结束后的处理
    policy_agent.reward_history.append((episode, total_reward))
    # 计算所有 episode 的平均奖励
    if len(policy_agent.reward_history) < 100:
        total_episode_rewards = sum([reward for _, reward in policy_agent.reward_history])
        episode_avg_reward = total_episode_rewards / len(policy_agent.reward_history)
    elif len(policy_agent.reward_history) >= 100:
        total_episode_rewards = sum([reward for _, reward in policy_agent.reward_history[-100:]])
        episode_avg_reward = total_episode_rewards / 100

    print(f'Episode: {episode}, Reward: {total_reward}, episode_avg_reward: {episode_avg_reward:.2f}')
    if total_loss is not None:
        print(f'Loss: {total_loss / total_steps:.6f}')

    # 关键：每隔几个 Episode 刷新一次图表，不要每个 Step 都刷，太慢
    if episode % 50 == 0:
        policy_agent.update_plot()
        plt.pause(0.1)  # 给 GUI 呼吸的时间

plt.ioff()
plt.show()  # 训练完不关闭，保持显示
env.close()