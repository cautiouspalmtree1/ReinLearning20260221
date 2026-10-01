import os

import gymnasium as gym
import time
import ale_py
import torch

from Chapter8.part8_4.DQNAgent import DQNAgent

# render_mode="human" 会弹出一个实时窗口展示画面
# "ALE/Pong-v5" 是经典乒乓球游戏
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")

# 推理过程
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device: " + str(device))
dqn_agent = DQNAgent(device=device)
RESUME_PATH = "checkpoints/ckpt_ep1350.pth"  # 改成具体路径即可续训，如 "checkpoints/ckpt_ep200.pth"
if RESUME_PATH and os.path.exists(RESUME_PATH):
    start_episode, total_steps = dqn_agent.load_checkpoint(RESUME_PATH)
dqn_agent.epsilon = 0.0

done = False
total_reward = 0.0
for epoch in range(300):
    print("Epoch: ", epoch)
    state, info = env.reset()
    env.step(1)
    done = False
    # 随机选择一个动作
    # action = env.action_space.sample()
    # 修改为DQN选择动作
    epoch_reward = 0.0
    dqn_agent.frame_stack = []

    while not done:
        # 显示画面
        env.render()
        # 获取动作
        # action = env.action_space.sample()
        # 获取动作
        action = dqn_agent.get_action_inference(dqn_agent.image_pre_process(state).unsqueeze(0))
        # action = 3
        # 执行动作
        observation, reward, terminated, truncated, _ = env.step(action)
        if reward != 0.0:
            print(reward)
        state = observation
        done = terminated or truncated
        epoch_reward += reward

    total_reward += epoch_reward
    aval_reward = total_reward / (epoch + 1)
    print("Epoch: {}, Reward: {}".format(epoch, aval_reward))

env.close()
