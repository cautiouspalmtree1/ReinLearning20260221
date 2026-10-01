import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn, optim

from PolicyNet import PolicyNet


class PolicyAgent:
    def __init__(self, gamma, input_size, hidden_size, output_size):
        self.ax = None
        self.fig = None
        self.line = None
        self.gamma = gamma
        self.policy_net = PolicyNet(input_size, hidden_size, output_size)
        self.buffer = []
        self.reward_history = []

        # 定义损失函数和优化器
        self.criterion = nn.MSELoss()  # 均方误差损失
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

    def get_action(self, state):
        prob = self.policy_net.forward(state)
        action = torch.multinomial(prob, 1).item()
        return action, prob[action]

    def update_policy_net(self, reward, done, prob):
        data = (reward, prob)
        self.buffer.append(data)
        G = torch.tensor(0.0)
        loss = torch.tensor(0.0)

        if done:
            for reward, prob in reversed(self.buffer):
                G = reward + self.gamma * G

            for reward, prob in self.buffer:
                loss += - torch.log(prob) * G
            self.policy_net.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.buffer = []
        return loss.item()
    
    def init_plot(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.line, = self.ax.plot([], [], label='Training Reward', color='#1f77b4')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.legend()

    def update_plot(self):
        if len(self.reward_history) > 0:
            episodes, rewards = zip(*self.reward_history)
            self.line.set_data(episodes, rewards)
            self.ax.relim()
            self.ax.autoscale_view()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()








