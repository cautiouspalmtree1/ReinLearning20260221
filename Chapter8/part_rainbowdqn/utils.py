"""
utils.py — 工具函数：Atari环境预处理

Atari原始画面是 (210, 160, 3) 的RGB图像，帧率60fps
我们需要做以下预处理让学习更容易：
  1. 灰度化：减少输入维度
  2. 缩放到84x84：标准化输入大小
  3. 帧堆叠(4帧)：让网络感知运动信息（光靠一帧看不出球在飞）
  4. 跳帧(frameskip)：每4帧选一次动作，加速训练
  5. 归一化像素到[0,1]：稳定梯度
"""

import numpy as np
import gymnasium as gym
from collections import deque
import cv2  # OpenCV用于图像处理
import ale_py

class AtariPreprocessor(gym.Wrapper):
    """
    Atari环境预处理包装器
    
    gym.Wrapper是gymnasium提供的基类，让我们可以在原始env外面
    "套一层"来修改观察、奖励等，而不改动原始环境代码。
    """
    
    def __init__(self, env, frame_stack=4, frame_skip=4):
        """
        参数:
            env: 原始Atari gymnasium环境
            frame_stack: 堆叠多少帧（通常4帧）
            frame_skip: 每次动作重复多少帧（通常4帧）
        """
        super().__init__(env)
        self.frame_stack = frame_stack
        self.frame_skip = frame_skip
        
        # 用双端队列存储最近frame_stack帧，maxlen自动丢弃最旧的帧
        # deque([f1, f2, f3, f4]) → 新帧进来 → deque([f2, f3, f4, f5])
        self.frames = deque(maxlen=frame_stack)
        
        # 重新定义观察空间：(4, 84, 84) 的灰度帧堆叠
        # gym需要知道观察空间才能做各种检查
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(frame_stack, 84, 84),  # (C, H, W) PyTorch格式
            dtype=np.float32
        )
    
    def _preprocess_frame(self, frame):
        """
        处理单帧：RGB图 → 灰度图 → 缩放到84x84 → 归一化到[0,1]
        
        Args:
            frame: numpy数组，形状(210, 160, 3)，值范围[0, 255]
        Returns:
            processed: numpy数组，形状(84, 84)，值范围[0.0, 1.0]
        """
        # Step 1: 转为灰度图
        # cv2.cvtColor做颜色空间转换，输出形状: (210, 160)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Step 2: 缩放到84x84
        # cv2.INTER_AREA是缩小图像时最好的插值方法（取区域平均）
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Step 3: 归一化到[0, 1]，转为float32
        # uint8的[0,255] → float32的[0.0, 1.0]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def _get_observation(self):
        """
        把deque里的4帧拼成一个观察张量
        
        Returns:
            形状(4, 84, 84)的numpy数组，每个通道是一帧
        """
        # np.array(list(self.frames)) 把deque转成 (4, 84, 84)
        return np.array(list(self.frames), dtype=np.float32)
    
    def reset(self, **kwargs):
        """
        重置环境（每个episode开始时调用）
        需要初始化帧缓存
        """
        # 调用原始环境的reset，获得初始帧
        obs, info = self.env.reset(**kwargs)
        
        # 预处理初始帧
        processed = self._preprocess_frame(obs)
        
        # 用同一帧填满4个槽位（冷启动，没有历史帧）
        for _ in range(self.frame_stack):
            self.frames.append(processed)
        
        return self._get_observation(), info
    
    def step(self, action):
        """
        执行动作，实现跳帧逻辑
        
        跳帧：执行同一个动作frame_skip次，累加奖励
        这样做的好处：减少决策频率，降低计算量，每个动作影响更明显
        
        同时处理"max-pooling over 2 frames"：
        取最后两帧的像素最大值，避免某些游戏中精灵闪烁导致看不到
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        
        # 保存最后两帧的原始画面（用于max-pooling）
        last_two_frames = []
        last_two_frames_fallback = []
        
        for i in range(self.frame_skip):
            # 执行动作一步
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            total_reward += reward  # 累加奖励
            
            # 收集最后两帧（用于处理精灵闪烁）
            if i >= self.frame_skip - 2:
                last_two_frames.append(obs)
            last_two_frames_fallback.append(obs)
            
            # 如果episode结束，提前退出
            if terminated or truncated:
                break
        
        # Max-pooling：对最后两帧取像素最大值
        # 某些Atari游戏的精灵每隔一帧才渲染，取最大值确保都能看到
        if len(last_two_frames) == 2:
            max_frame = np.maximum(last_two_frames[0], last_two_frames[1])
        elif len(last_two_frames) == 1:
            max_frame = last_two_frames[0]
        elif len(last_two_frames) == 0:
            max_frame = last_two_frames_fallback[-1]
        
        # 预处理并加入帧队列
        processed = self._preprocess_frame(max_frame)
        self.frames.append(processed)
        
        # 奖励裁剪(reward clipping)：将奖励限制在[-1, 1]
        # 让不同Atari游戏的奖励规模统一，稳定训练
        clipped_reward = np.sign(total_reward)  # -1, 0, 或 1
        
        return self._get_observation(), clipped_reward, terminated, truncated, info


def make_env(env_name="ALE/Pong-v5", render_mode=None):
    """
    创建预处理好的Atari环境
    
    Args:
        env_name: 环境名称，Pong用"ALE/Pong-v5"
        render_mode: None(训练时)或"human"(观看时)
    Returns:
        包装好的环境
    """
    # 创建原始Atari环境
    # full_action_space=False 只使用游戏相关的动作（Pong有6个，但有效的只有3个）
    gym.register_envs(ale_py)
    env = gym.make(
        env_name,
        render_mode=render_mode,
        full_action_space=False,
    )
    
    # 套上我们的预处理包装器
    env = AtariPreprocessor(env, frame_stack=4, frame_skip=4)
    
    return env


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("测试Atari环境预处理...")
    
    env = make_env("ALE/Pong-v5")
    obs, info = env.reset()
    
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"初始观察形状: {obs.shape}")        # 应该是 (4, 84, 84)
    print(f"观察值范围: [{obs.min():.2f}, {obs.max():.2f}]")  # 应该在[0,1]
    
    # 执行几步
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"动作: {action}, 奖励: {reward}, 结束: {terminated or truncated}")
    
    env.close()
    print("环境测试通过！")
