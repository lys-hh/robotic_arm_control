"""
强化学习智能体实现
主要使用stable-baselines3，同时提供自定义实现作为备选
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
import os

# 尝试导入stable-baselines3
try:
    from stable_baselines3 import PPO, TD3, SAC
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Stable-Baselines3 可用")
except ImportError:
    SB3_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  Stable-Baselines3 不可用，将使用自定义实现")

class RLAgent:
    """
    强化学习智能体基类
    优先使用stable-baselines3，备选自定义实现
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 algorithm: str = "PPO",
                 use_sb3: bool = True,
                 **kwargs):
        """
        初始化智能体
        
        Args:
            input_dim: 状态空间维度
            output_dim: 动作空间维度
            algorithm: 算法类型 ("PPO", "TD3", "SAC")
            use_sb3: 是否使用stable-baselines3
            **kwargs: 其他参数
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.algorithm = algorithm
        self.use_sb3 = use_sb3 and SB3_AVAILABLE
        
        if self.use_sb3:
            logger.info(f"使用Stable-Baselines3的{algorithm}算法")
            self._init_sb3_agent(**kwargs)
        else:
            logger.info(f"使用自定义{algorithm}算法")
            self._init_custom_agent(**kwargs)
    
    def _init_sb3_agent(self, **kwargs):
        """初始化Stable-Baselines3智能体"""
        # 这里只是占位，实际初始化在set_environment中
        self.sb3_agent = None
        self.env = None
        
    def _init_custom_agent(self, **kwargs):
        """初始化自定义智能体"""
        if self.algorithm == "PPO":
            self.agent = CustomPPOAgent(self.input_dim, self.output_dim, **kwargs)
        elif self.algorithm == "TD3":
            self.agent = CustomTD3Agent(self.input_dim, self.output_dim, **kwargs)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
    
    def set_environment(self, env):
        """
        设置训练环境
        
        Args:
            env: Gymnasium环境或VecEnv
        """
        self.env = env
        
        if self.use_sb3:
            self._init_sb3_agent_with_env(env)
    
    def _init_sb3_agent_with_env(self, env):
        """使用环境初始化SB3智能体"""
        if self.algorithm == "PPO":
            self.sb3_agent = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                tensorboard_log="./logs/"
            )
        elif self.algorithm == "TD3":
            self.sb3_agent = TD3(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=1000000,
                learning_starts=100,
                batch_size=100,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                tensorboard_log="./logs/"
            )
        elif self.algorithm == "SAC":
            self.sb3_agent = SAC(
                "MlpPolicy",
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=1000000,
                learning_starts=100,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                tensorboard_log="./logs/"
            )
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")
    
    def train(self, total_timesteps: int, **kwargs):
        """
        训练智能体
        
        Args:
            total_timesteps: 总训练步数
            **kwargs: 其他训练参数
        """
        if self.use_sb3:
            if self.sb3_agent is None:
                raise ValueError("请先调用set_environment设置环境")
            self.sb3_agent.learn(total_timesteps=total_timesteps, **kwargs)
        else:
            self.agent.train(self.env, total_timesteps, **kwargs)
    
    def predict(self, observation: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        预测动作
        
        Args:
            observation: 观察
            deterministic: 是否确定性动作
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: (动作, 状态)
        """
        if self.use_sb3:
            if self.sb3_agent is None:
                raise ValueError("请先调用set_environment设置环境")
            return self.sb3_agent.predict(observation, deterministic=deterministic)
        else:
            action = self.agent.get_action(observation)
            return action, None
    
    def save(self, filepath: str):
        """保存模型"""
        if self.use_sb3:
            if self.sb3_agent is None:
                raise ValueError("请先调用set_environment设置环境")
            self.sb3_agent.save(filepath)
        else:
            self.agent.save(filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        if self.use_sb3:
            if self.algorithm == "PPO":
                self.sb3_agent = PPO.load(filepath)
            elif self.algorithm == "TD3":
                self.sb3_agent = TD3.load(filepath)
            elif self.algorithm == "SAC":
                self.sb3_agent = SAC.load(filepath)
        else:
            self.agent.load(filepath)
    
    def get_info(self) -> Dict:
        """获取智能体信息"""
        info = {
            "algorithm": self.algorithm,
            "use_sb3": self.use_sb3,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim
        }
        
        if self.use_sb3 and self.sb3_agent is not None:
            info["sb3_info"] = {
                "policy": str(self.sb3_agent.policy),
                "learning_rate": self.sb3_agent.learning_rate,
                "gamma": self.sb3_agent.gamma
            }
        
        return info


# ==================== 自定义实现 (备选方案) ====================

class CustomPPOAgent:
    """自定义PPO智能体实现 (备选方案)"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 创建网络
        self.policy_network = PolicyNetwork(input_dim, output_dim, hidden_dim)
        self.value_network = ValueNetwork(input_dim, hidden_dim)
        
        # 优化器
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=3e-4)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=3e-4)
        
        # 训练参数
        self.clip_ratio = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
        logger.info("自定义PPO智能体初始化完成")
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """获取动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_mean, action_std = self.policy_network(state_tensor)
            
            # 采样动作
            action_dist = torch.distributions.Normal(action_mean, action_std)
            action = action_dist.sample()
            
            return action.squeeze(0).numpy()
    
    def train(self, env, total_timesteps: int, **kwargs):
        """训练智能体"""
        logger.info(f"开始训练自定义PPO智能体，总步数: {total_timesteps}")
        # 这里实现训练逻辑
        # 为了简化，这里只是占位
        pass
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])


class CustomTD3Agent:
    """自定义TD3智能体实现 (备选方案)"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 创建网络
        self.actor = ActorNetwork(input_dim, output_dim, hidden_dim)
        self.critic1 = CriticNetwork(input_dim, output_dim, hidden_dim)
        self.critic2 = CriticNetwork(input_dim, output_dim, hidden_dim)
        
        # 目标网络
        self.target_actor = ActorNetwork(input_dim, output_dim, hidden_dim)
        self.target_critic1 = CriticNetwork(input_dim, output_dim, hidden_dim)
        self.target_critic2 = CriticNetwork(input_dim, output_dim, hidden_dim)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)
        
        logger.info("自定义TD3智能体初始化完成")
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """获取动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.actor(state_tensor)
            return action.squeeze(0).numpy()
    
    def train(self, env, total_timesteps: int, **kwargs):
        """训练智能体"""
        logger.info(f"开始训练自定义TD3智能体，总步数: {total_timesteps}")
        # 这里实现训练逻辑
        # 为了简化，这里只是占位
        pass
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])


# ==================== 神经网络定义 ====================

class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, output_dim)
        self.fc_std = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc_mean(x))
        std = F.softplus(self.fc_std(x)) + 1e-6
        return mean, std


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorNetwork(nn.Module):
    """Actor网络 (TD3)"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic网络 (TD3)"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==================== 便捷函数 ====================

def create_agent(algorithm: str = "PPO", 
                input_dim: int = None, 
                output_dim: int = None,
                use_sb3: bool = True,
                **kwargs) -> RLAgent:
    """
    创建强化学习智能体
    
    Args:
        algorithm: 算法类型 ("PPO", "TD3", "SAC")
        input_dim: 状态空间维度
        output_dim: 动作空间维度
        use_sb3: 是否使用stable-baselines3
        **kwargs: 其他参数
        
    Returns:
        RLAgent: 智能体实例
    """
    if input_dim is None or output_dim is None:
        raise ValueError("必须指定input_dim和output_dim")
    
    return RLAgent(
        input_dim=input_dim,
        output_dim=output_dim,
        algorithm=algorithm,
        use_sb3=use_sb3,
        **kwargs
    )


def get_available_algorithms() -> list:
    """获取可用的算法列表"""
    algorithms = ["PPO", "TD3", "SAC"]
    if not SB3_AVAILABLE:
        logger.warning("Stable-Baselines3不可用，只能使用自定义实现")
    return algorithms
