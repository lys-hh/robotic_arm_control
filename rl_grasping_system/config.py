"""
强化学习抓取系统配置文件
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class GraspingConfig:
    """抓取任务配置"""
    # 环境配置
    xml_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/franka_emika_panda/single_cube_scene.xml")
    max_steps: int = 100  # 大幅减少最大步数，避免过长episode
    control_freq: int = 50  # 控制频率 (Hz)
    
    # 抓取配置 - 大幅放宽条件
    grasp_distance_threshold: float = 0.1   # 抓取距离阈值 (大幅放宽)
    grasp_success_threshold: float = 0.08   # 抓取成功阈值 (大幅放宽)
    gripper_closed_threshold: float = 0.05  # 夹爪闭合阈值 (大幅放宽)
    
    # 工作空间配置 - 围绕XML中的物体位置设计
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
        (0.3, 0.7),   # X轴范围 (围绕0.5)
        (-0.2, 0.2),  # Y轴范围 (围绕0.0)
        (0.05, 0.3)   # Z轴范围 (围绕0.1)
    )
    
    # 任务配置
    target_object: str = "target_cube"  # 目标物体名称
    use_fixed_position: bool = True  # 是否使用固定物体位置（便于训练）

@dataclass
class NetworkConfig:
    """网络配置"""
    # 策略网络
    policy_hidden_sizes: List[int] = None
    policy_activation: str = "relu"
    
    # 价值网络
    value_hidden_sizes: List[int] = None
    value_activation: str = "relu"
    
    def __post_init__(self):
        if self.policy_hidden_sizes is None:
            self.policy_hidden_sizes = [512, 512, 256]  # 增加网络深度和宽度
        if self.value_hidden_sizes is None:
            self.value_hidden_sizes = [512, 512, 256]  # 增加网络深度和宽度

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本参数
    total_timesteps: int = 2000000  # 增加总步数，给归一化更多时间
    learning_rate: float = 3e-4     # 适中的学习率
    batch_size: int = 1024          # 与n_steps匹配，避免警告
    n_steps: int = 1024             # 适中的n_steps
    n_epochs: int = 10              # 增加epochs，充分利用数据
    
    # PPO参数 - 增加探索和归一化
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.2           # 大幅增加熵系数，促进探索
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 归一化参数
    normalize_advantage: bool = True  # 优势函数归一化
    normalize_observations: bool = True  # 观察归一化
    normalize_rewards: bool = True  # 奖励归一化
    norm_obs_clip: float = 10.0  # 观察归一化裁剪
    norm_reward_clip: float = 10.0  # 奖励归一化裁剪
    
    # 训练控制
    save_freq: int = 50000          # 减少保存频率
    eval_freq: int = 25000          # 减少评估频率
    log_freq: int = 1000
    
    # 早停条件 - 放宽条件
    success_threshold: float = 0.6   # 降低成功率阈值 (原0.8)
    patience: int = 50              # 增加耐心值 (原30)

@dataclass
class RewardConfig:
    """奖励配置"""
    # 距离奖励 - 大幅减少惩罚，增加正向奖励
    distance_reward_scale: float = 0.1  # 大幅减少距离惩罚
    distance_threshold: float = 0.2     # 大幅增加距离阈值
    
    # 抓取奖励 - 大幅增加正向奖励
    grasp_reward: float = 500.0         # 大幅增加抓取奖励
    grasp_distance_threshold: float = 0.1  # 大幅增加阈值
    
    # 完成奖励 - 大幅增加完成奖励
    completion_reward: float = 1000.0   # 大幅增加完成奖励
    
    # 惩罚 - 大幅减少惩罚强度
    timeout_penalty: float = -0.01      # 大幅减少超时惩罚
    collision_penalty: float = -0.5     # 大幅减少碰撞惩罚
    invalid_action_penalty: float = -0.01  # 大幅减少无效动作惩罚
    
    # 夹爪控制奖励 - 大幅增加控制奖励
    gripper_control_reward: float = 1.0  # 大幅增加夹爪控制奖励

@dataclass
class SystemConfig:
    """系统配置"""
    # 路径配置 - 确保所有路径都在rl_grasping_system目录下
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    models_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    logs_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    results_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # 设备配置
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_envs: int = 1
    
    # 随机种子
    seed: int = 42
    
    # 日志配置
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = "rl-grasping"
    
    def __post_init__(self):
        # 创建必要的目录
        for dir_name in [self.models_dir, self.logs_dir, self.results_dir]:
            os.makedirs(dir_name, exist_ok=True)

@dataclass
class Config:
    """总配置"""
    grasping: GraspingConfig = None
    network: NetworkConfig = None
    training: TrainingConfig = None
    reward: RewardConfig = None
    system: SystemConfig = None
    
    def __post_init__(self):
        if self.grasping is None:
            self.grasping = GraspingConfig()
        if self.network is None:
            self.network = NetworkConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.system is None:
            self.system = SystemConfig()

# 默认配置实例
default_config = Config()

def get_config() -> Config:
    """获取配置"""
    return default_config

def update_config(config: Config, **kwargs):
    """更新配置"""
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            # 尝试更新子配置
            for sub_config_name in ['grasping', 'network', 'training', 'reward', 'system']:
                sub_config = getattr(config, sub_config_name)
                if hasattr(sub_config, key):
                    setattr(sub_config, key, value)
                    break
    return config
