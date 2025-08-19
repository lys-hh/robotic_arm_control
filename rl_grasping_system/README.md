# 强化学习抓取系统

这是一个独立的、基于强化学习的机械臂抓取系统，专门用于训练Panda机械臂执行"到达并抓取"任务。

## 系统特点

### 🎯 核心功能
- **独立模块**: 完全独立的抓取系统，不影响现有VLM项目
- **PPO算法**: 基于Stable-Baselines3的PPO实现
- **归一化技术**: 观察归一化、奖励归一化、优势函数归一化
- **云端支持**: 无头渲染，适用于云端训练
- **完整监控**: 训练进度、成功率、奇异点检测等

### 🔧 技术架构
- **环境**: MuJoCo + Panda机械臂
- **任务**: 到达并抓取固定位置物体
- **算法**: PPO with 完整归一化
- **监控**: 实时训练监控和可视化
- **部署**: 云端无头训练支持

## 快速开始

### 1. 安装依赖
```bash
cd rl_grasping_system
pip install -r requirements.txt
```

### 2. 云端训练
```bash
python train_cloud.py
```

### 3. 本地训练（带监控）
```bash
python train_with_monitor.py
```

### 4. 评估模型
```bash
python evaluate.py --model_path models/final_model.zip
```

## 系统架构

### 核心模块

#### 1. 环境模块 (`environment.py`)
- **PandaGraspingEnv**: 基于MuJoCo的抓取环境
- **状态空间**: 关节位置/速度/力矩、末端执行器位置/方向/速度、夹爪状态、目标信息
- **动作空间**: 关节命令 + 夹爪控制
- **奖励函数**: 距离奖励、抓取奖励、完成奖励、惩罚机制

#### 2. 智能体模块 (`agent.py`)
- **GraspingAgent**: PPO智能体封装
- **GraspingCallback**: 抓取任务专用回调函数
- **早停机制**: 基于成功率的智能早停
- **模型管理**: 保存/加载/评估功能

#### 3. 配置管理 (`config.py`)
- **GraspingConfig**: 抓取任务配置
- **NetworkConfig**: 网络架构配置
- **TrainingConfig**: 训练参数配置
- **RewardConfig**: 奖励函数配置
- **SystemConfig**: 系统配置

#### 4. 归一化系统 (`vec_normalize_wrapper.py`)
- **VecNormalizeWrapper**: 观察和奖励归一化
- **RunningMeanStd**: 在线统计计算
- **裁剪机制**: 防止归一化值过大

#### 5. 训练监控 (`training_monitor.py`)
- **TrainingMonitor**: 完整的训练监控系统
- **实时图表**: 奖励、成功率、奇异点统计
- **日志记录**: 详细的训练日志
- **早停支持**: 基于性能的智能早停

#### 6. 安全机制
- **SingularityHandler**: 奇异点检测和处理
- **SafeActionWrapper**: 动作安全约束
- **碰撞检测**: 防止机械臂碰撞

## 配置说明

### 训练配置
```python
# 基本参数
total_timesteps: int = 2000000  # 总训练步数
learning_rate: float = 3e-4     # 学习率
batch_size: int = 1024          # 批次大小
n_steps: int = 1024             # 每轮步数

# 归一化参数
normalize_observations: bool = True  # 观察归一化
normalize_rewards: bool = True       # 奖励归一化
norm_obs_clip: float = 10.0         # 观察裁剪
norm_reward_clip: float = 10.0      # 奖励裁剪
```

### 奖励配置
```python
# 距离奖励
distance_reward_scale: float = 0.1  # 距离奖励系数
distance_threshold: float = 0.2     # 距离阈值

# 抓取奖励
grasp_reward: float = 500.0         # 抓取成功奖励
grasp_distance_threshold: float = 0.1  # 抓取距离阈值

# 完成奖励
completion_reward: float = 1000.0   # 任务完成奖励
```

## 归一化技术

### 1. 观察归一化
- **目的**: 稳定训练，避免不同特征尺度差异
- **实现**: 在线计算均值和方差
- **裁剪**: 防止归一化值过大

### 2. 奖励归一化
- **目的**: 稳定梯度，避免奖励爆炸
- **实现**: 在线计算奖励统计量
- **裁剪**: 限制奖励范围

### 3. 优势函数归一化
- **目的**: 提高策略更新稳定性
- **实现**: PPO算法内部自动处理
- **方法**: GAE (Generalized Advantage Estimation)

## 训练监控

### 实时指标
- **Episode奖励**: 每个episode的总奖励
- **成功率**: 抓取成功的episode比例
- **平均步数**: 完成任务的步数统计
- **奇异点检测**: 机械臂奇异点统计

### 可视化图表
- **训练曲线**: 奖励、成功率、步数趋势
- **分布图**: 奖励分布、步数分布
- **实时更新**: 每2000个episodes自动保存

## 云端部署

### 环境设置
```bash
# 设置无头渲染
export MUJOCO_GL=egl
export DISPLAY=:0

# 运行云端训练
python train_cloud.py
```

### 特点
- **无头渲染**: 完全避免图形界面依赖
- **CPU优化**: 强制使用CPU，避免GPU警告
- **错误处理**: 完善的异常处理和日志记录
- **自动保存**: 定期保存模型和图表

## 性能优化

### 网络架构
```python
# 策略网络
policy_hidden_sizes: [512, 512, 256]

# 价值网络
value_hidden_sizes: [512, 512, 256]
```

### 训练参数
- **学习率**: 3e-4 (适中)
- **熵系数**: 0.2 (促进探索)
- **GAE lambda**: 0.95 (优势估计)
- **裁剪范围**: 0.2 (PPO裁剪)

## 故障排除

### 常见问题

#### 1. 观察类型错误
```
Exception: Unrecognized type of observation <class 'tuple'>
```
**解决方案**: 确保VecNormalizeWrapper返回正确的numpy数组格式

#### 2. 图形渲染失败
```
Cannot initialize a EGL device display
```
**解决方案**: 使用云端训练脚本，设置无头渲染

#### 3. 训练不收敛
**解决方案**: 
- 检查奖励函数设计
- 调整网络架构
- 增加训练步数
- 启用归一化

## 项目状态

### ✅ 已完成
- [x] 独立RL抓取系统架构
- [x] PPO智能体实现
- [x] 完整归一化技术
- [x] 云端训练支持
- [x] 训练监控系统
- [x] 安全机制实现

### 🔧 技术改进
- **归一化**: 观察/奖励归一化 + PPO内置优势函数归一化
- **云端优化**: 无头渲染、CPU训练、错误处理
- **监控增强**: 实时图表、早停机制、详细日志

## 文件结构

```
rl_grasping_system/
├── environment.py              # 抓取环境
├── agent.py                   # PPO智能体
├── config.py                  # 配置管理
├── training_monitor.py        # 训练监控
├── vec_normalize_wrapper.py   # 归一化包装器
├── train_cloud.py            # 云端训练脚本
├── train_with_monitor.py     # 本地训练脚本
├── evaluate.py               # 模型评估
├── singularity_handler.py    # 奇异点处理
├── action_wrapper.py         # 动作安全包装
├── state.py                  # 状态处理
├── reward.py                 # 奖励函数
├── requirements.txt          # 依赖列表
└── README.md                # 本文件
```

## 许可证

MIT License
