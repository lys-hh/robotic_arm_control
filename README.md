# 机器人强化学习项目

这是一个综合性的机器人控制系统项目，包含传统的PID轨迹跟踪控制和新兴的强化学习抓取系统。

## 项目结构

```
robotic_arm_control/
├── panda/                      # Panda机械臂PID控制系统
│   ├── pid_panda_simple.py     # 简化版PID控制系统
│   ├── pid_panda_mujoco_motor.py # 双闭环PID控制系统
│   ├── pid_panda_optimized.py  # 优化版PID控制系统
│   ├── pid_control_comparison.py # PID控制对比分析
│   └── PID_调参指南.md         # PID参数调优指南
├── ur5e_control/               # UR5e机械臂控制系统
│   ├── pid_trajectory_tracking.py # PID轨迹跟踪
│   ├── ee_trajectory_dual_pid.py # 末端执行器双PID控制
│   └── 机械臂初始与目标位置设置.md # 位置设置说明
├── vlm_rl_framework/           # VLM诊断和测试框架
│   ├── vlm_module/            # VLM处理模块
│   ├── qwen_vl_test.py        # Qwen-VL-Chat测试脚本
│   └── check_model.py         # 模型配置检查脚本
├── rl_grasping_system/        # 独立的强化学习抓取系统
│   ├── environment.py         # Panda机械臂抓取环境
│   ├── agent.py              # PPO智能体实现
│   ├── config.py             # 配置管理
│   ├── training_monitor.py   # 训练监控系统
│   ├── vec_normalize_wrapper.py # 归一化包装器
│   ├── train_cloud.py        # 云端训练脚本
│   └── README.md             # 抓取系统详细文档
├── models/                    # MuJoCo模型文件
│   ├── franka_emika_panda/   # Panda机械臂模型
│   └── universal_robots_ur5e/ # UR5e机械臂模型
└── README.md                 # 本文件
```

## 主要功能

### 1. 传统PID控制系统

#### Panda机械臂PID控制
- **简化版PID控制** (`pid_panda_simple.py`): 基于位置控制的PID轨迹跟踪
- **双闭环PID控制** (`pid_panda_mujoco_motor.py`): 基于MuJoCo虚拟电机的双闭环控制
- **优化版PID控制** (`pid_panda_optimized.py`): 优化的PID参数和控制策略
- **控制对比分析** (`pid_control_comparison.py`): 不同PID控制方法的性能对比
- **参数调优指南** (`PID_调参指南.md`): 详细的PID参数调优方法

#### UR5e机械臂PID控制
- **PID轨迹跟踪** (`pid_trajectory_tracking.py`): 基本的PID轨迹跟踪控制
- **末端执行器双PID控制** (`ee_trajectory_dual_pid.py`): 末端执行器的双PID控制策略

### 2. VLM诊断系统
- **Qwen-VL-Chat模型诊断**: 完整的视觉语言模型加载和推理测试
- **问题定位**: 逐步诊断VLM加载、图像处理、推理过程中的问题
- **模型验证**: 检查模型配置和视觉支持能力

### 3. 强化学习抓取系统
- **独立模块**: 完全独立的抓取系统，不影响现有PID控制系统
- **PPO算法**: 基于Stable-Baselines3的PPO实现
- **归一化技术**: 观察归一化、奖励归一化、优势函数归一化
- **云端支持**: 无头渲染，适用于云端训练
- **完整监控**: 训练进度、成功率、奇异点检测等

## 技术特点

### 传统PID控制
- **高精度轨迹跟踪**: 基于PID算法的精确轨迹控制
- **多种控制策略**: 简化版、双闭环、优化版等多种实现
- **参数调优**: 详细的调参指南和性能对比
- **实时监控**: 完整的控制数据记录和分析

### VLM系统
- 支持Qwen-VL-Chat模型
- 完整的图像预处理流程
- 详细的错误诊断和日志

### RL抓取系统
- **环境**: MuJoCo + Panda机械臂
- **算法**: PPO with 归一化
- **任务**: 到达并抓取固定位置物体
- **监控**: 实时训练监控和可视化
- **部署**: 云端无头训练支持

## 快速开始

### 传统PID控制
```bash
# Panda机械臂PID控制
cd panda
python pid_panda_simple.py        # 简化版
python pid_panda_mujoco_motor.py  # 双闭环版
python pid_panda_optimized.py     # 优化版

# UR5e机械臂PID控制
cd ../ur5e_control
python pid_trajectory_tracking.py
python ee_trajectory_dual_pid.py
```

### VLM诊断
```bash
cd vlm_rl_framework
python qwen_vl_test.py
```

### RL抓取训练
```bash
cd rl_grasping_system
python train_cloud.py
```

## 项目状态

### ✅ 已完成
- [x] Panda机械臂PID控制系统（多种实现）
- [x] UR5e机械臂PID控制系统
- [x] VLM诊断系统完整实现
- [x] 独立RL抓取系统架构
- [x] PPO智能体实现
- [x] 归一化技术集成
- [x] 云端训练支持
- [x] 完整监控系统

### 🔧 技术改进
- **PID控制**: 多种控制策略，详细参数调优
- **归一化**: 观察/奖励归一化 + PPO内置优势函数归一化
- **云端优化**: 无头渲染、CPU训练、错误处理
- **监控增强**: 实时图表、早停机制、详细日志

## 文件说明

### 传统PID控制
- `panda/`: Panda机械臂PID控制系统
- `ur5e_control/`: UR5e机械臂PID控制系统
- 详细文档请参考各目录下的README和指南文件

### VLM相关
- `vlm_rl_framework/`: VLM诊断和测试框架
- `qwen_vl_test.py`: Qwen-VL-Chat专用测试
- `check_model.py`: 模型配置检查

### RL抓取系统
- `rl_grasping_system/`: 完整的强化学习抓取系统
- 详细文档请参考 `rl_grasping_system/README.md`

## 依赖要求

### 传统PID控制
- mujoco
- numpy
- matplotlib
- scipy

### VLM系统
- transformers
- torch
- PIL
- numpy

### RL抓取系统
- stable-baselines3
- mujoco
- gymnasium
- matplotlib
- numpy

## 技术栈总结

### 传统控制技术
- **PID算法**: 经典的比例-积分-微分控制
- **轨迹规划**: 直线、圆弧、样条等多种轨迹
- **实时控制**: 高频率的实时控制循环
- **性能分析**: 控制精度、响应速度、稳定性分析

### 现代AI技术
- **强化学习**: PPO算法，Stable-Baselines3框架
- **物理仿真**: MuJoCo引擎，Panda机械臂
- **归一化**: 观察、奖励、优势函数归一化
- **监控**: 实时训练监控，可视化图表

### 部署技术
- **云端部署**: 无头渲染，CPU优化
- **错误处理**: 异常处理，日志记录
- **配置管理**: 模块化配置，灵活参数
- **文档系统**: 详细说明，使用指南

## 项目价值

### 1. 技术广度
- **传统控制**: PID算法、轨迹规划、实时控制
- **现代AI**: 强化学习、视觉语言模型、归一化技术
- **工程实践**: 系统集成、云端部署、监控运维

### 2. 实用价值
- **工业应用**: 传统PID控制适用于实际机器人
- **研究价值**: RL系统为机器人学习提供新思路
- **教育意义**: 完整的控制系统学习平台

### 3. 工程价值
- **系统设计**: 模块化、可扩展的架构
- **技术融合**: 传统控制与现代AI的结合
- **部署能力**: 从本地到云端的完整部署方案

## 贡献

本项目展示了从传统PID控制到现代强化学习的完整技术栈，包括：
1. 传统控制算法的实现和优化
2. 现代AI技术的应用和集成
3. 完整的工程实践和部署方案
4. 详细的技术文档和使用指南

## 许可证

MIT License 