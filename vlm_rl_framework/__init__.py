"""
基于强化学习与视觉语言模型的机械臂智能控制系统
VLM + RL Framework for Robotic Arm Control

作者: lys
版本: v1.0.0
日期: 2025-07-20
"""

__version__ = "1.0.0"
__author__ = "lys"

# 导入主要模块
from .vlm_module import VLMProcessor
from .rl_module import RLAgent, PandaEnvironment
from .system_integration import VLMRLSystem

__all__ = [
    "VLMProcessor",
    "RLAgent", 
    "PandaEnvironment",
    "VLMRLSystem"
]
