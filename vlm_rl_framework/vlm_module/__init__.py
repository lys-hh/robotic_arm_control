"""
视觉语言模型模块
Vision Language Model Module

功能:
- 图像理解和目标检测
- 自然语言指令解析
- 坐标转换和映射
"""

from .vlm_processor import VLMProcessor
from .coordinate_mapper import CoordinateMapper
from .instruction_parser import InstructionParser

__all__ = [
    "VLMProcessor",
    "CoordinateMapper", 
    "InstructionParser"
]
