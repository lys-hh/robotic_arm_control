"""
指令解析器 - 解析自然语言指令
"""

import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class InstructionParser:
    """
    自然语言指令解析器
    
    功能:
    - 解析自然语言指令
    - 提取动作、目标物体、位置等信息
    - 转换为机器人可执行的命令
    """
    
    def __init__(self):
        """初始化指令解析器"""
        # 定义动作关键词
        self.actions = {
            "抓取": ["抓取", "拿起", "握住", "抓", "拿"],
            "放置": ["放置", "放下", "放到", "放"],
            "移动": ["移动", "移到", "移动到", "移"],
            "识别": ["识别", "找到", "检测", "发现"],
            "观察": ["观察", "看", "查看", "观察"]
        }
        
        # 定义物体关键词
        self.objects = {
            "红色": ["红色", "红", "red"],
            "蓝色": ["蓝色", "蓝", "blue"], 
            "绿色": ["绿色", "绿", "green"],
            "立方体": ["立方体", "方块", "cube", "块"],
            "圆形": ["圆形", "圆", "circle", "球"],
            "长方体": ["长方体", "矩形", "rectangle", "块"]
        }
        
        # 定义位置关键词
        self.positions = {
            "中心": ["中心", "中间", "center", "中央"],
            "左边": ["左边", "左侧", "left"],
            "右边": ["右边", "右侧", "right"],
            "前面": ["前面", "前方", "front"],
            "后面": ["后面", "后方", "back"],
            "上面": ["上面", "上方", "top"],
            "下面": ["下面", "下方", "bottom"]
        }
        
        logger.info("指令解析器初始化完成")
    
    def parse_instruction(self, instruction: str) -> Dict:
        """
        解析自然语言指令
        
        Args:
            instruction: 自然语言指令
            
        Returns:
            Dict: 解析结果，包含动作、目标、位置等信息
        """
        try:
            result = {
                "action": self._extract_action(instruction),
                "target_object": self._extract_target_object(instruction),
                "position": self._extract_position(instruction),
                "confidence": self._calculate_confidence(instruction),
                "raw_instruction": instruction
            }
            
            logger.info(f"指令解析结果: {result}")
            return result
            
        except Exception as e:
            logger.error(f"指令解析失败: {e}")
            return self._get_fallback_result(instruction)
    
    def _extract_action(self, instruction: str) -> str:
        """提取动作"""
        instruction_lower = instruction.lower()
        
        for action, keywords in self.actions.items():
            for keyword in keywords:
                if keyword in instruction_lower:
                    return action
        
        return "未知动作"
    
    def _extract_target_object(self, instruction: str) -> Dict:
        """提取目标物体"""
        instruction_lower = instruction.lower()
        target = {"color": "未知", "shape": "未知", "full_name": "未知物体"}
        
        # 提取颜色
        for color, keywords in self.objects.items():
            if color in ["红色", "蓝色", "绿色"]:
                for keyword in keywords:
                    if keyword in instruction_lower:
                        target["color"] = color
                        break
        
        # 提取形状
        for shape, keywords in self.objects.items():
            if shape in ["立方体", "圆形", "长方体"]:
                for keyword in keywords:
                    if keyword in instruction_lower:
                        target["shape"] = shape
                        break
        
        # 组合完整名称
        if target["color"] != "未知" and target["shape"] != "未知":
            target["full_name"] = f"{target['color']}{target['shape']}"
        elif target["color"] != "未知":
            target["full_name"] = f"{target['color']}物体"
        elif target["shape"] != "未知":
            target["full_name"] = f"{target['shape']}"
        
        return target
    
    def _extract_position(self, instruction: str) -> str:
        """提取位置信息"""
        instruction_lower = instruction.lower()
        
        for position, keywords in self.positions.items():
            for keyword in keywords:
                if keyword in instruction_lower:
                    return position
        
        return "任意位置"
    
    def _calculate_confidence(self, instruction: str) -> float:
        """计算解析置信度"""
        confidence = 0.0
        instruction_lower = instruction.lower()
        
        # 检查是否包含动作关键词
        for action, keywords in self.actions.items():
            for keyword in keywords:
                if keyword in instruction_lower:
                    confidence += 0.3
                    break
        
        # 检查是否包含物体关键词
        for obj_type, keywords in self.objects.items():
            for keyword in keywords:
                if keyword in instruction_lower:
                    confidence += 0.3
                    break
        
        # 检查是否包含位置关键词
        for position, keywords in self.positions.items():
            for keyword in keywords:
                if keyword in instruction_lower:
                    confidence += 0.2
                    break
        
        # 检查指令长度
        if len(instruction) > 5:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _get_fallback_result(self, instruction: str) -> Dict:
        """获取备用解析结果"""
        return {
            "action": "识别",
            "target_object": {"color": "未知", "shape": "未知", "full_name": "未知物体"},
            "position": "任意位置",
            "confidence": 0.1,
            "raw_instruction": instruction
        }
    
    def validate_instruction(self, instruction: str) -> bool:
        """验证指令是否有效"""
        if not instruction or len(instruction.strip()) == 0:
            return False
        
        # 检查是否包含基本关键词
        instruction_lower = instruction.lower()
        has_action = any(keyword in instruction_lower 
                        for keywords in self.actions.values() 
                        for keyword in keywords)
        has_object = any(keyword in instruction_lower 
                        for keywords in self.objects.values() 
                        for keyword in keywords)
        
        return has_action or has_object
    
    def get_supported_actions(self) -> List[str]:
        """获取支持的动作列表"""
        return list(self.actions.keys())
    
    def get_supported_objects(self) -> List[str]:
        """获取支持的物体列表"""
        return list(self.objects.keys())
    
    def get_supported_positions(self) -> List[str]:
        """获取支持的位置列表"""
        return list(self.positions.keys())
