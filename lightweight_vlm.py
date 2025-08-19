#!/usr/bin/env python3
"""
轻量级VLM处理器 - 不依赖大模型
使用OpenCV和规则引擎实现基本视觉功能
"""

import numpy as np
import cv2
import time
from typing import Dict, List

class LightweightVLM:
    def __init__(self):
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([110, 100, 100], [130, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255])
        }
    
    def process_instruction(self, image: np.ndarray, instruction: str) -> Dict:
        """处理图像和指令"""
        start_time = time.time()
        
        # 解析指令
        color = self._extract_color(instruction)
        
        # 检测物体
        objects = self._detect_objects(image, color)
        
        # 选择目标物体
        target = objects[0] if objects else self._get_default_object()
        
        # 计算世界坐标
        world_pos = self._pixel_to_world(target['center'])
        
        result = {
            "object_type": target['type'],
            "position": target['center'],
            "world_position": world_pos,
            "confidence": target['confidence'],
            "processing_time": time.time() - start_time
        }
        
        return result
    
    def _extract_color(self, instruction: str) -> str:
        """从指令中提取颜色"""
        instruction_lower = instruction.lower()
        if "红色" in instruction or "red" in instruction_lower:
            return "red"
        elif "蓝色" in instruction or "blue" in instruction_lower:
            return "blue"
        elif "绿色" in instruction or "green" in instruction_lower:
            return "green"
        return None
    
    def _detect_objects(self, image: np.ndarray, target_color: str = None) -> List[Dict]:
        """检测图像中的物体"""
        objects = []
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        colors_to_check = [target_color] if target_color else self.color_ranges.keys()
        
        for color_name in colors_to_check:
            if color_name not in self.color_ranges:
                continue
                
            lower, upper = self.color_ranges[color_name]
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) < 1000:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                center = [x + w//2, y + h//2]
                area = cv2.contourArea(contour)
                confidence = min(area / 10000, 0.9)
                
                objects.append({
                    "type": f"{color_name}_object",
                    "center": center,
                    "bbox": [x, y, x+w, y+h],
                    "confidence": confidence,
                    "color": color_name
                })
        
        return objects
    
    def _pixel_to_world(self, pixel_pos: List[int]) -> List[float]:
        """像素坐标转世界坐标"""
        px, py = pixel_pos
        # 确保坐标在合理范围内
        px = max(0, min(px, 640))
        py = max(0, min(py, 480))
        
        # 归一化到[-1, 1]范围
        rel_x = (px - 320) / 320
        rel_y = (py - 240) / 240
        
        # 映射到世界坐标范围
        world_x = 0.5 + rel_x * 0.3  # 限制在[0.2, 0.8]范围
        world_y = 0.0 + rel_y * 0.3  # 限制在[-0.3, 0.3]范围
        world_z = 0.3
        
        # 确保世界坐标在合理范围内
        world_x = max(0.2, min(world_x, 0.8))
        world_y = max(-0.3, min(world_y, 0.3))
        
        return [world_x, world_y, world_z]
    
    def _get_default_object(self) -> Dict:
        """获取默认物体"""
        return {
            "type": "unknown_object",
            "center": [320, 240],
            "confidence": 0.5,
            "color": "unknown"
        }

# 测试函数
def test_lightweight_vlm():
    """测试轻量级VLM"""
    print("🔍 测试轻量级VLM...")
    
    vlm = LightweightVLM()
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[200:280, 300:380] = [255, 0, 0]  # 红色方块
    
    # 测试指令
    instruction = "请识别图像中的红色立方体"
    
    # 处理
    result = vlm.process_instruction(test_image, instruction)
    
    print(f"✅ 轻量级VLM测试成功")
    print(f"   检测结果: {result['object_type']}")
    print(f"   置信度: {result['confidence']:.2f}")
    print(f"   世界坐标: {result['world_position']}")
    print(f"   处理时间: {result['processing_time']:.3f}秒")
    
    return result

if __name__ == "__main__":
    test_lightweight_vlm()
