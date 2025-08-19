"""
奇异点检测和处理模块
处理Panda机械臂的奇异点问题
"""

import numpy as np
import logging
from typing import Tuple, Optional

class SingularityHandler:
    """奇异点处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Panda机械臂的关节限制
        self.joint_limits = {
            'joint1': (-2.8973, 2.8973),
            'joint2': (-1.7628, 1.7628),
            'joint3': (-2.8973, 2.8973),
            'joint4': (-3.0718, -0.0698),
            'joint5': (-2.8973, 2.8973),
            'joint6': (-0.0175, 3.7525),
            'joint7': (-2.8973, 2.8973)
        }
        
        # 已知的奇异点配置（大幅增加容忍度）
        self.singularity_configs = [
            # 肘部奇异点：joint3接近±π/2
            {'joint3': np.pi/2, 'tolerance': 0.5},   # 增加容忍度到0.5弧度
            {'joint3': -np.pi/2, 'tolerance': 0.5},
            
            # 腕部奇异点：joint5接近±π/2（进一步放宽）
            {'joint5': np.pi/2, 'tolerance': 0.3},   # 降低容忍度，减少误报
            {'joint5': -np.pi/2, 'tolerance': 0.3},
            
            # 肩部奇异点：joint2接近±π/2（进一步放宽）
            {'joint2': np.pi/2, 'tolerance': 0.3},   # 降低容忍度，减少误报
            {'joint2': -np.pi/2, 'tolerance': 0.3},
        ]
        
        # 安全配置（远离奇异点）
        self.safe_config = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.7854])
        
        # 警告控制
        self.warning_cooldown = 0  # 警告冷却时间
        self.last_warning_time = 0  # 上次警告时间
        self.warning_interval = 30.0  # 警告间隔（秒）- 大幅延长
    
    def detect_singularity(self, joint_positions: np.ndarray) -> Tuple[bool, str, float]:
        """
        检测是否处于奇异点
        
        Args:
            joint_positions: 关节位置数组 [7,]
            
        Returns:
            is_singular: 是否处于奇异点
            singularity_type: 奇异点类型
            singularity_score: 奇异点程度 (0-1)
        """
        if len(joint_positions) < 7:
            return False, "invalid", 0.0
        
        # 检查关节限制
        for i, (joint_name, (low, high)) in enumerate(self.joint_limits.items()):
            if joint_positions[i] < low or joint_positions[i] > high:
                return True, f"joint_limit_{joint_name}", 1.0
        
        # 检查已知奇异点配置
        max_score = 0.0
        singularity_type = "none"
        
        for config in self.singularity_configs:
            for joint_name, target_value in config.items():
                if joint_name == 'joint3':
                    joint_idx = 2
                elif joint_name == 'joint5':
                    joint_idx = 4
                elif joint_name == 'joint2':
                    joint_idx = 1
                else:
                    continue
                
                # 计算与奇异点的距离
                distance = abs(joint_positions[joint_idx] - target_value)
                tolerance = config['tolerance']
                
                # 计算奇异点程度 (0-1)
                if distance < tolerance:
                    score = 1.0 - (distance / tolerance)
                    if score > max_score:
                        max_score = score
                        singularity_type = f"singularity_{joint_name}"
        
        # 检查雅可比矩阵条件数（简化版本）
        jacobian_condition = self._estimate_jacobian_condition(joint_positions)
        # 大幅提高阈值：条件数 > 500 才认为是奇异
        if jacobian_condition > 500:  # 提高阈值从300到500
            score = min(1.0, (jacobian_condition - 500) / 500)  # 重新归一化
            if score > max_score:
                max_score = score
                singularity_type = "jacobian_singularity"
        
        # 大幅提高检测阈值：只有分数 > 0.9 才认为是奇异点
        return max_score > 0.9, singularity_type, max_score
    
    def _estimate_jacobian_condition(self, joint_positions: np.ndarray) -> float:
        """
        估算雅可比矩阵的条件数（改进版本）
        
        Args:
            joint_positions: 关节位置
            
        Returns:
            condition_number: 条件数估计值
        """
        # 改进的雅可比矩阵条件数估计
        # 只检查真正的奇异点附近
        
        # 检查肘部奇异点 (joint3 接近 ±π/2)
        elbow_angle = joint_positions[2]  # joint3
        elbow_singularity = abs(abs(elbow_angle) - np.pi/2)
        
        # 检查腕部奇异点 (joint5 接近 ±π/2)
        wrist_angle = joint_positions[4]  # joint5
        wrist_singularity = abs(abs(wrist_angle) - np.pi/2)
        
        # 检查肩部奇异点 (joint2 接近 ±π/2)
        shoulder_angle = joint_positions[1]  # joint2
        shoulder_singularity = abs(abs(shoulder_angle) - np.pi/2)
        
        # 计算最小奇异距离
        min_singularity_distance = min(elbow_singularity, wrist_singularity, shoulder_singularity)
        
        # 条件数估计：距离奇异点越近，条件数越大
        # 大幅调整敏感度：只有非常接近奇异点才认为是奇异
        if min_singularity_distance < 0.01:  # 非常接近奇异点（从0.02降低到0.01）
            condition_number = 1000.0
        elif min_singularity_distance < 0.1:  # 接近奇异点（从0.15降低到0.1）
            condition_number = 500.0 + (0.1 - min_singularity_distance) * 500 / 0.09
        else:  # 远离奇异点
            condition_number = 50.0
        
        return condition_number
    
    def get_progressive_safe_action(self, current_config: np.ndarray, step_size: float = 0.05) -> np.ndarray:
        """
        生成渐进式安全动作，逐步远离奇异点
        
        Args:
            current_config: 当前关节配置
            step_size: 最大步长
            
        Returns:
            safe_action: 渐进式安全动作
        """
        # 分析当前奇异点类型
        is_singular, singularity_type, score = self.detect_singularity(current_config)
        
        if not is_singular:
            return current_config  # 如果安全，保持当前配置
        
        # 根据奇异点类型生成渐进动作
        safe_action = current_config.copy()
        
        if "elbow" in singularity_type:
            # 肘部奇异：调整joint3，保持其他关节
            elbow_angle = current_config[2]
            if abs(elbow_angle - np.pi/2) < 0.1:
                safe_action[2] += step_size  # 远离π/2
            elif abs(elbow_angle + np.pi/2) < 0.1:
                safe_action[2] -= step_size  # 远离-π/2
                
        elif "wrist" in singularity_type:
            # 腕部奇异：调整joint5，保持其他关节
            wrist_angle = current_config[4]
            if abs(wrist_angle - np.pi/2) < 0.1:
                safe_action[4] += step_size  # 远离π/2
            elif abs(wrist_angle + np.pi/2) < 0.1:
                safe_action[4] -= step_size  # 远离-π/2
                
        elif "shoulder" in singularity_type:
            # 肩部奇异：调整joint2，保持其他关节
            shoulder_angle = current_config[1]
            if abs(shoulder_angle - np.pi/2) < 0.1:
                safe_action[1] += step_size  # 远离π/2
            elif abs(shoulder_angle + np.pi/2) < 0.1:
                safe_action[1] -= step_size  # 远离-π/2
                
        else:
            # 其他奇异点：向安全配置渐进移动
            direction = self.safe_config - current_config
            step = np.clip(direction, -step_size, step_size)
            safe_action = current_config + step
        
        # 确保在关节限制内
        for i, (joint_name, (low, high)) in enumerate(self.joint_limits.items()):
            safe_action[i] = np.clip(safe_action[i], low, high)
        
        return safe_action

    def get_safe_config(self, current_config: np.ndarray) -> np.ndarray:
        """
        获取安全配置（使用渐进式方法）
        
        Args:
            current_config: 当前配置
            
        Returns:
            safe_config: 安全配置
        """
        return self.get_progressive_safe_action(current_config, step_size=0.05)
    
    def generate_safe_initial_config(self) -> np.ndarray:
        """
        生成安全的初始配置
        
        Returns:
            safe_config: 安全的初始配置
        """
        # 在更保守的安全范围内随机生成配置
        safe_ranges = [
            (-0.5, 0.5),     # joint1: 更保守的范围
            (-0.3, 0.3),     # joint2: 避免肩部奇异
            (-0.5, 0.5),     # joint3: 避免肘部奇异
            (-2.0, -1.0),    # joint4: 更保守的范围
            (-0.5, 0.5),     # joint5: 避免腕部奇异
            (1.0, 2.0),      # joint6: 更保守的范围
            (-0.5, 0.5),     # joint7: 更保守的范围
        ]
        
        config = np.array([
            np.random.uniform(low, high) for low, high in safe_ranges
        ])
        
        # 验证配置安全性
        is_singular, _, _ = self.detect_singularity(config)
        if is_singular:
            # 如果仍然奇异，使用预定义安全配置
            config = self.safe_config.copy()
        
        return config
    
    def add_singularity_penalty(self, reward: float, joint_positions: np.ndarray) -> float:
        """
        添加奇异点惩罚
        
        Args:
            reward: 原始奖励
            joint_positions: 关节位置
            
        Returns:
            modified_reward: 修改后的奖励
        """
        is_singular, singularity_type, score = self.detect_singularity(joint_positions)
        
        if is_singular:
            # 大幅减少奇异点惩罚，避免过度干扰学习
            penalty = -1.0 * score  # 最大惩罚-1（原来是-10）
            self.logger.warning(f"检测到奇异点: {singularity_type}, 程度: {score:.3f}, 惩罚: {penalty:.3f}")
            return reward + penalty
        
        return reward
    
    def check_singularity_recovery(self, joint_positions: np.ndarray, 
                                 previous_positions: np.ndarray) -> bool:
        """
        检查是否从奇异点恢复
        
        Args:
            joint_positions: 当前关节位置
            previous_positions: 前一步关节位置
            
        Returns:
            recovered: 是否恢复
        """
        was_singular, _, _ = self.detect_singularity(previous_positions)
        is_singular, _, _ = self.detect_singularity(joint_positions)
        
        return was_singular and not is_singular

    def detect_singularity_with_warning_control(self, joint_positions: np.ndarray, current_time: float = None) -> Tuple[bool, str, float, bool]:
        """
        带警告控制的奇异点检测
        
        Args:
            joint_positions: 关节位置 [7,]
            current_time: 当前时间（用于控制警告频率）
            
        Returns:
            is_singular: 是否处于奇异点
            singularity_type: 奇异点类型
            singularity_score: 奇异程度 (0-1)
            should_warn: 是否应该发出警告
        """
        import time
        
        if current_time is None:
            current_time = time.time()
        
        # 检测奇异点
        is_singular, singularity_type, score = self.detect_singularity(joint_positions)
        
        # 警告控制逻辑
        should_warn = False
        
        if is_singular:
            # 检查是否应该发出警告
            if current_time - self.last_warning_time > self.warning_interval:
                should_warn = True
                self.last_warning_time = current_time
                
                # 根据奇异程度调整警告间隔
                if score > 0.9:  # 严重奇异
                    self.warning_interval = 10.0  # 缩短间隔
                elif score > 0.7:  # 中等奇异
                    self.warning_interval = 30.0  # 标准间隔
                else:  # 轻微奇异
                    self.warning_interval = 60.0  # 大幅延长间隔
        
        return is_singular, singularity_type, score, should_warn
    
    def reset_warning_control(self):
        """重置警告控制"""
        self.last_warning_time = 0
        self.warning_interval = 5.0
