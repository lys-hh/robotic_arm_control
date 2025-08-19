"""
安全动作包装器
提供渐进式奇异点处理和动作安全约束，包含肌腱控制
"""

import numpy as np
import logging
from typing import Dict, Any

class SafeActionWrapper:
    """安全动作包装器"""
    
    def __init__(self, env):
        self.env = env
        self.logger = logging.getLogger(__name__)
        
        # 安全参数
        self.max_joint_velocity = 0.5  # 最大关节速度 (rad/s)
        self.singularity_threshold = 0.1  # 提高可操作度阈值，减少误报
        self.slowdown_factor = 0.3  # 奇异点附近降速因子
        self.max_tension = 100.0  # 最大允许张力(N)
        
        # 关节限制
        self.joint_limits = {
            'joint1': (-2.8973, 2.8973),
            'joint2': (-1.7628, 1.7628),
            'joint3': (-2.8973, 2.8973),
            'joint4': (-3.0718, -0.0698),
            'joint5': (-2.8973, 2.8973),
            'joint6': (-0.0175, 3.7525),
            'joint7': (-2.8973, 2.8973)
        }
        
        # 奇异点处理参数
        self.singularity_slowdown_factors = {
            'elbow': 0.2,      # 肘部奇异点：更慢
            'wrist': 0.3,      # 腕部奇异点：中等
            'shoulder': 0.4,   # 肩部奇异点：较快
            'jacobian': 0.1,   # 雅可比奇异：最慢
            'default': 0.3     # 默认降速
        }
    
    def apply(self, raw_action: np.ndarray, current_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        应用安全约束到原始动作
        
        Args:
            raw_action: 原始动作 [8,] 或字典格式
            current_state: 当前状态信息
            
        Returns:
            safe_action: 安全动作字典
        """
        # 处理输入格式
        if isinstance(raw_action, np.ndarray):
            # 转换为字典格式
            action_dict = {
                'joint_commands': raw_action[:7],
                'tendon_command': raw_action[7] if len(raw_action) > 7 else 0.0,
                'gripper_command': raw_action[7] if len(raw_action) > 7 else 0.0
            }
        else:
            action_dict = raw_action.copy()
        
        # 1. 奇异点检测和处理（与SingularityHandler协调）
        current_joint_pos = current_state.get('joint_positions', np.zeros(7))
        if hasattr(self.env, 'singularity_handler'):
            # 使用奇异点处理器的检测结果
            is_singular, singularity_type, singularity_score = self.env.singularity_handler.detect_singularity(current_joint_pos)
            
            if is_singular:
                # 根据奇异点类型选择降速因子
                slowdown_factor = self._get_singularity_slowdown_factor(singularity_type, singularity_score)
                action_dict['joint_commands'] *= slowdown_factor
                
                # 记录奇异点处理（仅在调试模式下）
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"奇异点处理: 类型={singularity_type}, 程度={singularity_score:.3f}, 降速因子={slowdown_factor:.2f}")
        else:
            # 备用方案：使用可操作度
            manipulability = current_state.get('manipulability', 1.0)
            if manipulability < self.singularity_threshold:
                action_dict['joint_commands'] *= self.slowdown_factor
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"可操作度降速: 可操作度={manipulability:.3f}")
        
        # 2. 肌腱张力保护
        tendon_tension = current_state.get('tendon_tension', 0.0)
        if tendon_tension > self.max_tension * 0.8:
            # 张力过高时只允许放松
            action_dict['tendon_command'] = min(action_dict['tendon_command'], 0)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"肌腱张力保护: 张力={tendon_tension:.2f}N")
        
        # 3. 关节速度限制
        current_joint_vel = current_state.get('joint_velocities', np.zeros(7))
        joint_commands = action_dict['joint_commands']
        
        # 限制关节速度变化
        max_velocity_change = self.max_joint_velocity * 0.01  # 每步最大速度变化
        velocity_change = joint_commands - current_joint_vel
        velocity_change = np.clip(velocity_change, -max_velocity_change, max_velocity_change)
        action_dict['joint_commands'] = current_joint_vel + velocity_change
        
        # 4. 关节位置限制
        current_joint_pos = current_state.get('joint_positions', np.zeros(7))
        for i, (joint_name, (low, high)) in enumerate(self.joint_limits.items()):
            # 确保动作不会导致超出关节限制
            predicted_pos = current_joint_pos[i] + action_dict['joint_commands'][i] * 0.01
            if predicted_pos < low or predicted_pos > high:
                # 调整动作以避免超出限制
                max_action = (high - current_joint_pos[i]) / 0.01
                min_action = (low - current_joint_pos[i]) / 0.01
                action_dict['joint_commands'][i] = np.clip(action_dict['joint_commands'][i], min_action, max_action)
                
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"关节限制保护: {joint_name}, 预测位置={predicted_pos:.3f}, 限制=[{low:.3f}, {high:.3f}]")
        
        # 5. 夹爪控制限制
        action_dict['gripper_command'] = np.clip(action_dict['gripper_command'], 0.0, 0.04)
        
        # 6. 肌腱控制限制
        action_dict['tendon_command'] = np.clip(action_dict['tendon_command'], 0.0, 1.0)
        
        return action_dict
    
    def _get_singularity_slowdown_factor(self, singularity_type: str, singularity_score: float) -> float:
        """
        根据奇异点类型和程度获取降速因子
        
        Args:
            singularity_type: 奇异点类型
            singularity_score: 奇异程度 (0-1)
            
        Returns:
            slowdown_factor: 降速因子
        """
        # 基础降速因子
        base_factor = self.singularity_slowdown_factors.get(singularity_type, self.singularity_slowdown_factors['default'])
        
        # 根据奇异程度调整
        if singularity_score > 0.9:  # 严重奇异
            return base_factor * 0.5  # 进一步降速
        elif singularity_score > 0.7:  # 中等奇异
            return base_factor
        else:  # 轻微奇异
            return base_factor * 1.5  # 稍微放宽
        
        return base_factor
    
    def get_safe_initial_action(self) -> Dict[str, np.ndarray]:
        """获取安全的初始动作"""
        return {
            'joint_commands': np.zeros(7, dtype=np.float32),
            'tendon_command': 0.0,
            'gripper_command': 0.02  # 半开状态
        }
    
    def tendon_dynamics(self, cmd: float, current_pos: float, tension: float, dt: float = 0.01) -> tuple:
        """
        肌腱动力学模型
        
        Args:
            cmd: 控制命令 (0-1)
            current_pos: 当前肌腱位置
            tension: 当前张力
            dt: 时间步长
            
        Returns:
            new_pos: 新的肌腱位置
            new_tension: 新的张力
        """
        # 肌腱动力学参数
        max_speed = 0.5  # m/s
        spring_constant = 500.0  # N/m
        rest_length = 0.02  # m
        
        # 限制速度
        speed = np.clip(cmd, -max_speed, max_speed)
        
        # 更新位置
        new_pos = current_pos + speed * dt
        
        # 计算张力 (F = k * Δx)
        new_tension = spring_constant * (new_pos - rest_length)
        
        return new_pos, new_tension
