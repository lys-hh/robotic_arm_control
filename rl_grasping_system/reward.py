"""
奖励函数实现
包含肌腱控制和接触奖励
"""

import numpy as np
from typing import Dict, Any

def calculate_reward(state: Dict[str, np.ndarray], prev_state: Dict[str, np.ndarray], action: Dict[str, np.ndarray]) -> float:
    """
    计算奖励函数
    
    Args:
        state: 当前状态
        prev_state: 前一步状态
        action: 当前动作
        
    Returns:
        reward: 总奖励
    """
    reward = 0.0
    
    # 1. 位置奖励
    pos_error = np.linalg.norm(state['ee_position'] - state['target_position'])
    r_position = -5.0 * pos_error
    reward += r_position
    
    # 2. 方向奖励
    quat_diff = 1 - abs(np.dot(state['ee_orientation'], state['target_orientation']))
    r_orientation = -2.0 * quat_diff
    reward += r_orientation
    
    # 3. 接触奖励 (基于肌腱张力)
    r_contact = 0
    if state['tendon_tension'] > 5.0:  # 接触阈值
        r_contact = 2.0 * min(1.0, state['tendon_tension']/50.0)
    reward += r_contact
    
    # 4. 抓取成功奖励
    gripper_closed = state['gripper_state'] < 0.05
    in_target_zone = pos_error < 0.05
    r_grasp = 10.0 if (gripper_closed and in_target_zone) else 0
    reward += r_grasp
    
    # 5. 奇异点规避奖励
    r_manipulability = 1.5 * state['manipulability']
    reward += r_manipulability
    
    # 6. 动作平滑惩罚
    if prev_state is not None:
        action_diff = np.linalg.norm(action['joint_commands'] - prev_state['joint_velocities'])
        r_smooth = -0.1 * action_diff
        reward += r_smooth
    
    # 7. 肌腱控制奖励
    r_tendon = calculate_tendon_reward(state, action)
    reward += r_tendon
    
    # 8. 接触力奖励
    r_force = calculate_contact_force_reward(state)
    reward += r_force
    
    return reward

def calculate_tendon_reward(state: Dict[str, np.ndarray], action: Dict[str, np.ndarray]) -> float:
    """
    计算肌腱控制奖励
    
    Args:
        state: 当前状态
        action: 当前动作
        
    Returns:
        reward: 肌腱奖励
    """
    reward = 0.0
    
    # 肌腱张力奖励
    tendon_tension = state['tendon_tension']
    target_tension = 20.0  # 目标张力
    
    # 张力接近目标时给予奖励
    tension_error = abs(tendon_tension - target_tension)
    if tension_error < 5.0:
        reward += 1.0 * (1.0 - tension_error / 5.0)
    
    # 肌腱位置奖励
    tendon_pos = state['tendon_position']
    target_pos = 0.02  # 目标位置
    
    # 位置接近目标时给予奖励
    pos_error = abs(tendon_pos - target_pos)
    if pos_error < 0.01:
        reward += 0.5 * (1.0 - pos_error / 0.01)
    
    # 肌腱控制平滑性奖励
    tendon_cmd = action['tendon_command']
    if 0.3 <= tendon_cmd <= 0.7:  # 适中的控制命令
        reward += 0.2
    
    return reward

def calculate_contact_force_reward(state: Dict[str, np.ndarray]) -> float:
    """
    计算接触力奖励
    
    Args:
        state: 当前状态
        
    Returns:
        reward: 接触力奖励
    """
    reward = 0.0
    
    # 接触力奖励
    contact_force = state['contact_force']
    left_force = np.linalg.norm(state['left_finger_force'])
    right_force = np.linalg.norm(state['right_finger_force'])
    
    # 适中的接触力给予奖励
    if 1.0 <= contact_force <= 10.0:
        reward += 1.0 * min(1.0, contact_force / 10.0)
    
    # 左右手指力平衡奖励
    force_balance = 1.0 - abs(left_force - right_force) / (left_force + right_force + 1e-6)
    if force_balance > 0.8:
        reward += 0.5 * force_balance
    
    return reward

def calculate_grasp_quality_reward(state: Dict[str, np.ndarray]) -> float:
    """
    计算抓取质量奖励
    
    Args:
        state: 当前状态
        
    Returns:
        reward: 抓取质量奖励
    """
    reward = 0.0
    
    # 基于肌腱张力的抓取质量
    tendon_tension = state['tendon_tension']
    gripper_state = state['gripper_state']
    
    # 理想的抓取状态：适中的张力和闭合的夹爪
    if 10.0 <= tendon_tension <= 30.0 and gripper_state < 0.02:
        reward += 5.0
    
    # 过度抓取惩罚
    if tendon_tension > 50.0:
        reward -= 2.0
    
    # 抓取不足惩罚
    if tendon_tension < 5.0 and gripper_state < 0.02:
        reward -= 1.0
    
    return reward
