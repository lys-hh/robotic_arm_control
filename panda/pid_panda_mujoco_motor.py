#!/usr/bin/env python3
"""
基于MuJoCo虚拟电机的Panda机械臂PID轨迹跟踪系统

使用MuJoCo内置的actuator系统,支持摩擦力、阻尼等非线性因素调节
控制架构：目标位置 → 任务空间PID → 逆运动学 → 关节PID → MuJoCo虚拟电机

作者:lys
日期:2025
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List
import mujoco
import ikpy.chain
import warnings

# 修复ikpy兼容性问题
if not hasattr(np, 'float'):
    np.float = float

class JointPIDController:
    """关节PID控制器"""
    
    def __init__(self, kp: float = 200.0, ki: float = 0.0, kd: float = 5.0,
                 integral_limit: float = 50.0, output_limit: float = 100.0):
        """
        初始化关节PID控制器
        
        Args:
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            integral_limit: 积分限幅
            output_limit: 输出限幅
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        
        self.reset()
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = 0.0
    
    def compute(self, target: float, current: float, current_velocity: float, dt: float) -> float:
        """
        计算控制输出
        
        Args:
            target: 目标位置
            current: 当前位置
            current_velocity: 当前速度
            dt: 时间步长
            
        Returns:
            control_output: 控制输出（力矩）
        """
        error = target - current
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # 微分项
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        # 总输出
        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        # 更新状态
        self.prev_error = error
        self.prev_time += dt
        
        return output

class TaskSpacePIDController:
    """任务空间PID控制器"""
    
    def __init__(self, kp: float = 50.0, ki: float = 5.0, kd: float = 10.0,
                 integral_limit: float = 10.0, output_limit: float = 1.0):
        """
        初始化任务空间PID控制器
        
        Args:
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            integral_limit: 积分限幅
            output_limit: 输出限幅
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        
        self.reset()
    
    def reset(self):
        """重置控制器状态"""
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = 0.0
    
    def compute(self, target: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
        """
        计算控制输出
        
        Args:
            target: 目标位置 [x, y, z]
            current: 当前位置 [x, y, z]
            dt: 时间步长
            
        Returns:
            control_output: 控制输出 [dx, dy, dz]
        """
        error = target - current
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        
        # 微分项
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = np.zeros(3)
        d_term = self.kd * derivative
        
        # 总输出
        output = p_term + i_term + d_term
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        # 更新状态
        self.prev_error = error.copy()
        self.prev_time += dt
        
        return output

class TrajectoryGenerator:
    """轨迹生成器"""
    
    def __init__(self, center: np.ndarray, radius: float, height: float, 
                 start_angle: float = 0, end_angle: float = 2*np.pi):
        """
        初始化轨迹生成器
        
        Args:
            center: 圆心位置 [x, y, z]
            radius: 半径
            height: 高度偏移
            start_angle: 起始角度
            end_angle: 结束角度
        """
        self.center = center
        self.radius = radius
        self.height = height
        self.start_angle = start_angle
        self.end_angle = end_angle
    
    def generate_circular_trajectory(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成圆形轨迹
        
        Args:
            num_points: 轨迹点数
            
        Returns:
            (positions, velocities): 位置和速度数组
        """
        angles = np.linspace(self.start_angle, self.end_angle, num_points)
        
        positions = np.zeros((num_points, 3))
        velocities = np.zeros((num_points, 3))
        
        for i, angle in enumerate(angles):
            # 计算位置
            x = self.center[0] + self.radius * np.cos(angle)
            y = self.center[1] + self.radius * np.sin(angle)
            z = self.center[2] + self.height
            
            positions[i] = [x, y, z]
            
            # 计算速度（如果i > 0）
            if i > 0:
                dt = (self.end_angle - self.start_angle) / (num_points - 1)
                angular_velocity = 1.0  # 角速度
                
                vx = -self.radius * angular_velocity * np.sin(angle)
                vy = self.radius * angular_velocity * np.cos(angle)
                vz = 0.0
                
                velocities[i] = [vx, vy, vz]
        
        return positions, velocities
    
    def generate_linear_trajectory(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                  num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成直线轨迹
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            num_points: 轨迹点数
            
        Returns:
            (positions, velocities): 位置和速度数组
        """
        t = np.linspace(0, 1, num_points)
        
        positions = np.zeros((num_points, 3))
        velocities = np.zeros((num_points, 3))
        
        for i, ti in enumerate(t):
            # 线性插值
            positions[i] = start_pos + ti * (end_pos - start_pos)
            
            # 速度（如果i > 0）
            if i > 0:
                velocities[i] = (end_pos - start_pos) / (num_points - 1)
        
        return positions, velocities

class PandaMujocoController:
    """基于MuJoCo虚拟电机的Panda控制器"""
    
    def __init__(self, urdf_path: str, xml_path: str):
        """
        初始化Panda MuJoCo控制器
        
        Args:
            urdf_path: URDF文件路径
            xml_path: MuJoCo XML文件路径
        """
        self.urdf_path = urdf_path
        self.xml_path = xml_path
        
        # 初始化MuJoCo
        self.model = None
        self.data = None
        self._init_mujoco()
        
        # 初始化ikpy链
        self.chain = None
        self.num_joints = 7
        self._init_ikpy_chain()
        
        # 设置机械臂到安全的home姿态
        self._set_home_position()
        
        # 坐标系偏移补偿（初始化为零，由一致性验证计算）
        self.coordinate_offset = np.zeros(3)
        
        # 初始化控制器（使用保守的PID参数）
        self.task_pid = TaskSpacePIDController(kp=100.0, ki=0.0, kd=1.0, output_limit=10)
        self.joint_pids = [JointPIDController() for _ in range(self.num_joints)]
        
        # 夹爪控制器
        self.gripper_pid = JointPIDController(kp=200.0, ki=20.0, kd=10.0)
        
        # 夹爪参数
        self.gripper_open_position = 0.04   # 夹爪完全打开位置
        self.gripper_closed_position = 0.0  # 夹爪完全闭合位置
        self.gripper_force_limit = 50.0     # 夹爪最大力
        self.gripper_state = "open"         # 夹爪状态: "open", "closed", "grasping"
        self.gripper_target_position = self.gripper_open_position
        
        # 轨迹生成器
        self.trajectory_generator = None
        
        # 数据记录
        self.target_positions = []
        self.actual_positions = []
        self.joint_positions = []
        self.joint_velocities = []
        self.motor_torques = []
        self.control_errors = []
        self.timestamps = []
        
        # 错误处理
        self.ik_error_count = 0
        self.max_ik_errors = 5
        self.last_ik_error = ""
    
    def _init_mujoco(self):
        """初始化MuJoCo"""
        try:
            if not os.path.exists(self.xml_path):
                raise FileNotFoundError(f"XML file not found: {self.xml_path}")
            
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
            
            print(f"✅ MuJoCo model loaded: {self.xml_path}")
            print(f"✅ Actuators: {self.model.nu}")
            print(f"✅ Joints: {self.model.njnt}")
            
        except Exception as e:
            print(f"❌ Failed to load MuJoCo model: {e}")
            self.model = None
            self.data = None
    
    def _set_home_position(self):
        """设置机械臂到安全的home姿态"""
        if self.model is None or self.data is None:
            return
            
        # 使用适合mjx环境的安全home姿态
        home_joints = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        
        try:
            # 设置关节位置
            if len(home_joints) <= len(self.data.qpos):
                self.data.qpos[:len(home_joints)] = home_joints
                
                # 前向运动学计算
                mujoco.mj_forward(self.model, self.data)
                
                # 获取并显示home位置
                home_pos = self.get_end_effector_position()
                print(f"✅ 设置home姿态: [{home_pos[0]:.3f}, {home_pos[1]:.3f}, {home_pos[2]:.3f}]")
            else:
                print("⚠️ 关节数量不匹配，跳过home姿态设置")
                
        except Exception as e:
            print(f"⚠️ 设置home姿态失败: {e}")
    

    def _init_ikpy_chain(self):
        """初始化ikpy链"""
        try:
            if not os.path.exists(self.urdf_path):
                raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
            
            # 应用ikpy修复
            if not hasattr(np, 'float'):
                np.float = float
            
            self.chain = ikpy.chain.Chain.from_urdf_file(
                self.urdf_path,
                base_elements=["panda_link0"],
                active_links_mask=[False] + [True] * 7 + [False] * 3
            )
            
            print(f"✅ ikpy chain initialized with {len(self.chain.links)} links")
            
            # 验证ikpy链与MuJoCo模型的一致性
            self._validate_ikpy_consistency()
            
            # 关节名映射
            self.ikpy_joint_names = [link.name for link in self.chain.links if 'joint' in link.name][:7]
            self.mujoco_joint_names = []
            for i in range(self.model.njnt):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if name and name.startswith('joint'):
                    self.mujoco_joint_names.append(name)
            
            # 建立ikpy到mujoco的索引映射
            self.ikpy_to_mujoco = []
            for mujoco_name in self.mujoco_joint_names:
                found = False
                for idx, ikpy_name in enumerate(self.ikpy_joint_names):
                    if mujoco_name.replace('joint', '') == ikpy_name.replace('panda_joint', ''):
                        self.ikpy_to_mujoco.append(idx)
                        found = True
                        break
                if not found:
                    self.ikpy_to_mujoco.append(-1)
            
            print(f"关节名映射: {self.ikpy_to_mujoco}")
            
        except Exception as e:
            print(f"❌ Failed to initialize ikpy chain: {e}")
            self.chain = None
    
    def _validate_ikpy_consistency(self):
        """验证ikpy链与MuJoCo模型的一致性"""
        if self.chain is None or self.data is None:
            print("⚠️ 跳过一致性验证: ikpy链或MuJoCo数据未初始化")
            return
            
        try:
            # 获取home位置的关节角度
            home_joints = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
            
            # MuJoCo正运动学
            for i in range(self.num_joints):
                self.data.qpos[i] = home_joints[i]
            mujoco.mj_forward(self.model, self.data)
            mujoco_position = self.get_end_effector_position()
            
            # ikpy正运动学
            full_joints = self.chain.active_to_full(home_joints, [0] * len(self.chain.links))
            fk_result = self.chain.forward_kinematics(full_joints)
            ikpy_position = fk_result[:3, 3]
            
            # 计算位置差异
            position_diff = np.linalg.norm(ikpy_position - mujoco_position)
            
            print(f"🔍 一致性验证:")
            print(f"   MuJoCo位置: [{mujoco_position[0]:.3f}, {mujoco_position[1]:.3f}, {mujoco_position[2]:.3f}]")
            print(f"   ikpy正运动学: [{ikpy_position[0]:.3f}, {ikpy_position[1]:.3f}, {ikpy_position[2]:.3f}]")
            print(f"   位置差异: {position_diff*1000:.1f}mm")
            
            if position_diff > 0.01:  # 1cm阈值
                print(f"⚠️ 警告: ikpy链与MuJoCo模型存在显著差异 ({position_diff*1000:.1f}mm)")
                print("   计算坐标系补偿...")
                
                # 计算坐标变换偏移
                self.coordinate_offset = mujoco_position - ikpy_position
                print(f"🔧 坐标补偿: [{self.coordinate_offset[0]:.3f}, {self.coordinate_offset[1]:.3f}, {self.coordinate_offset[2]:.3f}]")
                
                # 验证补偿是否符合预期（基于URDF分析）
                expected_z_offset = 0.107  # joint8的固定偏移
                actual_z_offset = self.coordinate_offset[2]
                
                if abs(actual_z_offset - expected_z_offset) < 0.06:  # 6cm阈值内
                    print("✅ 偏移符合URDF中joint8的固定变换")
                else:
                    print(f"⚠️ Z轴偏移异常: 期望{expected_z_offset:.3f}m, 实际{actual_z_offset:.3f}m")
            else:
                print("✅ ikpy链与MuJoCo模型一致性良好")
                # 不重置偏移量，保持零值
                
        except Exception as e:
            print(f"❌ 一致性验证失败: {e}")
            # 默认使用已知的165mm偏移
            self.coordinate_offset = np.array([0.0, 0.0, 0.165])
            print(f"🔧 使用默认坐标补偿: {self.coordinate_offset}")
    
    def set_trajectory(self, trajectory_generator: TrajectoryGenerator):
        """设置轨迹生成器"""
        self.trajectory_generator = trajectory_generator
    
    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        if self.data is None:
            return np.array([0.0, 0.0, 0.0])
        
        # 使用MuJoCo获取末端位置
        mujoco.mj_forward(self.model, self.data)
        try:
            # 直接使用link7（手腕位置）
            end_effector_id = self.model.body("link7").id
            return self.data.xpos[end_effector_id].copy()
        except:
            # 备选：hand位置
            try:
                end_effector_id = self.model.body("hand").id
                return self.data.xpos[end_effector_id].copy()
            except:
                # 如果都找不到，返回默认位置
                return np.array([0.0, 0.0, 0.0])
    
    def get_joint_positions(self) -> np.ndarray:
        """获取关节位置"""
        if self.data is None:
            return np.zeros(self.num_joints)
        return self.data.qpos[:self.num_joints].copy()
    
    def get_joint_velocities(self) -> np.ndarray:
        """获取关节速度"""
        if self.data is None:
            return np.zeros(self.num_joints)
        return self.data.qvel[:self.num_joints].copy()
    
    def inverse_kinematics(self, target_position: np.ndarray, 
                          initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        逆运动学求解
        
        Args:
            target_position: 手腕（link7）目标位置 [x, y, z]
            initial_guess: 初始猜测
            
        Returns:
            joint_positions: 关节位置
            
        注意：直接控制link7，无偏移补偿
        """
        if self.chain is None:
            if self.ik_error_count < self.max_ik_errors:
                print("Warning: ikpy chain not initialized, returning default joint positions")
                self.ik_error_count += 1
            return np.zeros(self.num_joints)
        
        if initial_guess is None:
            initial_guess = self.get_joint_positions()
        
        # 应用坐标系补偿
        if hasattr(self, 'coordinate_offset') and np.any(self.coordinate_offset != 0):
            ik_target_position = target_position - self.coordinate_offset
        else:
            ik_target_position = target_position.copy()
        
        # 检查补偿后的目标位置是否在合理范围内
        if np.any(np.abs(ik_target_position) > 1.0):
            if self.ik_error_count < self.max_ik_errors:
                print(f"Warning: IK target position {ik_target_position} is out of reasonable range (wrist target: {target_position})")
                self.ik_error_count += 1
            return initial_guess
        
        try:
            # 设置目标方向（保持当前方向）
            target_orientation = [
                [1, 0, 0],  # X轴方向
                [0, 0, 1],  # Y轴方向
                [0, -1, 0]  # Z轴方向
            ]
            
            # 准备初始关节角度
            if len(initial_guess) != len(self.chain.links):
                full_joints = self.chain.active_to_full(initial_guess, [0] * len(self.chain.links))
            else:
                full_joints = initial_guess.copy()
            
            # 创建目标变换矩阵（使用补偿后的位置）
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = ik_target_position
            target_matrix[:3, :3] = target_orientation
            
            # 使用ikpy求解逆运动学
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                
                joint_angles = self.chain.inverse_kinematics(
                    target=target_matrix,
                    initial_position=full_joints
                )
            
            # 提取活动关节角度
            active_joint_angles = self.chain.active_from_full(joint_angles)
            
            # 检查解的有效性
            if np.any(np.isnan(active_joint_angles)) or np.any(np.isinf(active_joint_angles)):
                if self.ik_error_count < self.max_ik_errors:
                    print(f"Warning: Invalid IK solution for target {target_position}")
                    self.ik_error_count += 1
                return initial_guess
            
            # 检查关节限位
            joint_limits = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                                   [-3.0718, -0.0698], [-2.8973, 2.8973], [-0.0175, 3.7525],
                                   [-2.8973, 2.8973]])
            
            for i, angle in enumerate(active_joint_angles):
                if i < len(joint_limits):
                    if angle < joint_limits[i][0] or angle > joint_limits[i][1]:
                        if self.ik_error_count < self.max_ik_errors:
                            print(f"Warning: Joint {i} angle {angle:.3f} out of limits {joint_limits[i]}")
                            self.ik_error_count += 1
                        return initial_guess
            
            return active_joint_angles
            
        except Exception as e:
            error_msg = str(e)
            if error_msg != self.last_ik_error:
                if self.ik_error_count < self.max_ik_errors:
                    print(f"Inverse kinematics failed: {error_msg}")
                    self.ik_error_count += 1
                self.last_ik_error = error_msg
            return initial_guess
    
    def compute_joint_torques(self, target_joints: np.ndarray, dt: float) -> np.ndarray:
        """
        计算关节力矩（使用PID控制器）
        
        Args:
            target_joints: 目标关节角度
            dt: 时间步长
            
        Returns:
            torques: 关节力矩
        """
        current_joints = self.get_joint_positions()
        current_velocities = self.get_joint_velocities()
        
        torques = np.zeros(self.num_joints)
        
        for i in range(self.num_joints):
            # 使用关节PID控制器计算力矩
            torque = self.joint_pids[i].compute(
                target_joints[i], 
                current_joints[i], 
                current_velocities[i], 
                dt
            )
            torques[i] = torque
        
        return torques
    
    def move_to_position_with_trajectory(self, target_position: np.ndarray, duration: float = 3.0, num_points: int = 50):
        """
        使用轨迹生成的点到点控制
        
        Args:
            target_position: 目标位置
            duration: 运动时间
            num_points: 轨迹点数
            
        Returns:
            success: 是否成功
        """
        print(f"🎯 生成直线轨迹到目标位置: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        # 1. 获取当前位置
        current_pos = self.get_end_effector_position()
        print(f"📍 当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        
        # 2. 生成直线轨迹
        generator = TrajectoryGenerator(center=np.array([0, 0, 0]), radius=0.1, height=0)
        positions, velocities = generator.generate_linear_trajectory(current_pos, target_position, num_points)
        
        # 3. 跟踪轨迹
        print(f"🚀 开始轨迹跟踪，预计用时: {duration:.1f}s")
        self.run_trajectory(duration, dt=duration/num_points)
        
        # 4. 检查最终误差
        final_pos = self.get_end_effector_position()
        final_error = np.linalg.norm(final_pos - target_position)
        print(f"✅ 轨迹完成，最终误差: {final_error*1000:.1f}mm")
        
        return final_error < 0.02  # 2cm精度认为成功
    
    def step(self, target_position: np.ndarray, dt: float) -> np.ndarray:
        """
        执行一个控制步
        
        Args:
            target_position: 目标位置
            dt: 时间步长
            
        Returns:
            control_output: 控制输出
        """
        # 检查MuJoCo模型是否已正确加载
        if self.model is None or self.data is None:
            print("❌ MuJoCo模型未正确加载，无法执行控制步")
            return np.zeros(3)
        
        # 获取当前位置
        current_position = self.get_end_effector_position()
        
        # 任务空间PID控制
        task_output = self.task_pid.compute(target_position, current_position, dt)
        
        # 计算期望位置，限制增量避免发散
        position_increment = task_output * dt
        # 使用固定的保守步长
        max_increment = 0.02  # 固定2cm步长，避免振荡
        increment_magnitude = np.linalg.norm(position_increment)
        if increment_magnitude > max_increment:
            position_increment = position_increment * (max_increment / increment_magnitude)
        
        desired_position = current_position + position_increment
        
        # 确保目标位置在合理范围内
        desired_position = np.clip(desired_position, [-1.0, -1.0, 0.0], [1.0, 1.0, 1.0])
        
        # 逆运动学求解目标关节角度
        target_joints = self.inverse_kinematics(desired_position)
        
        # 调试信息：检查IK求解
        if target_joints is None:
            distance = np.linalg.norm(target_position - current_position)
            if distance > 0.1:  # 只在大误差时显示
                print(f"⚠️ IK求解失败，误差: {distance*1000:.1f}mm")
            return np.zeros(self.num_joints)
        
        # 调试信息（仅在大误差时显示）
        position_error = np.linalg.norm(target_position - current_position)
        if position_error > 0.1:  # 误差大于10cm时显示调试信息
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 1
            
            if self._debug_count % 100 == 0:  # 每1秒显示一次
                print(f"调试: 目标={target_position}, 当前={current_position}, 期望={desired_position}, 误差={position_error:.3f}m")
        
        # 计算关节力矩
        torques = self.compute_joint_torques(target_joints, dt)
        
        # 应用MuJoCo执行器力矩限制 (±87 N·m)
        torques = np.clip(torques, -87.0, 87.0)
        
        # 设置关节力矩（使用MuJoCo的ctrl）
        if self.data.ctrl.size >= self.num_joints:
            self.data.ctrl[:self.num_joints] = torques
        else:
            print(f"❌ 控制器维度不匹配: 需要{self.num_joints}, 实际{self.data.ctrl.size}")
        
        return task_output
    

    def get_end_effector_velocity(self) -> np.ndarray:
        """获取末端执行器速度"""
        if self.model is None or self.data is None:
            return np.zeros(3)
        
        try:
            # 简化实现：通过位置差分计算速度
            if hasattr(self, '_last_ee_position') and hasattr(self, '_last_time'):
                current_pos = self.get_end_effector_position()
                current_time = self.data.time
                dt = current_time - self._last_time
                
                if dt > 0:
                    velocity = (current_pos - self._last_ee_position) / dt
                    self._last_ee_position = current_pos.copy()
                    self._last_time = current_time
                    return velocity
            
            # 初始化
            self._last_ee_position = self.get_end_effector_position().copy()
            self._last_time = self.data.time
            return np.zeros(3)
            
        except Exception as e:
            return np.zeros(3)
    
    def open_gripper(self):
        """打开夹爪"""
        self.gripper_target_position = self.gripper_open_position
        self.gripper_state = "open"
        print("夹爪正在打开...")
    
    def close_gripper(self):
        """闭合夹爪"""
        self.gripper_target_position = self.gripper_closed_position
        self.gripper_state = "closed"
        print("夹爪正在闭合...")
    
    def set_gripper_position(self, position: float):
        """设置夹爪位置
        
        Args:
            position: 夹爪位置 (0.0-0.04m)
        """
        self.gripper_target_position = np.clip(position, 
                                             self.gripper_closed_position, 
                                             self.gripper_open_position)
        if position <= 0.01:
            self.gripper_state = "closed"
        elif position >= 0.035:
            self.gripper_state = "open"
        else:
            self.gripper_state = "grasping"
    
    def get_gripper_position(self) -> float:
        """获取当前夹爪位置"""
        if self.model is None or self.data is None:
            return 0.0
            
        try:
            # 获取finger_joint1的位置（两个手指是同步的）
            finger_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
            if finger_joint_id >= 0:
                return self.data.qpos[finger_joint_id]
            else:
                print("Warning: finger_joint1 not found")
                return 0.0
        except Exception as e:
            print(f"Error getting gripper position: {e}")
            return 0.0
    
    def get_gripper_force(self) -> float:
        """获取夹爪当前力"""
        if self.model is None or self.data is None:
            return 0.0
            
        try:
            # 方法1: 使用执行器力
            actuator8_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
            if actuator8_id >= 0 and len(self.data.actuator_force) > actuator8_id:
                actuator_force = abs(self.data.actuator_force[actuator8_id])
                if actuator_force > 1e-6:  # 如果执行器力有值，使用它
                    return actuator_force
            
            # 方法2: 使用关节力矩
            finger_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
            if finger_joint_id >= 0 and len(self.data.qfrc_actuator) > finger_joint_id:
                joint_torque = abs(self.data.qfrc_actuator[finger_joint_id])
                if joint_torque > 1e-6:  # 如果关节力矩有值，使用它
                    return joint_torque
            
            # 方法3: 基于控制输入估算力（备用方法）
            if actuator8_id >= 0 and len(self.data.ctrl) > actuator8_id:
                control_val = abs(self.data.ctrl[actuator8_id])
                # 根据XML中的gainprm="350 0 0"，估算实际力
                estimated_force = control_val * 3.5  # 缩放因子，使力的数值更合理
                return estimated_force
            
            return 0.0
            
        except Exception as e:
            print(f"Error getting gripper force: {e}")
            return 0.0
    
    def get_gripper_contact_force(self) -> float:
        """获取夹爪接触力（当夹爪接触物体时）"""
        if self.model is None or self.data is None:
            return 0.0
            
        try:
            total_contact_force = 0.0
            
            # 遍历所有接触点
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                
                # 获取接触的几何体名称
                geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) if contact.geom1 >= 0 else ""
                geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) if contact.geom2 >= 0 else ""
                
                # 检查是否涉及夹爪（finger）
                if geom1_name and ("finger" in geom1_name.lower() or "hand" in geom1_name.lower()):
                    # 计算接触力的大小
                    contact_force_magnitude = np.linalg.norm(contact.force)
                    total_contact_force += contact_force_magnitude
                elif geom2_name and ("finger" in geom2_name.lower() or "hand" in geom2_name.lower()):
                    contact_force_magnitude = np.linalg.norm(contact.force)
                    total_contact_force += contact_force_magnitude
            
            return total_contact_force
            
        except Exception as e:
            print(f"Error getting contact force: {e}")
            return 0.0
    
    def control_gripper(self, dt: float):
        """控制夹爪运动
        
        Args:
            dt: 时间步长
        """
        if self.model is None or self.data is None:
            return
            
        current_position = self.get_gripper_position()
        current_velocity = 0.0  # 简化：假设速度为0
        
        # 使用PID控制器计算夹爪力矩
        torque = self.gripper_pid.compute(
            self.gripper_target_position,
            current_position, 
            current_velocity,
            dt
        )
        
        # 限制力矩
        torque = np.clip(torque, -self.gripper_force_limit, self.gripper_force_limit)
        
        # 设置夹爪力矩
        try:
            # 使用第8个执行器（actuator8）控制夹爪
            if self.data.ctrl.size > self.num_joints:
                # actuator8对应索引7（从0开始）
                self.data.ctrl[7] = torque * 100  # 放大信号，因为可能需要更大的控制量
                return
                
        except Exception as e:
            print(f"Warning: 夹爪控制失败: {e}")
    
    def is_object_grasped(self) -> bool:
        """检测是否成功抓取物体"""
        gripper_force = self.get_gripper_force()
        gripper_position = self.get_gripper_position()
        contact_force = self.get_gripper_contact_force()
        
        # 检测条件
        position_indicates_grasp = 0.005 < gripper_position < 0.03  # 夹爪部分闭合
        force_indicates_grasp = gripper_force > 1.0  # 有执行器力
        contact_indicates_grasp = contact_force > 0.1  # 有接触力
        
        # 满足位置条件且有力输出或接触力，认为抓取到物体
        return position_indicates_grasp and (force_indicates_grasp or contact_indicates_grasp)
    
    def step_with_gripper(self, target_position: np.ndarray, dt: float) -> np.ndarray:
        """执行一个控制步（包含夹爪控制）
        
        Args:
            target_position: 目标位置
            dt: 时间步长
            
        Returns:
            control_output: 控制输出
        """
        # 执行机械臂控制
        control_output = self.step(target_position, dt)
        
        # 执行夹爪控制
        self.control_gripper(dt)
        
        return control_output
    
    def run_trajectory(self, duration: float = 1.0, dt: float = 0.01) -> None:
        """
        直接轨迹跟踪（关节空间控制）
        
        Args:
            duration: 运行时间
            dt: 时间步长
        """
        if self.trajectory_generator is None:
            raise ValueError("请先设置轨迹生成器")
        
        # 生成轨迹
        positions, velocities = self.trajectory_generator.generate_circular_trajectory(
            int(duration / dt)
        )
        
        print("开始直接轨迹跟踪（关节空间控制）...")
        start_time = time.time()
        
        # 清空数据记录
        self.target_positions = []
        self.actual_positions = []
        self.joint_positions = []
        self.joint_velocities = []
        self.motor_torques = []
        self.control_errors = []
        self.timestamps = []
        
        # 重置错误计数器
        self.ik_error_count = 0
        self.last_ik_error = ""
        
        # 直接关节空间轨迹跟踪
        for i, target_pos in enumerate(positions):
            current_time = i * dt
            self.timestamps.append(current_time)
            
            # 逆运动学求解目标关节角度
            target_joints = self.inverse_kinematics(target_pos)
            
            if target_joints is not None:
                # 直接设置关节目标角度（跳过任务空间PID）
                current_joints = self.get_joint_positions()
                
                # 计算关节力矩
                torques = np.zeros(self.num_joints)
                for j in range(self.num_joints):
                    # 关节PID控制
                    torque = self.joint_pids[j].compute(
                        target_joints[j], 
                        current_joints[j], 
                        0.0,  # 简化：忽略速度
                        dt
                    )
                    torques[j] = torque
                
                # 应用MuJoCo执行器力矩限制 (±87 N·m)
                torques = np.clip(torques, -87.0, 87.0)
                
                # 设置控制输出
                if self.data.ctrl.size >= self.num_joints:
                    self.data.ctrl[:self.num_joints] = torques
            
            # 步进仿真
            mujoco.mj_step(self.model, self.data)
            
            # 记录数据
            self.target_positions.append(target_pos.copy())
            self.actual_positions.append(self.get_end_effector_position().copy())
            self.joint_positions.append(self.get_joint_positions().copy())
            self.joint_velocities.append(self.get_joint_velocities().copy())
            self.motor_torques.append(self.data.ctrl[:self.num_joints].copy())
            self.control_errors.append(target_pos - self.get_end_effector_position())
            
            # 显示进度
            if i % 50 == 0 or i == len(positions) - 1:
                progress = (i + 1) / len(positions) * 100
                current_error = np.linalg.norm(target_pos - self.get_end_effector_position())
                print(f"Progress: {progress:.1f}% ({i+1}/{len(positions)}), 当前误差: {current_error*1000:.1f}mm")
        
        end_time = time.time()
        print(f"轨迹跟踪完成，用时: {end_time - start_time:.2f}s")
    
    def run_mujoco_simulation(self, duration: float = 10.0, dt: float = 0.01):
        """
        运行MuJoCo仿真
        
        Args:
            duration: 仿真时间
            dt: 时间步长
        """
        if self.trajectory_generator is None:
            raise ValueError("Please set trajectory generator first")
        
        if self.model is None or self.data is None:
            print("❌ MuJoCo model not loaded, cannot run simulation")
            return
        
        # 生成轨迹
        positions, velocities = self.trajectory_generator.generate_circular_trajectory(
            int(duration / dt)
        )
        
        # 重置控制器
        self.task_pid.reset()
        for pid in self.joint_pids:
            pid.reset()
        
        # 重置错误计数器
        self.ik_error_count = 0
        self.last_ik_error = ""
        
        # 创建MuJoCo查看器
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            print("Starting MuJoCo simulation (Virtual Motors)...")
            print("Press Ctrl+C to exit simulation")
            
            # 设置初始关节位置
            if len(positions) > 0:
                initial_joints = self.inverse_kinematics(positions[0])
                self.data.qpos[:self.num_joints] = initial_joints
                mujoco.mj_forward(self.model, self.data)
            
            try:
                for i, target_pos in enumerate(positions):
                    # 执行控制步
                    control_output = self.step(target_pos, dt)
                    
                    # 更新MuJoCo
                    mujoco.mj_step(self.model, self.data)
                    
                    # 强制刷新viewer
                    try:
                        if hasattr(viewer, 'sync'):
                            viewer.sync()
                        elif hasattr(viewer, 'render'):
                            viewer.render()
                    except:
                        pass
                    
                    # 调试信息
                    if i % 50 == 0:
                        current_pos = self.get_end_effector_position()
                        print(f"Step {i}: Target={target_pos}, Current={current_pos}")
                    
                    # 控制仿真速度
                    time.sleep(dt)
                    
            except KeyboardInterrupt:
                print("Simulation interrupted by user")
            finally:
                print("Simulation ended")
                
        except Exception as e:
            print(f"❌ MuJoCo viewer failed to start: {e}")
    
    def visualize_trajectory(self):
        """可视化轨迹跟踪结果"""
        if not self.target_positions:
            print("No trajectory data to visualize")
            return
        
        target_positions = np.array(self.target_positions)
        actual_positions = np.array(self.actual_positions)
        control_errors = np.array(self.control_errors)
        joint_positions = np.array(self.joint_positions)
        joint_velocities = np.array(self.joint_velocities)
        motor_torques = np.array(self.motor_torques)
        
        # 创建图形
        fig = plt.figure(figsize=(20, 12))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(3, 4, 1, projection='3d')
        ax1.plot(target_positions[:, 0], target_positions[:, 1], target_positions[:, 2], 
                'b-', label='Target Trajectory', linewidth=2)
        ax1.plot(actual_positions[:, 0], actual_positions[:, 1], actual_positions[:, 2], 
                'r--', label='Actual Trajectory', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory Tracking')
        ax1.legend()
        ax1.grid(True)
        
        # 位置误差图
        ax2 = fig.add_subplot(3, 4, 2)
        error_magnitude = np.linalg.norm(control_errors, axis=1)
        ax2.plot(self.timestamps, error_magnitude, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error')
        ax2.grid(True)
        
        # X方向跟踪
        ax3 = fig.add_subplot(3, 4, 3)
        ax3.plot(self.timestamps, target_positions[:, 0], 'b-', label='Target', linewidth=2)
        ax3.plot(self.timestamps, actual_positions[:, 0], 'r--', label='Actual', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('X Position (m)')
        ax3.set_title('X Position Tracking')
        ax3.legend()
        ax3.grid(True)
        
        # Y方向跟踪
        ax4 = fig.add_subplot(3, 4, 4)
        ax4.plot(self.timestamps, target_positions[:, 1], 'b-', label='Target', linewidth=2)
        ax4.plot(self.timestamps, actual_positions[:, 1], 'r--', label='Actual', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Y Position Tracking')
        ax4.legend()
        ax4.grid(True)
        
        # 关节位置
        ax5 = fig.add_subplot(3, 4, 5)
        for i in range(min(joint_positions.shape[1], 7)):
            ax5.plot(self.timestamps, joint_positions[:, i], label=f'Joint {i+1}')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Joint Position (rad)')
        ax5.set_title('Joint Positions')
        ax5.legend()
        ax5.grid(True)
        
        # 关节速度
        ax6 = fig.add_subplot(3, 4, 6)
        for i in range(min(joint_velocities.shape[1], 7)):
            ax6.plot(self.timestamps, joint_velocities[:, i], label=f'Joint {i+1}')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Joint Velocity (rad/s)')
        ax6.set_title('Joint Velocities')
        ax6.legend()
        ax6.grid(True)
        
        # 电机力矩
        ax7 = fig.add_subplot(3, 4, 7)
        for i in range(min(motor_torques.shape[1], 7)):
            ax7.plot(self.timestamps, motor_torques[:, i], label=f'Joint {i+1}')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Motor Torque (N⋅m)')
        ax7.set_title('Motor Torques')
        ax7.legend()
        ax7.grid(True)
        
        # 控制误差统计
        ax8 = fig.add_subplot(3, 4, 8)
        error_magnitude = np.linalg.norm(control_errors, axis=1)
        ax8.hist(error_magnitude, bins=20, alpha=0.7, color='red')
        ax8.set_xlabel('Position Error (m)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Error Distribution')
        ax8.grid(True)
        
        # Z方向跟踪
        ax9 = fig.add_subplot(3, 4, 9)
        ax9.plot(self.timestamps, target_positions[:, 2], 'b-', label='Target', linewidth=2)
        ax9.plot(self.timestamps, actual_positions[:, 2], 'r--', label='Actual', linewidth=2)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Z Position (m)')
        ax9.set_title('Z Position Tracking')
        ax9.legend()
        ax9.grid(True)
        
        # 任务空间PID输出
        ax10 = fig.add_subplot(3, 4, 10)
        task_outputs = np.array(self.control_errors) * 50  # 近似任务空间输出
        for i in range(3):
            ax10.plot(self.timestamps, task_outputs[:, i], label=f'Axis {i+1}')
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Task Space Output')
        ax10.set_title('Task Space PID Output')
        ax10.legend()
        ax10.grid(True)
        
        # 关节位置误差
        ax11 = fig.add_subplot(3, 4, 11)
        joint_errors = np.abs(joint_positions - joint_positions[0])
        for i in range(min(joint_errors.shape[1], 7)):
            ax11.plot(self.timestamps, joint_errors[:, i], label=f'Joint {i+1}')
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Joint Position Error (rad)')
        ax11.set_title('Joint Position Errors')
        ax11.legend()
        ax11.grid(True)
        
        # 力矩统计
        ax12 = fig.add_subplot(3, 4, 12)
        torque_magnitude = np.linalg.norm(motor_torques, axis=1)
        ax12.hist(torque_magnitude, bins=20, alpha=0.7, color='blue')
        ax12.set_xlabel('Torque Magnitude (N⋅m)')
        ax12.set_ylabel('Frequency')
        ax12.set_title('Torque Distribution')
        ax12.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print("\n=== 轨迹跟踪统计 ===")
        error_magnitude = np.linalg.norm(control_errors, axis=1)
        print(f"平均位置误差: {np.mean(error_magnitude):.6f} m")
        print(f"最大位置误差: {np.max(error_magnitude):.6f} m")
        print(f"位置误差标准差: {np.std(error_magnitude):.6f} m")
        
        torque_magnitude = np.linalg.norm(motor_torques, axis=1)
        print(f"平均力矩: {np.mean(torque_magnitude):.6f} N⋅m")
        print(f"最大力矩: {np.max(torque_magnitude):.6f} N⋅m")
        print(f"力矩标准差: {np.std(torque_magnitude):.6f} N⋅m")

def main():
    """主函数 - 使用重构后的系统"""
    print("⚠️ 注意: 建议使用重构后的系统 'python pid_panda_refactored.py'")
    print("当前使用旧版本控制器，功能有限\n")
    
    # 文件路径 - 使用相对于脚本文件的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    urdf_path = os.path.join(project_root, "models", "franka_emika_panda", "frankaEmikaPanda.urdf")
    xml_path = os.path.join(project_root, "models", "franka_emika_panda", "scene.xml")
    
    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"错误: URDF文件不存在: {urdf_path}")
        return
    
    if not os.path.exists(xml_path):
        print(f"错误: XML文件不存在: {xml_path}")
        return
    
    # 创建Panda MuJoCo控制器
    print("初始化Panda MuJoCo控制器...")
    controller = PandaMujocoController(urdf_path, xml_path)
    
    # 创建圆弧轨迹
    print("创建圆弧轨迹...")
    center = np.array([0.5, 0.0, 0.4])  # 圆心
    radius = 0.15  # 半径
    height = 0.0  # 高度偏移
    trajectory = TrajectoryGenerator(center, radius, height)
    
    # 设置轨迹
    controller.set_trajectory(trajectory)
    
    # 运行轨迹跟踪
    print("运行轨迹跟踪（MuJoCo虚拟电机）...")
    controller.run_trajectory(duration=2.0, dt=0.01)
    
    # 可视化结果
    print("生成可视化结果...")
    controller.visualize_trajectory()
    
    # 询问是否运行MuJoCo仿真
    response = input("\n是否运行MuJoCo仿真? (y/n): ")
    if response.lower() == 'y':
        controller.run_mujoco_simulation(duration=2.0, dt=0.01)

if __name__ == "__main__":
    main()