#!/usr/bin/env python3
"""
Franka Panda机械臂PID控制对比系统
支持两种控制架构：
1. 直接关节空间控制 (Direct Joint Control)
2. 修复的任务空间控制 (Fixed Task Space Control)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import mujoco.viewer
import ikpy.chain
from typing import List, Tuple, Optional
import time
import os
import warnings

# ikpy修复
if not hasattr(np, 'float'):
    np.float = float

class TrajectoryGenerator:
    """轨迹生成器类"""
    
    def __init__(self, center: np.ndarray, radius: float, height: float, 
                 start_angle: float = 0, end_angle: float = 2*np.pi):
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.start_angle = start_angle
        self.end_angle = end_angle
    
    def generate_circular_trajectory(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """生成圆弧轨迹"""
        angles = np.linspace(self.start_angle, self.end_angle, num_points)
        positions = np.zeros((num_points, 3))
        
        for i, angle in enumerate(angles):
            positions[i] = np.array([
                self.center[0] + self.radius * np.cos(angle),
                self.center[1] + self.radius * np.sin(angle),
                self.center[2] + self.height * np.sin(2 * angle) * 0.1
            ])
        
        return positions, angles

class JointPIDController:
    """单关节PID控制器"""
    
    def __init__(self, kp: float = 100.0, ki: float = 5.0, kd: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.reset()
    
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
    
    def compute(self, target: float, current: float, dt: float) -> float:
        error = target - current
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项（防饱和）
        self.integral += error * dt
        self.integral = np.clip(self.integral, -1.0, 1.0)
        i_term = self.ki * self.integral
        
        # 微分项
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        
        self.prev_error = error
        return p_term + i_term + d_term

class FixedTaskSpacePIDController:
    """修复的任务空间PID控制器 - 输出速度而非位置增量"""
    
    def __init__(self, kp: float = 50.0, ki: float = 5.0, kd: float = 10.0, 
                 velocity_limit: float = 0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.velocity_limit = velocity_limit
        self.reset()
    
    def reset(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
    
    def compute(self, target: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
        """
        计算控制输出
        
        Returns:
            velocity: 目标速度 [vx, vy, vz] (m/s)
        """
        error = target - current
        error_magnitude = np.linalg.norm(error)
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项（仅在小误差时启用，防止积分饱和）
        if error_magnitude < 0.05:  # 5cm内才使用积分
            self.integral += error * dt
            self.integral = np.clip(self.integral, -0.1, 0.1)  # 限制积分项
        else:
            self.integral *= 0.9  # 大误差时衰减积分
        i_term = self.ki * self.integral
        
        # 微分项
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = np.zeros(3)
        d_term = self.kd * derivative
        
        # 输出速度
        velocity = p_term + i_term + d_term
        
        # 限制速度
        velocity_magnitude = np.linalg.norm(velocity)
        if velocity_magnitude > self.velocity_limit:
            velocity = velocity * (self.velocity_limit / velocity_magnitude)
        
        self.prev_error = error.copy()
        return velocity

class PandaController:
    """Panda控制器 - 支持两种控制模式"""
    
    def __init__(self, urdf_path: str, xml_path: str):
        self.urdf_path = urdf_path
        self.xml_path = xml_path
        self.num_joints = 7
        
        # 初始化MuJoCo
        self.model = None
        self.data = None
        self._init_mujoco()
        
        # 初始化ikpy链
        self.chain = None
        self._init_ikpy_chain()
        
        # 坐标系补偿
        self.coordinate_offset = np.zeros(3)
        self._validate_ikpy_consistency()
        
        # 控制器模式选择
        self.control_mode = "joint"  # "joint" 或 "task"
        
        # 方法1: 直接关节空间控制器
        joint_params = [
            {'kp': 300.0, 'ki': 10.0, 'kd': 20.0},   # Joint 1
            {'kp': 250.0, 'ki': 8.0, 'kd': 18.0},    # Joint 2
            {'kp': 200.0, 'ki': 6.0, 'kd': 15.0},    # Joint 3
            {'kp': 150.0, 'ki': 5.0, 'kd': 12.0},    # Joint 4
            {'kp': 120.0, 'ki': 4.0, 'kd': 10.0},    # Joint 5
            {'kp': 100.0, 'ki': 3.0, 'kd': 8.0},     # Joint 6
            {'kp': 80.0, 'ki': 2.0, 'kd': 6.0}       # Joint 7
        ]
        
        self.joint_pids = []
        for i, params in enumerate(joint_params):
            controller = JointPIDController(
                kp=params['kp'], 
                ki=params['ki'], 
                kd=params['kd']
            )
            self.joint_pids.append(controller)
        
        # 方法2: 修复的任务空间控制器
        self.task_pid = FixedTaskSpacePIDController(
            kp=50.0,
            ki=5.0,
            kd=10.0,
            velocity_limit=0.5
        )
        
        # 性能监控
        self.performance_monitor = {
            'position_errors': [],
            'control_outputs': [],
            'computation_times': [],
            'ik_success_rate': []
        }
        
        # 数据记录
        self.target_positions = []
        self.actual_positions = []
        self.joint_positions = []
        self.control_errors = []
        self.timestamps = []
        
        print(f"[OK] Panda控制器初始化完成")
        print(f"[INFO] 支持控制模式: joint (直接关节控制), task (任务空间控制)")
    
    def _init_mujoco(self):
        """初始化MuJoCo"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
            self.data = mujoco.MjData(self.model)
            print(f"[OK] MuJoCo model loaded: {self.xml_path}")
            print(f"[OK] Actuators: {self.model.nu}")
            print(f"[OK] Joints: {self.model.nq}")
        except Exception as e:
            print(f"[ERROR] Failed to load MuJoCo model: {e}")
            self.model = None
            self.data = None
    
    def _init_ikpy_chain(self):
        """初始化ikpy链"""
        try:
            self.chain = ikpy.chain.Chain.from_urdf_file(
                self.urdf_path,
                base_elements=["panda_link0"],
                active_links_mask=[False] + [True] * 7 + [False] * 3
            )
            print(f"[OK] ikpy chain initialized with {len(self.chain.links)} links")
        except Exception as e:
            print(f"[ERROR] Failed to initialize ikpy chain: {e}")
            self.chain = None
    
    def _validate_ikpy_consistency(self):
        """验证ikpy链与MuJoCo模型的一致性"""
        if self.chain is None or self.data is None:
            return
            
        try:
            # Home位置测试
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
            
            # 计算偏移
            position_diff = np.linalg.norm(ikpy_position - mujoco_position)
            
            if position_diff > 0.01:  # 1cm阈值
                self.coordinate_offset = mujoco_position - ikpy_position
                print(f"[FIX] 坐标补偿: {self.coordinate_offset}")
            else:
                print(f"[OK] ikpy与MuJoCo模型一致")
                
        except Exception as e:
            print(f"[WARNING] 一致性验证失败: {e}")
            self.coordinate_offset = np.array([0.0, 0.0, 0.165])  # 默认偏移
    
    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        try:
            end_effector_id = self.model.body("hand").id
            return self.data.xpos[end_effector_id].copy()
        except:
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
    
    def inverse_kinematics(self, target_position: np.ndarray) -> np.ndarray:
        """逆运动学求解"""
        if self.chain is None:
            return self.get_joint_positions()
        
        # 应用坐标补偿
        if np.any(self.coordinate_offset != 0):
            target_position = target_position - self.coordinate_offset
        
        try:
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_position
            
            current_joints = self.get_joint_positions()
            full_joints = self.chain.active_to_full(current_joints, [0] * len(self.chain.links))
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                joint_angles = self.chain.inverse_kinematics(
                    target=target_matrix,
                    initial_position=full_joints
                )
            
            active_joint_angles = self.chain.active_from_full(joint_angles)
            
            # 验证解的有效性
            if not (np.any(np.isnan(active_joint_angles)) or np.any(np.isinf(active_joint_angles))):
                return active_joint_angles
            
            return current_joints
            
        except Exception as e:
            return self.get_joint_positions()
    
    def set_control_mode(self, mode: str):
        """设置控制模式"""
        if mode in ["joint", "task"]:
            self.control_mode = mode
            print(f"[INFO] 控制模式设置为: {mode}")
            # 重置控制器状态
            for pid in self.joint_pids:
                pid.reset()
            self.task_pid.reset()
        else:
            print(f"[ERROR] 无效的控制模式: {mode}")
    
    def step_joint_control(self, target_position: np.ndarray, dt: float) -> np.ndarray:
        """方法1: 直接关节空间控制"""
        start_time = time.time()
        
        # 1. IK求解得到目标关节角度
        target_joints = self.inverse_kinematics(target_position)
        
        # 2. 关节空间PID控制
        current_joints = self.get_joint_positions()
        current_velocities = self.get_joint_velocities()
        
        torques = np.zeros(self.num_joints)
        for i in range(self.num_joints):
            torque = self.joint_pids[i].compute(
                target_joints[i], 
                current_joints[i], 
                dt
            )
            torques[i] = torque
        
        # 3. 应用力矩限制并设置控制
        torques = np.clip(torques, -87.0, 87.0)
        if self.data.ctrl.size >= self.num_joints:
            self.data.ctrl[:self.num_joints] = torques
        
        # 4. 记录性能
        computation_time = time.time() - start_time
        current_position = self.get_end_effector_position()
        error_magnitude = np.linalg.norm(target_position - current_position)
        
        self.performance_monitor['position_errors'].append(error_magnitude)
        self.performance_monitor['control_outputs'].append(np.linalg.norm(torques))
        self.performance_monitor['computation_times'].append(computation_time)
        
        return torques
    
    def step_task_control(self, target_position: np.ndarray, dt: float) -> np.ndarray:
        """方法2: 修复的任务空间控制"""
        start_time = time.time()
        
        # 1. 任务空间PID计算目标速度
        current_position = self.get_end_effector_position()
        target_velocity = self.task_pid.compute(target_position, current_position, dt)
        
        # 2. 积分得到期望位置
        desired_position = current_position + target_velocity * dt
        
        # 3. IK求解
        target_joints = self.inverse_kinematics(desired_position)
        
        # 4. 关节空间控制（简化版，仅比例控制）
        current_joints = self.get_joint_positions()
        joint_errors = target_joints - current_joints
        
        # 简单比例控制
        kp_joint = 50.0
        torques = kp_joint * joint_errors
        
        # 5. 应用力矩限制并设置控制
        torques = np.clip(torques, -87.0, 87.0)
        if self.data.ctrl.size >= self.num_joints:
            self.data.ctrl[:self.num_joints] = torques
        
        # 6. 记录性能
        computation_time = time.time() - start_time
        error_magnitude = np.linalg.norm(target_position - current_position)
        
        self.performance_monitor['position_errors'].append(error_magnitude)
        self.performance_monitor['control_outputs'].append(np.linalg.norm(torques))
        self.performance_monitor['computation_times'].append(computation_time)
        
        return torques
    
    def step(self, target_position: np.ndarray, dt: float) -> np.ndarray:
        """执行一步控制"""
        if self.control_mode == "joint":
            return self.step_joint_control(target_position, dt)
        else:
            return self.step_task_control(target_position, dt)
    
    def run_trajectory(self, positions: np.ndarray, duration: float = 3.0, dt: float = 0.01):
        """运行轨迹跟踪"""
        print(f"[INFO] 开始轨迹跟踪 - 控制模式: {self.control_mode}")
        
        # 清除历史数据
        self.target_positions = []
        self.actual_positions = []
        self.joint_positions = []
        self.control_errors = []
        self.timestamps = []
        self.performance_monitor = {
            'position_errors': [],
            'control_outputs': [],
            'computation_times': [],
            'ik_success_rate': []
        }
        
        start_time = time.time()
        
        for i, target_pos in enumerate(positions):
            current_time = time.time() - start_time
            
            # 执行控制步
            control_output = self.step(target_pos, dt)
            
            # 步进仿真
            mujoco.mj_step(self.model, self.data)
            
            # 记录数据
            actual_pos = self.get_end_effector_position()
            self.target_positions.append(target_pos.copy())
            self.actual_positions.append(actual_pos.copy())
            self.joint_positions.append(self.get_joint_positions().copy())
            self.timestamps.append(current_time)
            
            error = target_pos - actual_pos
            self.control_errors.append(error)
            
            # 显示进度
            if i % 50 == 0 or i == len(positions) - 1:
                progress = (i + 1) / len(positions) * 100
                error_magnitude = np.linalg.norm(error)
                print(f"Progress: {progress:.1f}% ({i+1}/{len(positions)}), "
                      f"误差: {error_magnitude*1000:.1f}mm")
        
        total_time = time.time() - start_time
        print(f"[OK] 轨迹跟踪完成，用时: {total_time:.2f}s")
    
    def print_performance_summary(self):
        """打印性能摘要"""
        if not self.performance_monitor['position_errors']:
            print("[WARNING] 没有性能数据")
            return
        
        errors = np.array(self.performance_monitor['position_errors'])
        outputs = np.array(self.performance_monitor['control_outputs'])
        times = np.array(self.performance_monitor['computation_times'])
        
        print(f"\n=== 性能摘要 ({self.control_mode.upper()} 模式) ===")
        print(f"平均位置误差: {np.mean(errors)*1000:.2f} mm")
        print(f"最大位置误差: {np.max(errors)*1000:.2f} mm")
        print(f"误差标准差: {np.std(errors)*1000:.2f} mm")
        print(f"95%误差分位数: {np.percentile(errors, 95)*1000:.2f} mm")
        print("-" * 50)
        print(f"平均控制输出: {np.mean(outputs):.4f}")
        print(f"最大控制输出: {np.max(outputs):.4f}")
        print(f"平均计算时间: {np.mean(times)*1000:.2f} ms")
        
        # 控制精度评估
        avg_error_mm = np.mean(errors) * 1000
        if avg_error_mm < 1:
            print("[EXCELLENT] 控制精度: 优秀 (< 1mm)")
        elif avg_error_mm < 5:
            print("[GOOD] 控制精度: 良好 (< 5mm)")
        elif avg_error_mm < 10:
            print("[ACCEPTABLE] 控制精度: 可接受 (< 10mm)")
        else:
            print("[POOR] 控制精度: 需要改进 (> 10mm)")

def main():
    """主函数"""
    print("="*60)
    print("Panda机械臂PID控制方法对比系统")
    print("="*60)
    
    # 路径设置
    urdf_path = "../models/franka_emika_panda/frankaEmikaPanda.urdf"
    xml_path = "../models/franka_emika_panda/scene.xml"
    
    # 如果相对路径不存在，尝试绝对路径
    if not os.path.exists(urdf_path):
        urdf_path = "models/franka_emika_panda/frankaEmikaPanda.urdf"
        xml_path = "models/franka_emika_panda/scene.xml"
    
    # 检查文件存在
    if not os.path.exists(urdf_path):
        print(f"[ERROR] URDF文件不存在: {urdf_path}")
        return
    if not os.path.exists(xml_path):
        print(f"[ERROR] XML文件不存在: {xml_path}")
        return
    
    # 初始化控制器
    print("[INFO] 初始化Panda控制器...")
    controller = PandaController(urdf_path, xml_path)
    
    # 创建测试轨迹
    print("[INFO] 创建测试轨迹...")
    center = np.array([0.5, 0.0, 0.4])  # 保守的可达位置
    radius = 0.05  # 5cm半径
    height = 0.0
    trajectory = TrajectoryGenerator(center, radius, height)
    positions, _ = trajectory.generate_circular_trajectory(200)
    
    # 用户选择控制模式
    print("\n请选择控制模式进行对比:")
    print("1. 直接关节空间控制 (Direct Joint Control)")
    print("2. 修复的任务空间控制 (Fixed Task Space Control)")
    print("3. 两种模式都测试 (Both Methods)")
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    results = {}
    
    if choice in ["1", "3"]:
        # 测试方法1: 直接关节空间控制
        print("\n" + "="*60)
        print("测试方法1: 直接关节空间控制")
        print("="*60)
        
        controller.set_control_mode("joint")
        controller.run_trajectory(positions, duration=3.0, dt=0.01)
        controller.print_performance_summary()
        
        # 保存结果
        results["joint"] = {
            'errors': np.array(controller.performance_monitor['position_errors']),
            'outputs': np.array(controller.performance_monitor['control_outputs']),
            'times': np.array(controller.performance_monitor['computation_times'])
        }
    
    if choice in ["2", "3"]:
        # 测试方法2: 修复的任务空间控制
        print("\n" + "="*60)
        print("测试方法2: 修复的任务空间控制")
        print("="*60)
        
        controller.set_control_mode("task")
        controller.run_trajectory(positions, duration=3.0, dt=0.01)
        controller.print_performance_summary()
        
        # 保存结果
        results["task"] = {
            'errors': np.array(controller.performance_monitor['position_errors']),
            'outputs': np.array(controller.performance_monitor['control_outputs']),
            'times': np.array(controller.performance_monitor['computation_times'])
        }
    
    # 对比结果
    if len(results) == 2:
        print("\n" + "="*60)
        print("对比分析")
        print("="*60)
        
        joint_avg_error = np.mean(results["joint"]['errors']) * 1000
        task_avg_error = np.mean(results["task"]['errors']) * 1000
        
        print(f"直接关节控制平均误差: {joint_avg_error:.2f} mm")
        print(f"任务空间控制平均误差: {task_avg_error:.2f} mm")
        
        if joint_avg_error < task_avg_error:
            improvement = (task_avg_error - joint_avg_error) / task_avg_error * 100
            print(f"[RESULT] 直接关节控制更优，误差减少 {improvement:.1f}%")
        else:
            improvement = (joint_avg_error - task_avg_error) / joint_avg_error * 100
            print(f"[RESULT] 任务空间控制更优，误差减少 {improvement:.1f}%")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)

if __name__ == "__main__":
    main()
