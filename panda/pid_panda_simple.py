#!/usr/bin/env python3
"""
Franka Panda机械臂PID控制圆弧轨迹跟踪系统（简化版）
使用ikpy进行逆解，MuJoCo进行渲染和可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mujoco
import mujoco.viewer

# 重新添加ikpy依赖
import ikpy.chain
from typing import List, Tuple, Optional
import time
import os

# 应用简单的ikpy修复
try:
    from simple_ikpy_fix import simple_ikpy_fix
    simple_ikpy_fix()
except ImportError:
    print("⚠️ simple_ikpy_fix not found, trying direct fix")
    import numpy as np
    if not hasattr(np, 'float'):
        np.float = float

class TrajectoryGenerator:
    """轨迹生成器类"""
    
    def __init__(self, center: np.ndarray, radius: float, height: float, 
                 start_angle: float = 0, end_angle: float = 2*np.pi):
        """
        初始化轨迹生成器
        
        Args:
            center: 圆弧中心点 [x, y, z]
            radius: 圆弧半径
            height: 圆弧高度
            start_angle: 起始角度
            end_angle: 结束角度
        """
        self.center = np.array(center)
        self.radius = radius
        self.height = height
        self.start_angle = start_angle
        self.end_angle = end_angle
    
    def generate_circular_trajectory(self, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成圆弧轨迹
        
        Args:
            num_points: 轨迹点数量
            
        Returns:
            positions: 位置轨迹 [num_points, 3]
            velocities: 速度轨迹 [num_points, 3]
        """
        angles = np.linspace(self.start_angle, self.end_angle, num_points)
        
        # 生成圆弧轨迹
        x = self.center[0] + self.radius * np.cos(angles)
        y = self.center[1] + self.radius * np.sin(angles)
        z = self.center[2] + self.height * np.ones_like(angles)
        
        positions = np.column_stack([x, y, z])
        
        # 计算速度（角速度 * 半径）
        angular_velocity = (self.end_angle - self.start_angle) / (num_points - 1)
        vx = -self.radius * angular_velocity * np.sin(angles)
        vy = self.radius * angular_velocity * np.cos(angles)
        vz = np.zeros_like(angles)
        
        velocities = np.column_stack([vx, vy, vz])
        
        return positions, velocities
    
    def generate_linear_trajectory(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                  num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成直线轨迹
        
        Args:
            start_pos: 起始位置
            end_pos: 结束位置
            num_points: 轨迹点数量
            
        Returns:
            positions: 位置轨迹
            velocities: 速度轨迹
        """
        positions = np.linspace(start_pos, end_pos, num_points)
        
        # 计算速度
        total_distance = np.linalg.norm(end_pos - start_pos)
        velocity_magnitude = total_distance / (num_points - 1)
        direction = (end_pos - start_pos) / total_distance
        velocities = np.tile(direction * velocity_magnitude, (num_points, 1))
        
        return positions, velocities

class PIDController:
    """PID控制器类"""
    
    def __init__(self, kp: float = 100.0, ki: float = 10.0, kd: float = 20.0, 
                 integral_limit: float = 100.0, output_limit: float = 1000.0):
        """
        初始化PID控制器
        
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
        
        # 状态变量
        self.prev_error = None
        self.integral = np.zeros(3)
        self.last_time = None
    
    def reset(self):
        """重置控制器状态"""
        self.prev_error = None
        self.integral = np.zeros(3)
        self.last_time = None
    
    def compute(self, target: np.ndarray, current: np.ndarray, dt: float) -> np.ndarray:
        """
        计算PID控制输出
        
        Args:
            target: 目标位置
            current: 当前位置
            dt: 时间步长
            
        Returns:
            control_output: 控制输出
        """
        error = target - current
        
        # 比例项
        proportional = self.kp * error
        
        # 积分项
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        integral = self.ki * self.integral
        
        # 微分项
        derivative = np.zeros(3)
        if self.prev_error is not None:
            derivative = self.kd * (error - self.prev_error) / dt
        
        # 总输出
        output = proportional + integral + derivative
        output = np.clip(output, -self.output_limit, self.output_limit)
        
        self.prev_error = error.copy()
        
        return output

class PandaController:
    """Panda机械臂控制器类"""
    
    def __init__(self, urdf_path: str, xml_path: str):
        """
        初始化Panda控制器
        
        Args:
            urdf_path: URDF文件路径
            xml_path: MuJoCo XML文件路径
        """
        self.urdf_path = urdf_path
        self.xml_path = xml_path
        
        # 初始化ikpy链（使用修复后的本地URDF文件）
        try:
            # 使用原始的URDF文件（已经修复了参数）
            fixed_urdf_path = urdf_path
            
            self.chain = ikpy.chain.Chain.from_urdf_file(
                fixed_urdf_path,
                base_elements=["panda_link0"],  # 只设置基座
                active_links_mask=[False] + [True] * 7 + [False] * 3,  # 激活7个关节，忽略手指
                name="panda_arm"
            )
            print("✅ ikpy chain initialized successfully (using fixed local URDF)")
            print(f"   Chain has {len(self.chain.links)} links")
            print(f"   Active joints: {sum(self.chain.active_links_mask)}")
        except Exception as e:
            print(f"❌ Fixed local URDF failed: {e}")
            print("   Trying online URDF...")
            try:
                # 备用方案：使用在线URDF文件
                online_urdf_url = "https://raw.githubusercontent.com/Phylliade/ikpy/master/src/ikpy/data/panda.urdf"
                self.chain = ikpy.chain.Chain.from_urdf_file(
                    online_urdf_url,
                    base_elements=["panda_link0"],
                    active_links_mask=[False] + [True] * 7 + [False] * 3,
                    name="panda_arm"
                )
                print("✅ ikpy chain initialized successfully (using online URDF)")
            except Exception as e2:
                print(f"❌ Online URDF also failed: {e2}")
                self.chain = None
        
        # 初始化MuJoCo
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
            self.data = mujoco.MjData(self.model)
            print("✅ MuJoCo model loaded successfully")
        except Exception as e:
            print(f"❌ MuJoCo model loading failed: {e}")
            self.model = None
            self.data = None
        
        # 关节数量
        if self.chain:
            # 使用活动关节数量
            self.num_joints = sum(self.chain.active_links_mask)
            print(f"✅ Found {self.num_joints} active joints")
        else:
            self.num_joints = 7  # 默认Panda关节数
            print(f"⚠️ Using default {self.num_joints} joints")
        
        # 初始化控制器
        self.pid_controller = PIDController()
        
        # 轨迹生成器
        self.trajectory_generator = None
        
        # 错误计数器和限制
        self.ik_error_count = 0
        self.max_ik_errors = 3  # 最多显示3次相同错误
        self.last_ik_error = ""
        
        # 数据记录
        self.target_positions = []
        self.actual_positions = []
        self.joint_positions = []
        self.control_errors = []
        self.timestamps = []
        
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
                self.ikpy_to_mujoco.append(-1)  # 占位，后续可报错
        
        # C. 检查关节名映射
        print("关节名映射:", self.ikpy_to_mujoco)
    
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
            end_effector_id = self.model.body("link7").id
            return self.data.xpos[end_effector_id].copy()
        except:
            # 如果找不到link7，尝试其他可能的末端执行器名称
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
    
    def set_joint_positions(self, positions: np.ndarray):
        """设置关节目标位置（通过ctrl控制）"""
        if self.data is None:
            return
        # 关节名映射
        if hasattr(self, 'ikpy_to_mujoco') and len(self.ikpy_to_mujoco) == self.num_joints:
            ordered_positions = np.array([positions[i] for i in self.ikpy_to_mujoco])
        else:
            ordered_positions = positions
        # 使用ctrl控制关节，而不是直接设置qpos
        self.data.ctrl[:self.num_joints] = ordered_positions
        mujoco.mj_forward(self.model, self.data)
    
    def set_joint_positions_direct(self, positions: np.ndarray):
        """直接设置关节位置（用于初始化）"""
        if self.data is None:
            return
        # 关节名映射
        if hasattr(self, 'ikpy_to_mujoco') and len(self.ikpy_to_mujoco) == self.num_joints:
            ordered_positions = np.array([positions[i] for i in self.ikpy_to_mujoco])
        else:
            ordered_positions = positions
        # 直接设置qpos（用于初始化）
        self.data.qpos[:self.num_joints] = ordered_positions
        mujoco.mj_forward(self.model, self.data)
    
    def inverse_kinematics(self, target_position: np.ndarray,
                          initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        逆运动学求解 - 使用ikpy库
        
        Args:
            target_position: 目标位置 [x, y, z]
            initial_guess: 初始猜测
            
        Returns:
            joint_positions: 关节位置
        """
        if self.chain is None:
            if self.ik_error_count < self.max_ik_errors:
                print("Warning: ikpy chain not initialized, returning default joint positions")
                self.ik_error_count += 1
            return np.zeros(self.num_joints)
        
        if initial_guess is None:
            initial_guess = self.get_joint_positions()
        
        # 检查目标位置是否在合理范围内
        if np.any(np.abs(target_position) > 1.0):  # 限制在1米范围内
            if self.ik_error_count < self.max_ik_errors:
                print(f"Warning: Target position {target_position} is out of reasonable range")
                self.ik_error_count += 1
            return initial_guess
        
        try:
            # 设置目标方向（保持当前方向）
            target_orientation = [
                [1, 0, 0],  # X轴方向
                [0, 0, 1],  # Y轴方向
                [0, -1, 0]  # Z轴方向
            ]
            
            # 准备初始关节角度（使用ikpy的格式）
            if len(initial_guess) != len(self.chain.links):
                # 调整初始猜测长度
                full_joints = self.chain.active_to_full(initial_guess, [0] * len(self.chain.links))
            else:
                full_joints = initial_guess.copy()
            
            # 创建目标变换矩阵
            target_matrix = np.eye(4)
            target_matrix[:3, 3] = target_position
            target_matrix[:3, :3] = target_orientation
            
            # 使用ikpy求解逆运动学
            import warnings
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
            
            # 检查关节角度是否在合理范围内
            if np.any(np.abs(active_joint_angles) > np.pi):  # 限制在±π范围内
                if self.ik_error_count < self.max_ik_errors:
                    print(f"Warning: Joint angles out of range for target {target_position}")
                    self.ik_error_count += 1
                return initial_guess
            
            return active_joint_angles
            
        except Exception as e:
            error_msg = str(e)
            if error_msg != self.last_ik_error:  # 只显示新的错误类型
                if self.ik_error_count < self.max_ik_errors:
                    print(f"Inverse kinematics failed: {error_msg}")
                    self.ik_error_count += 1
                self.last_ik_error = error_msg
            return initial_guess
            
        except Exception as e:
            error_msg = str(e)
            if error_msg != self.last_ik_error:  # 只显示新的错误类型
                if self.ik_error_count < self.max_ik_errors:
                    print(f"Inverse kinematics failed: {error_msg}")
                    self.ik_error_count += 1
                self.last_ik_error = error_msg
            return initial_guess
    
    def step(self, target_position: np.ndarray, dt: float) -> np.ndarray:
        """
        执行一个控制步
        
        Args:
            target_position: 目标位置
            dt: 时间步长
            
        Returns:
            control_output: 控制输出
        """
        # 获取当前位置
        current_position = self.get_end_effector_position()
        
        # PID控制
        control_output = self.pid_controller.compute(target_position, current_position, dt)
        
        # 记录数据
        self.target_positions.append(target_position.copy())
        self.actual_positions.append(current_position.copy())
        self.joint_positions.append(self.get_joint_positions().copy())
        self.control_errors.append(target_position - current_position)
        
        return control_output
    
    def run_trajectory(self, duration: float = 1.0, dt: float = 0.01) -> None:
        """
        运行轨迹跟踪
        
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
        
        # 重置控制器
        self.pid_controller.reset()
        
        # 清空数据记录
        self.target_positions = []
        self.actual_positions = []
        self.joint_positions = []
        self.control_errors = []
        self.timestamps = []
        
        print("开始轨迹跟踪...")
        start_time = time.time()
        
        # 重置错误计数器
        self.ik_error_count = 0
        self.last_ik_error = ""
        
        for i, target_pos in enumerate(positions):
            current_time = i * dt
            self.timestamps.append(current_time)
            
            # 获取当前位置
            current_position = self.get_end_effector_position()
            
            # PID控制计算期望位置
            control_output = self.pid_controller.compute(target_pos, current_position, dt)
            
            # 记录数据
            self.target_positions.append(target_pos.copy())
            self.actual_positions.append(current_position.copy())
            self.joint_positions.append(self.get_joint_positions().copy())
            self.control_errors.append(target_pos - current_position)
            
            # 使用PID控制后的位置做逆运动学
            desired_position = target_pos  # 或者 current_position + control_output * dt
            joint_positions = self.inverse_kinematics(desired_position)
            
            self.set_joint_positions(joint_positions)
            
            # 更新MuJoCo物理仿真
            mujoco.mj_step(self.model, self.data)
            
            # 显示进度（减少频率）
            if i % 200 == 0 or i == len(positions) - 1:
                progress = (i + 1) / len(positions) * 100
                print(f"Progress: {progress:.1f}% ({i+1}/{len(positions)})")
        
        end_time = time.time()
        print(f"Trajectory tracking completed, time: {end_time - start_time:.2f}s")
    
    def visualize_trajectory(self):
        """Visualize trajectory tracking results"""
        if not self.target_positions:
            print("No trajectory data to visualize")
            return
        
        target_positions = np.array(self.target_positions)
        actual_positions = np.array(self.actual_positions)
        control_errors = np.array(self.control_errors)
        
        # 创建图形
        fig = plt.figure(figsize=(15, 10))
        
        # 3D轨迹图
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
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
        ax2 = fig.add_subplot(2, 3, 2)
        error_magnitude = np.linalg.norm(control_errors, axis=1)
        ax2.plot(self.timestamps, error_magnitude, 'r-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error')
        ax2.grid(True)
        
        # X方向跟踪
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(self.timestamps, target_positions[:, 0], 'b-', label='Target', linewidth=2)
        ax3.plot(self.timestamps, actual_positions[:, 0], 'r--', label='Actual', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('X Position (m)')
        ax3.set_title('X Direction Tracking')
        ax3.legend()
        ax3.grid(True)
        
        # Y方向跟踪
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(self.timestamps, target_positions[:, 1], 'b-', label='Target', linewidth=2)
        ax4.plot(self.timestamps, actual_positions[:, 1], 'r--', label='Actual', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Y Position (m)')
        ax4.set_title('Y Direction Tracking')
        ax4.legend()
        ax4.grid(True)
        
        # Z方向跟踪
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(self.timestamps, target_positions[:, 2], 'b-', label='Target', linewidth=2)
        ax5.plot(self.timestamps, actual_positions[:, 2], 'r--', label='Actual', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Z Position (m)')
        ax5.set_title('Z Direction Tracking')
        ax5.legend()
        ax5.grid(True)
        
        # 关节位置图
        ax6 = fig.add_subplot(2, 3, 6)
        joint_positions = np.array(self.joint_positions)
        if joint_positions.size > 0:  # 确保有数据
            for i in range(min(joint_positions.shape[1], 7)):  # 最多显示7个关节
                ax6.plot(self.timestamps, joint_positions[:, i], 
                        label=f'Joint {i+1}', linewidth=1)
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Joint Angle (rad)')
            ax6.set_title('Joint Positions')
            ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print("\nTrajectory Tracking Statistics:")
        print(f"Average Position Error: {np.mean(error_magnitude):.6f} m")
        print(f"Maximum Position Error: {np.max(error_magnitude):.6f} m")
        print(f"Position Error Std Dev: {np.std(error_magnitude):.6f} m")
    
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
        self.pid_controller.reset()
        
        # 重置错误计数器
        self.ik_error_count = 0
        self.last_ik_error = ""
        
        # 创建MuJoCo查看器
        try:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            print("Starting MuJoCo simulation...")
            print("Press Ctrl+C to exit simulation")
            
            # 设置初始关节位置（使用第一个目标位置的逆运动学解）
            if len(positions) > 0:
                initial_joints = self.inverse_kinematics(positions[0])
                self.set_joint_positions_direct(initial_joints)
            else:
                initial_joints = np.zeros(self.num_joints)
                self.set_joint_positions_direct(initial_joints)
            
            try:
                for i, target_pos in enumerate(positions):
                    # 获取当前位置
                    current_position = self.get_end_effector_position()
                    
                    # PID控制计算期望位置
                    control_output = self.pid_controller.compute(target_pos, current_position, dt)
                    
                    # 使用PID控制后的位置做逆运动学
                    desired_position = target_pos  # 或者 current_position + control_output * dt
                    joint_positions = self.inverse_kinematics(desired_position)
                    self.set_joint_positions(joint_positions)
                    
                    # 调试信息（每10步打印一次）
                    if i % 10 == 0:
                        print(f"Step {i}: Target={target_pos}, Current={current_position}, Joints={joint_positions[:3]}...")
                    
                    # 更新MuJoCo
                    mujoco.mj_step(self.model, self.data)
                    
                    # 强制刷新viewer（如果支持）
                    try:
                        if hasattr(viewer, 'sync'):
                            viewer.sync()
                        elif hasattr(viewer, 'render'):
                            viewer.render()
                    except:
                        pass  # 如果不支持，继续执行
                    
                    # 控制仿真速度
                    time.sleep(dt)
                    
            except KeyboardInterrupt:
                print("Simulation interrupted by user")
            finally:
                print("Simulation ended")
                
        except Exception as e:
            print(f"❌ MuJoCo viewer failed to start: {e}")

def main():
    """主函数"""
    # 文件路径
    urdf_path = "./models/franka_emika_panda/frankaEmikaPanda.urdf"
    xml_path = "./models/franka_emika_panda/scene.xml"  # 使用标准版本，适合MuJoCo 3.3.3
    
    # 检查文件是否存在
    if not os.path.exists(urdf_path):
        print(f"错误: URDF文件不存在: {urdf_path}")
        return
    
    if not os.path.exists(xml_path):
        print(f"错误: XML文件不存在: {xml_path}")
        return
    
    # 创建Panda控制器
    print("初始化Panda控制器...")
    controller = PandaController(urdf_path, xml_path)
    
    # 创建圆弧轨迹（使用更合理的工作空间参数）
    print("创建圆弧轨迹...")
    center = np.array([0.5, 0.0, 0.4])  # 圆弧中心，更靠近机械臂基座
    radius = 0.15  # 圆弧半径，增大到15cm
    height = 0.0  # 高度偏移
    trajectory = TrajectoryGenerator(center, radius, height)
    
    # 设置轨迹
    controller.set_trajectory(trajectory)
    
    # 运行轨迹跟踪
    print("运行轨迹跟踪...")
    controller.run_trajectory(duration=1.0, dt=0.01)
    
    # 可视化结果
    print("生成可视化结果...")
    controller.visualize_trajectory()
    
    # 询问是否运行MuJoCo仿真
    response = input("\n是否运行MuJoCo仿真? (y/n): ")
    if response.lower() == 'y':
        controller.run_mujoco_simulation(duration=1.0, dt=0.01)

if __name__ == "__main__":
    main() 