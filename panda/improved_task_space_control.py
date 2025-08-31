#!/usr/bin/env python3
"""
改进的任务空间力矩控制系统
解决任务空间力矩控制的关键问题，提高控制效果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, List
import time
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# MuJoCo相关导入
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️ MuJoCo未安装，MuJoCo渲染功能不可用")

class ImprovedTaskSpaceController:
    """改进的任务空间控制器"""
    
    def __init__(self, 
                 kp: float = 100.0, 
                 ki: float = 10.0, 
                 kd: float = 20.0,
                 impedance_k: float = 1000.0,
                 impedance_d: float = 100.0,
                 adaptive_gains: bool = True):
        """
        初始化改进的任务空间控制器
        
        Args:
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            impedance_k: 阻抗刚度
            impedance_d: 阻抗阻尼
            adaptive_gains: 是否启用自适应增益
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.impedance_k = impedance_k
        self.impedance_d = impedance_d
        self.adaptive_gains = adaptive_gains
        
        # 控制器状态
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = 0.0
        
        # 自适应参数
        self.base_kp = kp
        self.base_ki = ki
        self.base_kd = kd
        
        # 性能监控
        self.performance_history = {
            'errors': [],
            'forces': [],
            'velocities': [],
            'gains': []
        }
        
        # 重力补偿
        self.gravity_compensation = True
        self.gravity_vector = np.array([0, 0, -9.81])  # 重力向量
        
        # 摩擦补偿
        self.friction_compensation = True
        self.friction_coefficients = np.array([0.1, 0.1, 0.1])  # 摩擦系数
        
        print(f"[OK] 改进的任务空间控制器初始化完成")
        print(f"[INFO] PID参数: kp={kp}, ki={ki}, kd={kd}")
        print(f"[INFO] 阻抗参数: k={impedance_k}, d={impedance_d}")
        print(f"[INFO] 重力补偿: {'启用' if self.gravity_compensation else '禁用'}")
        print(f"[INFO] 摩擦补偿: {'启用' if self.friction_compensation else '禁用'}")
    
    def reset(self):
        """重置控制器状态"""
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.prev_time = 0.0
        self.performance_history = {
            'errors': [],
            'forces': [],
            'velocities': [],
            'gains': []
        }
    
    def adapt_gains(self, error: np.ndarray, error_rate: np.ndarray, velocity: np.ndarray):
        """自适应增益调节"""
        if not self.adaptive_gains:
            return
        
        error_magnitude = np.linalg.norm(error)
        velocity_magnitude = np.linalg.norm(velocity)
        
        # 根据误差大小调整增益
        if error_magnitude > 0.1:  # 大误差
            kp_factor = 1.5  # 提高比例增益
            ki_factor = 0.8  # 降低积分增益
            kd_factor = 0.7  # 降低微分增益
        elif error_magnitude > 0.01:  # 中误差
            kp_factor = 1.2
            ki_factor = 1.0
            kd_factor = 1.0
        else:  # 小误差
            kp_factor = 0.9
            ki_factor = 1.2
            kd_factor = 1.5
        
        # 根据速度调整增益
        if velocity_magnitude > 0.5:  # 高速运动
            kd_factor *= 1.3  # 增加阻尼
        
        self.kp = self.base_kp * kp_factor
        self.ki = self.base_ki * ki_factor
        self.kd = self.base_kd * kd_factor
    
    def compute_gravity_compensation(self, position: np.ndarray, mass: float = 1.0) -> np.ndarray:
        """计算重力补偿力"""
        if not self.gravity_compensation:
            return np.zeros(3)
        
        # 简化的重力补偿模型
        gravity_force = mass * self.gravity_vector
        return gravity_force
    
    def compute_friction_compensation(self, velocity: np.ndarray) -> np.ndarray:
        """计算摩擦补偿力"""
        if not self.friction_compensation:
            return np.zeros(3)
        
        # 库仑摩擦模型
        friction_force = -self.friction_coefficients * np.sign(velocity) * np.abs(velocity)
        return friction_force
    
    def compute_impedance_control(self, 
                                target_position: np.ndarray, 
                                current_position: np.ndarray,
                                target_velocity: np.ndarray,
                                current_velocity: np.ndarray) -> np.ndarray:
        """计算阻抗控制力"""
        # 位置误差
        position_error = target_position - current_position
        
        # 速度误差
        velocity_error = target_velocity - current_velocity
        
        # 阻抗控制力
        impedance_force = (self.impedance_k * position_error + 
                          self.impedance_d * velocity_error)
        
        return impedance_force
    
    def compute_control_force(self, 
                            target_position: np.ndarray,
                            current_position: np.ndarray,
                            target_velocity: np.ndarray,
                            current_velocity: np.ndarray,
                            dt: float) -> np.ndarray:
        """
        计算控制力
        
        Args:
            target_position: 目标位置 [x, y, z]
            current_position: 当前位置 [x, y, z]
            target_velocity: 目标速度 [vx, vy, vz]
            current_velocity: 当前速度 [vx, vy, vz]
            dt: 时间步长
            
        Returns:
            control_force: 控制力 [fx, fy, fz]
        """
        # 计算误差
        position_error = target_position - current_position
        velocity_error = target_velocity - current_velocity
        
        # 自适应增益调节
        self.adapt_gains(position_error, velocity_error, current_velocity)
        
        # PID控制力
        p_force = self.kp * position_error
        
        # 积分项（带抗饱和）
        if np.linalg.norm(p_force) < 100.0:  # 未饱和时积分
            self.integral += position_error * dt
            self.integral = np.clip(self.integral, -1.0, 1.0)
        i_force = self.ki * self.integral
        
        # 微分项
        if dt > 0:
            derivative = (position_error - self.prev_error) / dt
        else:
            derivative = np.zeros(3)
        d_force = self.kd * derivative
        
        # PID控制力
        pid_force = p_force + i_force + d_force
        
        # 阻抗控制力
        impedance_force = self.compute_impedance_control(
            target_position, current_position, target_velocity, current_velocity
        )
        
        # 重力补偿
        gravity_force = self.compute_gravity_compensation(current_position)
        
        # 摩擦补偿
        friction_force = self.compute_friction_compensation(current_velocity)
        
        # 总控制力
        total_force = pid_force + impedance_force + gravity_force + friction_force
        
        # 力限制
        max_force = 100.0  # 最大控制力
        force_magnitude = np.linalg.norm(total_force)
        if force_magnitude > max_force:
            total_force = total_force * (max_force / force_magnitude)
        
        # 更新状态
        self.prev_error = position_error.copy()
        self.prev_time += dt
        
        # 记录性能数据
        self.performance_history['errors'].append(np.linalg.norm(position_error))
        self.performance_history['forces'].append(np.linalg.norm(total_force))
        self.performance_history['velocities'].append(np.linalg.norm(current_velocity))
        self.performance_history['gains'].append([self.kp, self.ki, self.kd])
        
        return total_force

class TaskSpaceControlSimulator:
    """任务空间控制仿真器"""
    
    def __init__(self, controller: ImprovedTaskSpaceController):
        self.controller = controller
        self.mass = 1.0  # 质量
        self.damping = 0.1  # 阻尼系数
        
        # 状态变量
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        # 记录数据
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.force_history = []
        self.target_history = []
        self.error_history = []
    
    def reset(self):
        """重置仿真器"""
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.acceleration = np.array([0.0, 0.0, 0.0])
        
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.force_history = []
        self.target_history = []
        self.error_history = []
        
        self.controller.reset()
    
    def step(self, target_position: np.ndarray, target_velocity: np.ndarray, dt: float):
        """执行一个仿真步"""
        # 计算控制力
        control_force = self.controller.compute_control_force(
            target_position, self.position, target_velocity, self.velocity, dt
        )
        
        # 动力学仿真 (简化的二阶系统)
        # F = ma + bv
        self.acceleration = (control_force - self.damping * self.velocity) / self.mass
        
        # 积分更新状态
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        
        # 记录数据
        self.time_history.append(len(self.time_history) * dt)
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.force_history.append(control_force.copy())
        self.target_history.append(target_position.copy())
        self.error_history.append(np.linalg.norm(target_position - self.position))
    
    def run_trajectory(self, target_trajectory: np.ndarray, target_velocities: np.ndarray, dt: float):
        """运行轨迹跟踪"""
        self.reset()
        
        for i, (target_pos, target_vel) in enumerate(zip(target_trajectory, target_velocities)):
            self.step(target_pos, target_vel, dt)
            
            if i % 100 == 0:
                error = np.linalg.norm(target_pos - self.position)
                print(f"Step {i}: Error = {error:.4f}m, Position = [{self.position[0]:.3f}, {self.position[1]:.3f}, {self.position[2]:.3f}]")
    
    def plot_results(self):
        """绘制结果"""
        if not self.time_history:
            print("❌ 没有数据可绘制")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 轨迹跟踪
        target_pos = np.array(self.target_history)
        actual_pos = np.array(self.position_history)
        
        ax1.plot(target_pos[:, 0], target_pos[:, 1], 'b-', linewidth=2, label='目标轨迹')
        ax1.plot(actual_pos[:, 0], actual_pos[:, 1], 'r--', linewidth=2, label='实际轨迹')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('XY平面轨迹跟踪')
        ax1.legend()
        ax1.grid(True)
        
        # 位置误差
        ax2.plot(self.time_history, self.error_history, 'r-', linewidth=2)
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('位置误差 (m)')
        ax2.set_title('位置误差随时间变化')
        ax2.grid(True)
        
        # 控制力
        forces = np.array(self.force_history)
        ax3.plot(self.time_history, forces[:, 0], 'r-', label='Fx')
        ax3.plot(self.time_history, forces[:, 1], 'g-', label='Fy')
        ax3.plot(self.time_history, forces[:, 2], 'b-', label='Fz')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('控制力 (N)')
        ax3.set_title('控制力随时间变化')
        ax3.legend()
        ax3.grid(True)
        
        # 自适应增益
        gains = np.array(self.controller.performance_history['gains'])
        if len(gains) > 0:
            ax4.plot(self.time_history, gains[:, 0], 'r-', label='Kp')
            ax4.plot(self.time_history, gains[:, 1], 'g-', label='Ki')
            ax4.plot(self.time_history, gains[:, 2], 'b-', label='Kd')
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('增益值')
            ax4.set_title('自适应增益变化')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.error_history:
            return {}
        
        errors = np.array(self.error_history)
        
        metrics = {
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'final_error': errors[-1],
            'settling_time': self._calculate_settling_time(errors),
            'overshoot': self._calculate_overshoot(errors)
        }
        
        return metrics
    
    def _calculate_settling_time(self, errors: np.ndarray, threshold: float = 0.02) -> float:
        """计算稳定时间"""
        for i, error in enumerate(errors):
            if abs(error) < threshold:
                return i * 0.01  # 假设dt=0.01
        return len(errors) * 0.01
    
    def _calculate_overshoot(self, errors: np.ndarray) -> float:
        """计算超调量"""
        if len(errors) < 10:
            return 0.0
        
        # 找到第一个峰值
        for i in range(1, len(errors)-1):
            if errors[i] > errors[i-1] and errors[i] > errors[i+1]:
                return errors[i]
        return 0.0

def generate_test_trajectory(num_points: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """生成测试轨迹 - 平行于地面的侧方圆弧"""
    t = np.linspace(0, 5, num_points)
    
    # 圆弧参数设计
    radius = 0.15  # 半径15cm，比之前稍大
    center = np.array([0.3, 0.2, 0.4])  # 圆心在机械臂侧方，高度适中
    
    positions = np.zeros((num_points, 3))
    velocities = np.zeros((num_points, 3))
    
    for i, time in enumerate(t):
        # 圆弧角度范围：从-π/2到π/2，形成半圆弧
        angle = -np.pi/2 + np.pi * time / 5.0
        
        # 圆弧在XY平面，平行于地面
        # X轴：从圆心向侧方延伸
        # Y轴：从圆心向前后延伸
        positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
        
        # 计算速度（切线方向）
        velocities[i] = radius * np.pi / 5.0 * np.array([-np.sin(angle), np.cos(angle), 0])
    
    return positions, velocities

def generate_smooth_test_trajectory(num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """生成更平滑的测试轨迹 - 改进版本，增大圆弧便于观察"""
    t = np.linspace(0, 8, num_points)  # 更长时间，更平滑
    
    # 圆弧参数设计 - 增大圆弧，更明显的运动
    radius = 0.25  # 半径25cm，增大便于观察
    center = np.array([0.5, 0.0, 0.3])  # 圆心位置：更靠前，高度适中
    
    positions = np.zeros((num_points, 3))
    velocities = np.zeros((num_points, 3))
    
    for i, time in enumerate(t):
        # 使用平滑的加速/减速曲线
        progress = time / 8.0
        smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))  # 平滑的S曲线
        
        # 圆弧角度范围：从-π到π，形成完整圆弧，运动更明显
        angle = -np.pi + 2*np.pi * smooth_progress
        
        # 圆弧在XY平面，平行于地面
        positions[i] = center + radius * np.array([np.cos(angle), np.sin(angle), 0])
        
        # 计算速度（切线方向）- 考虑平滑曲线
        angle_velocity = 2*np.pi * np.pi/8.0 * np.sin(np.pi * progress)
        velocities[i] = radius * angle_velocity * np.array([-np.sin(angle), np.cos(angle), 0])
    
    return positions, velocities

def compare_control_methods():
    """对比不同控制方法"""
    print("🧪 对比不同控制方法")
    
    # 生成测试轨迹
    target_positions, target_velocities = generate_test_trajectory()
    
    # 测试不同配置
    configs = [
        {
            'name': '基础PID',
            'kp': 100.0, 'ki': 10.0, 'kd': 20.0,
            'impedance_k': 0.0, 'impedance_d': 0.0,
            'adaptive_gains': False,
            'gravity_compensation': False,
            'friction_compensation': False
        },
        {
            'name': '改进PID',
            'kp': 100.0, 'ki': 10.0, 'kd': 20.0,
            'impedance_k': 0.0, 'impedance_d': 0.0,
            'adaptive_gains': True,
            'gravity_compensation': False,
            'friction_compensation': False
        },
        {
            'name': 'PID+阻抗',
            'kp': 100.0, 'ki': 10.0, 'kd': 20.0,
            'impedance_k': 1000.0, 'impedance_d': 100.0,
            'adaptive_gains': True,
            'gravity_compensation': False,
            'friction_compensation': False
        },
        {
            'name': '完整控制',
            'kp': 100.0, 'ki': 10.0, 'kd': 20.0,
            'impedance_k': 1000.0, 'impedance_d': 100.0,
            'adaptive_gains': True,
            'gravity_compensation': True,
            'friction_compensation': True
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n📊 测试配置: {config['name']}")
        
        # 创建控制器
        controller = ImprovedTaskSpaceController(
            kp=config['kp'], ki=config['ki'], kd=config['kd'],
            impedance_k=config['impedance_k'], impedance_d=config['impedance_d'],
            adaptive_gains=config['adaptive_gains']
        )
        
        # 设置补偿选项
        controller.gravity_compensation = config['gravity_compensation']
        controller.friction_compensation = config['friction_compensation']
        
        # 创建仿真器
        simulator = TaskSpaceControlSimulator(controller)
        
        # 运行仿真
        simulator.run_trajectory(target_positions, target_velocities, dt=0.01)
        
        # 计算性能指标
        metrics = simulator.calculate_performance_metrics()
        results[config['name']] = metrics
        
        print(f"   RMSE: {metrics['rmse']:.4f}m")
        print(f"   最大误差: {metrics['max_error']:.4f}m")
        print(f"   最终误差: {metrics['final_error']:.4f}m")
        print(f"   稳定时间: {metrics['settling_time']:.2f}s")
    
    # 显示对比结果
    plot_comparison_results(results)
    
    return results

def plot_comparison_results(results: dict):
    """绘制对比结果 - 简化的性能对比"""
    methods = list(results.keys())
    metrics = ['rmse', 'max_error', 'final_error', 'settling_time']
    
    print("\n📊 性能对比结果:")
    print("=" * 80)
    print(f"{'方法':<15} {'RMSE':<10} {'最大误差':<10} {'最终误差':<10} {'稳定时间':<10}")
    print("=" * 80)
    
    for method, metrics_data in results.items():
        print(f"{method:<15} {metrics_data['rmse']:<10.4f} {metrics_data['max_error']:<10.4f} "
              f"{metrics_data['final_error']:<10.4f} {metrics_data['settling_time']:<10.2f}")
    
    print("=" * 80)
    
    # 找出最佳方法
    best_method = min(results.keys(), key=lambda x: results[x]['rmse'])
    print(f"\n🏆 最佳方法: {best_method}")
    print(f"   RMSE: {results[best_method]['rmse']:.4f}m")

def main():
    """主函数"""
    print("🚀 改进的任务空间力矩控制系统")
    
    # 对比不同控制方法
    results = compare_control_methods()
    
    print("\n💡 改进建议:")
    print("1. 启用自适应增益调节")
    print("2. 添加阻抗控制")
    print("3. 实现重力补偿")
    print("4. 添加摩擦补偿")
    print("5. 优化PID参数")

class MuJoCoTaskSpaceSimulator:
    """MuJoCo任务空间控制仿真器"""
    
    def __init__(self, model_path: str = None, control_frequency: int = 1000):
        """
        初始化MuJoCo仿真器 - 改进版本
        
        Args:
            model_path: MuJoCo模型文件路径
            control_frequency: 控制频率 (Hz)，默认1000Hz
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo未安装，无法使用MuJoCo仿真器")
        
        # 设置默认模型路径
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(project_root, "models", "franka_emika_panda", "scene.xml")
        else:
            self.model_path = model_path
        
        self.model = None
        self.data = None
        self.viewer = None
        
        # 改进的控制参数
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        
        # 使用优化的控制器参数
        self.controller = ImprovedTaskSpaceController(
            kp=150.0, ki=15.0, kd=30.0,  # 提高增益
            impedance_k=1500.0, impedance_d=150.0,  # 提高阻抗参数
            adaptive_gains=True
        )
        
        # 仿真参数
        self.dt = 0.01  # 时间步长
        self.simulation_time = 0.0
        
        # 记录数据
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.force_history = []
        self.target_history = []
        self.error_history = []
        
        # 初始化MuJoCo
        self._initialize_mujoco()
        
        print(f"[OK] MuJoCo任务空间控制仿真器初始化完成")
        print(f"[INFO] 模型文件: {self.model_path}")
        print(f"[INFO] 时间步长: {self.dt}s")
        print(f"[INFO] 使用已验证的控制器: PID + 阻抗控制")
    
    def _initialize_mujoco(self):
        """初始化MuJoCo"""
        try:
            # 检查模型文件是否存在
            if not os.path.exists(self.model_path):
                print(f"❌ 模型文件不存在: {self.model_path}")
                return False
            
            # 加载模型
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            
            print(f"[OK] MuJoCo模型加载成功")
            print(f"[INFO] 关节数量: {self.model.nq}")
            print(f"[INFO] 执行器数量: {self.model.nu}")
            
            return True
            
        except Exception as e:
            print(f"❌ MuJoCo初始化失败: {e}")
            return False
    
    def start_viewer(self):
        """启动MuJoCo查看器"""
        if self.model is None:
            print("❌ 模型未加载，无法启动查看器")
            return False
        
        try:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("[OK] MuJoCo查看器启动成功")
            return True
        except Exception as e:
            print(f"❌ 查看器启动失败: {e}")
            return False
    
    def close_viewer(self):
        """关闭查看器"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            print("[OK] MuJoCo查看器已关闭")
    
    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置 - 使用夹爪作为真正的末端"""
        if self.data is None:
            return np.zeros(3)
        
        # 使用MuJoCo的正向运动学获取夹爪末端位置
        try:
            # 尝试获取左夹爪位置（作为末端执行器）
            left_finger_id = self.model.body('left_finger').id
            right_finger_id = self.model.body('right_finger').id
            
            # 计算两个夹爪的中点作为末端执行器位置
            left_pos = self.data.xpos[left_finger_id]
            right_pos = self.data.xpos[right_finger_id]
            position = (left_pos + right_pos) / 2.0
            
        except:
            try:
                # 如果找不到夹爪，使用hand
                end_effector_id = self.model.body('hand').id
                position = self.data.xpos[end_effector_id].copy()
            except:
                # 最后使用最后一个body
                position = self.data.xpos[-1].copy()
        
        return position
    
    def get_end_effector_velocity(self) -> np.ndarray:
        """获取末端执行器速度 - 使用夹爪的Jacobian计算"""
        if self.data is None:
            return np.zeros(3)
        
        # 获取关节速度
        joint_velocities = self.data.qvel[:self.model.nv]
        
        # 计算Jacobian矩阵 (3 x nv)
        jacobian = np.zeros((3, self.model.nv))
        
        try:
            # 使用左夹爪的Jacobian（代表夹爪末端）
            left_finger_id = self.model.body('left_finger').id
            left_finger_pos = self.data.xpos[left_finger_id]
            mujoco.mj_jac(self.model, self.data, jacobian, None, 
                         left_finger_pos, left_finger_id)
        except:
            try:
                # 如果找不到夹爪，使用hand
                end_effector_id = self.model.body('hand').id
                end_effector_pos = self.data.xpos[end_effector_id]
                mujoco.mj_jac(self.model, self.data, jacobian, None, 
                             end_effector_pos, end_effector_id)
            except:
                # 最后使用最后一个body
                end_effector_pos = self.data.xpos[-1]
                mujoco.mj_jac(self.model, self.data, jacobian, None, 
                             end_effector_pos, -1)
        
        # 使用Jacobian计算末端执行器速度
        # v_end = J * q_dot
        velocity = jacobian @ joint_velocities
        
        return velocity
    
    def apply_control_force(self, control_force: np.ndarray):
        """应用控制力到MuJoCo - 使用夹爪的Jacobian矩阵转换"""
        if self.data is None:
            return
        
        # 计算Jacobian矩阵 (3 x nv)
        jacobian = np.zeros((3, self.model.nv))
        
        try:
            # 使用左夹爪的Jacobian（代表夹爪末端）
            left_finger_id = self.model.body('left_finger').id
            left_finger_pos = self.data.xpos[left_finger_id]
            mujoco.mj_jac(self.model, self.data, jacobian, None, 
                         left_finger_pos, left_finger_id)
        except:
            try:
                # 如果找不到夹爪，使用hand
                end_effector_id = self.model.body('hand').id
                end_effector_pos = self.data.xpos[end_effector_id]
                mujoco.mj_jac(self.model, self.data, jacobian, None, 
                             end_effector_pos, end_effector_id)
            except:
                # 最后使用最后一个body
                end_effector_pos = self.data.xpos[-1]
                mujoco.mj_jac(self.model, self.data, jacobian, None, 
                             end_effector_pos, -1)
        
        # 使用Jacobian转置将任务空间力转换为关节空间力矩
        # τ = J^T * F_task
        joint_torques = jacobian.T @ control_force
        
        # 确保力矩在合理范围内
        max_torque = 50.0  # 最大力矩限制
        joint_torques = np.clip(joint_torques, -max_torque, max_torque)
        
        # 只应用前7个关节的力矩（Panda有7个关节）
        # 确保维度匹配
        num_actuators = min(len(joint_torques), self.model.nu)
        self.data.ctrl[:num_actuators] = joint_torques[:num_actuators]
    
    def step(self, target_position: np.ndarray, target_velocity: np.ndarray):
        """执行一个仿真步"""
        if self.model is None or self.data is None:
            return
        
        # 获取当前状态
        current_position = self.get_end_effector_position()
        current_velocity = self.get_end_effector_velocity()
        
        # 使用已验证的控制器计算控制力
        control_force = self.controller.compute_control_force(
            target_position, current_position, target_velocity, current_velocity, self.dt
        )
        
        # 应用控制力
        self.apply_control_force(control_force)
        
        # 执行MuJoCo仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 更新仿真时间
        self.simulation_time += self.dt
        
        # 记录数据
        self.time_history.append(self.simulation_time)
        self.position_history.append(current_position.copy())
        self.velocity_history.append(current_velocity.copy())
        self.force_history.append(control_force.copy())
        self.target_history.append(target_position.copy())
        self.error_history.append(np.linalg.norm(target_position - current_position))
        
        # 更新查看器
        if self.viewer is not None:
            try:
                if hasattr(self.viewer, 'sync'):
                    self.viewer.sync()
                elif hasattr(self.viewer, 'render'):
                    self.viewer.render()
            except:
                pass
    
    def run_trajectory(self, target_trajectory: np.ndarray, target_velocities: np.ndarray, 
                      enable_viewer: bool = True, real_time: bool = True):
        """运行轨迹跟踪 - 改进版本"""
        if self.model is None or self.data is None:
            print("❌ 仿真器未初始化")
            return
        
        print(f"🚀 开始改进的MuJoCo轨迹跟踪仿真 (控制频率: {self.control_frequency}Hz)")
        
        # 启动查看器
        if enable_viewer:
            self.start_viewer()
        
        # 重置状态
        self.reset()
        
        # 插值到更高频率
        num_points = len(target_trajectory)
        high_freq_points = int(num_points * self.control_frequency / 100)  # 假设原始频率100Hz
        
        # 创建高频率轨迹
        t_original = np.linspace(0, 1, num_points)
        t_high_freq = np.linspace(0, 1, high_freq_points)
        
        high_freq_trajectory = np.zeros((high_freq_points, 3))
        high_freq_velocities = np.zeros((high_freq_points, 3))
        
        for i in range(3):
            high_freq_trajectory[:, i] = np.interp(t_high_freq, t_original, target_trajectory[:, i])
            high_freq_velocities[:, i] = np.interp(t_high_freq, t_original, target_velocities[:, i])
        
        try:
            for i, (target_pos, target_vel) in enumerate(zip(high_freq_trajectory, high_freq_velocities)):
                # 执行仿真步
                self.step(target_pos, target_vel)
                
                # 实时显示进度
                if i % (self.control_frequency // 10) == 0:  # 每0.1秒显示一次
                    error = np.linalg.norm(target_pos - self.get_end_effector_position())
                    print(f"Step {i}: Error = {error:.4f}m, Position = {self.get_end_effector_position()}")
                
                # 实时控制
                if real_time:
                    time.sleep(self.dt)
        
        except KeyboardInterrupt:
            print("\n用户中断仿真")
        
        finally:
            # 关闭查看器
            if enable_viewer:
                self.close_viewer()
        
        print("✅ 改进的MuJoCo轨迹跟踪仿真完成")
    
    def reset(self):
        """重置仿真器"""
        if self.data is not None:
            mujoco.mj_resetData(self.model, self.data)
        
        self.simulation_time = 0.0
        self.time_history = []
        self.position_history = []
        self.velocity_history = []
        self.force_history = []
        self.target_history = []
        self.error_history = []
        
        self.controller.reset()
    
    def calculate_performance_metrics(self):
        """计算性能指标"""
        if not self.error_history:
            return {}
        
        errors = np.array(self.error_history)
        
        metrics = {
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'final_error': errors[-1],
            'settling_time': self._calculate_settling_time(errors),
            'overshoot': self._calculate_overshoot(errors)
        }
        
        return metrics
    
    def _calculate_settling_time(self, errors: np.ndarray, threshold: float = 0.02) -> float:
        """计算稳定时间"""
        for i, error in enumerate(errors):
            if abs(error) < threshold:
                return i * self.dt
        return len(errors) * self.dt
    
    def _calculate_overshoot(self, errors: np.ndarray) -> float:
        """计算超调量"""
        if len(errors) < 10:
            return 0.0
        
        # 找到第一个峰值
        for i in range(1, len(errors)-1):
            if errors[i] > errors[i-1] and errors[i] > errors[i+1]:
                return errors[i]
        return 0.0
    
    def plot_mujoco_results(self, target_trajectory: np.ndarray, target_velocities: np.ndarray):
        """绘制MuJoCo仿真结果 - 专门用于调参分析"""
        if not self.time_history:
            print("❌ 没有数据可绘制")
            return
        
        fig = plt.figure(figsize=(20, 16))
        
        # 创建子图布局
        ax1 = plt.subplot(2, 3, 1)  # XY平面轨迹
        ax2 = plt.subplot(2, 3, 2)  # 位置误差
        ax3 = plt.subplot(2, 3, 3)  # 控制力
        ax4 = plt.subplot(2, 3, 4)  # 自适应增益
        ax5 = plt.subplot(2, 3, 5, projection='3d')  # 3D轨迹
        ax6 = plt.subplot(2, 3, 6)  # Z轴高度变化
        
        # 1. 3D轨迹对比图
        target_pos = np.array(self.target_history)
        actual_pos = np.array(self.position_history)
        
        # XY平面轨迹对比（平行于地面的低高度圆弧）
        ax1.plot(target_pos[:, 0], target_pos[:, 1], 'b-', linewidth=3, label='目标夹爪圆弧轨迹', alpha=0.8)
        ax1.plot(actual_pos[:, 0], actual_pos[:, 1], 'r--', linewidth=2, label='MuJoCo夹爪实际轨迹', alpha=0.8)
        ax1.scatter(target_pos[0, 0], target_pos[0, 1], color='green', s=100, marker='o', label='起始点')
        ax1.scatter(target_pos[-1, 0], target_pos[-1, 1], color='red', s=100, marker='s', label='结束点')
        ax1.set_xlabel('X (m) - 侧方方向')
        ax1.set_ylabel('Y (m) - 前后方向')
        ax1.set_title('MuJoCo仿真 - 夹爪末端圆弧轨迹跟踪对比（半径25cm，Z=0.3m）')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # 2. 位置误差随时间变化
        ax2.plot(self.time_history, self.error_history, 'r-', linewidth=2, label='位置误差')
        ax2.axhline(y=0.01, color='g', linestyle='--', alpha=0.7, label='1cm误差线')
        ax2.axhline(y=0.005, color='orange', linestyle='--', alpha=0.7, label='5mm误差线')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('位置误差 (m)')
        ax2.set_title('MuJoCo仿真 - 位置误差随时间变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, max(self.error_history) * 1.1)
        
        # 3. 控制力分析
        forces = np.array(self.force_history)
        force_magnitude = np.linalg.norm(forces, axis=1)
        
        ax3.plot(self.time_history, forces[:, 0], 'r-', label='Fx', alpha=0.8)
        ax3.plot(self.time_history, forces[:, 1], 'g-', label='Fy', alpha=0.8)
        ax3.plot(self.time_history, forces[:, 2], 'b-', label='Fz', alpha=0.8)
        ax3.plot(self.time_history, force_magnitude, 'k-', linewidth=2, label='合力大小', alpha=0.8)
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('控制力 (N)')
        ax3.set_title('MuJoCo仿真 - 控制力分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 自适应增益变化
        gains = np.array(self.controller.performance_history['gains'])
        if len(gains) > 0:
            ax4.plot(self.time_history, gains[:, 0], 'r-', label='Kp (比例增益)', linewidth=2)
            ax4.plot(self.time_history, gains[:, 1], 'g-', label='Ki (积分增益)', linewidth=2)
            ax4.plot(self.time_history, gains[:, 2], 'b-', label='Kd (微分增益)', linewidth=2)
            ax4.set_xlabel('时间 (s)')
            ax4.set_ylabel('增益值')
            ax4.set_title('MuJoCo仿真 - 自适应增益变化')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, '无增益数据', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('MuJoCo仿真 - 自适应增益变化')
        
        # 5. 3D轨迹图
        ax5.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2], 'b-', linewidth=3, label='目标夹爪圆弧轨迹', alpha=0.8)
        ax5.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 'r--', linewidth=2, label='MuJoCo夹爪实际轨迹', alpha=0.8)
        ax5.scatter(target_pos[0, 0], target_pos[0, 1], target_pos[0, 2], color='green', s=100, marker='o', label='起始点')
        ax5.scatter(target_pos[-1, 0], target_pos[-1, 1], target_pos[-1, 2], color='red', s=100, marker='s', label='结束点')
        ax5.set_xlabel('X (m) - 侧方方向')
        ax5.set_ylabel('Y (m) - 前后方向')
        ax5.set_zlabel('Z (m) - 高度方向')
        ax5.set_title('MuJoCo仿真 - 3D夹爪末端圆弧轨迹（半径25cm，Z=0.3m）')
        ax5.legend()
        
        # 6. Z轴高度变化
        ax6.plot(self.time_history, target_pos[:, 2], 'b-', linewidth=2, label='目标高度', alpha=0.8)
        ax6.plot(self.time_history, actual_pos[:, 2], 'r--', linewidth=2, label='实际高度', alpha=0.8)
        ax6.set_xlabel('时间 (s)')
        ax6.set_ylabel('高度 (m)')
        ax6.set_title('MuJoCo仿真 - Z轴高度跟踪')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 5. 稳定性分析图
        self._plot_stability_analysis()
    
    def _plot_stability_analysis(self):
        """绘制稳定性分析图"""
        if not self.time_history:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 误差分布直方图
        errors = np.array(self.error_history)
        ax1.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'平均误差: {np.mean(errors):.4f}m')
        ax1.axvline(np.std(errors), color='orange', linestyle='--', linewidth=2, label=f'标准差: {np.std(errors):.4f}m')
        ax1.set_xlabel('位置误差 (m)')
        ax1.set_ylabel('频次')
        ax1.set_title('MuJoCo仿真 - 误差分布分析')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 速度分析
        velocities = np.array(self.velocity_history)
        velocity_magnitude = np.linalg.norm(velocities, axis=1)
        
        ax2.plot(self.time_history, velocity_magnitude, 'purple', linewidth=2, label='速度大小')
        ax2.plot(self.time_history, velocities[:, 0], 'r-', alpha=0.6, label='Vx')
        ax2.plot(self.time_history, velocities[:, 1], 'g-', alpha=0.6, label='Vy')
        ax2.plot(self.time_history, velocities[:, 2], 'b-', alpha=0.6, label='Vz')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('速度 (m/s)')
        ax2.set_title('MuJoCo仿真 - 速度分析')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 速度跟踪分析
        target_vel = np.array(self.target_history)  # 这里需要修正，应该是target_velocities
        actual_vel = np.array(self.velocity_history)
        
        # 计算速度大小
        actual_vel_mag = np.linalg.norm(actual_vel, axis=1)
        
        ax3.plot(self.time_history, actual_vel_mag, 'r-', linewidth=2, label='实际速度大小')
        ax3.set_xlabel('时间 (s)')
        ax3.set_ylabel('速度大小 (m/s)')
        ax3.set_title('MuJoCo仿真 - 速度分析')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 累积误差分析
        cumulative_error = np.cumsum(np.array(self.error_history))
        ax4.plot(self.time_history, cumulative_error, 'darkgreen', linewidth=2, label='累积误差')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('累积误差 (m)')
        ax4.set_title('MuJoCo仿真 - 累积误差分析')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印稳定性指标
        self._print_stability_metrics()
    
    def _print_stability_metrics(self):
        """打印稳定性指标"""
        if not self.error_history:
            return
        
        errors = np.array(self.error_history)
        
        print("\n📊 MuJoCo仿真稳定性分析:")
        print("=" * 50)
        print(f"平均误差: {np.mean(errors):.6f}m")
        print(f"误差标准差: {np.std(errors):.6f}m")
        print(f"最大误差: {np.max(errors):.6f}m")
        print(f"最小误差: {np.min(errors):.6f}m")
        print(f"误差范围: {np.max(errors) - np.min(errors):.6f}m")
        print(f"误差变异系数: {np.std(errors)/np.mean(errors)*100:.2f}%")
        
        # 稳定性判断
        if np.std(errors) < 0.01:
            stability = "优秀"
        elif np.std(errors) < 0.02:
            stability = "良好"
        elif np.std(errors) < 0.05:
            stability = "一般"
        else:
            stability = "需要改进"
        
        print(f"稳定性评级: {stability}")
        print("=" * 50)

def run_mujoco_simulation():
    """运行改进的MuJoCo仿真"""
    if not MUJOCO_AVAILABLE:
        print("❌ MuJoCo未安装，无法运行仿真")
        return
    
    print("🚀 运行改进的MuJoCo任务空间控制仿真")
    
    # 生成更平滑的测试轨迹
    target_positions, target_velocities = generate_smooth_test_trajectory()
    
    # 创建改进的MuJoCo仿真器 - 使用1000Hz控制频率
    mujoco_simulator = MuJoCoTaskSpaceSimulator(control_frequency=100)
    
    if mujoco_simulator.model is None:
        print("❌ MuJoCo仿真器初始化失败")
        return
    
    # 运行仿真
    mujoco_simulator.run_trajectory(target_positions, target_velocities, 
                                  enable_viewer=True, real_time=True)
    
    # 计算性能指标
    metrics = mujoco_simulator.calculate_performance_metrics()
    
    print(f"\n📊 改进的MuJoCo仿真性能指标:")
    print(f"   RMSE: {metrics['rmse']:.4f}m")
    print(f"   最大误差: {metrics['max_error']:.4f}m")
    print(f"   最终误差: {metrics['final_error']:.4f}m")
    print(f"   稳定时间: {metrics['settling_time']:.2f}s")
    
    # 绘制详细的调参分析图
    print("\n🎨 绘制改进的MuJoCo仿真分析图...")
    mujoco_simulator.plot_mujoco_results(target_positions, target_velocities)
    
    return metrics

if __name__ == "__main__":
    # 运行原始对比测试
    main()
    
    # 如果MuJoCo可用，运行MuJoCo仿真
    if MUJOCO_AVAILABLE:
        print("\n" + "="*60)
        print("🎮 启动MuJoCo仿真")
        print("="*60)
        run_mujoco_simulation()
    else:
        print("\n💡 提示: 安装MuJoCo后可运行3D仿真")
