import numpy as np
import mujoco
import mujoco.viewer
import os
import time
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PPoly


class QuinticPolynomialInterpolator:
    def __init__(self, start, end, num_steps):
        self.start = start
        self.end = end
        self.num_steps = num_steps
    def generate(self):
        # 5次多项式边界条件：起止点位置、速度、加速度均为0
        t = np.linspace(0, 1, self.num_steps)
        n = len(self.start)
        traj = np.zeros((self.num_steps, n))
        for j in range(n):
            # 5次多项式系数
            # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
            # 边界条件：q(0)=q0, q(1)=q1, q'(0)=q'(1)=0, q''(0)=q''(1)=0
            q0, q1 = self.start[j], self.end[j]
            a0 = q0
            a1 = 0
            a2 = 0
            a3 = 10*(q1-q0)
            a4 = -15*(q1-q0)
            a5 = 6*(q1-q0)
            traj[:, j] = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
        return traj

class PIDController:
    """
    用于多关节轨迹跟踪的PID控制器。
    每个关节独立控制，支持向量化操作。
    """
    def __init__(self, kp, ki, kd, num_joints):
        self.kp = np.array(kp)  # 比例增益，数组
        self.ki = np.array(ki)  # 积分增益，数组
        self.kd = np.array(kd)  # 微分增益，数组
        self.prev_error = np.zeros(num_joints)  # 上一次误差，数组
        self.integral = np.zeros(num_joints)    # 积分项，数组

    def compute_control(self, error, dt):
        """
        根据误差计算每个关节的控制信号。
        :param error: 当前误差，数组
        :param dt: 时间步长
        :return: 控制信号，数组
        """
        self.integral += error * dt  # 更新积分项
        derivative = (error - self.prev_error) / dt  # 计算微分项
        self.prev_error = error  # 更新上一次误差
        return self.kp * error + self.ki * self.integral + self.kd * derivative  # 返回控制信号

class AdaptivePIDController(PIDController):
    """
    误差自适应PID控制器：根据当前误差动态调整kp。
    末端关节自适应幅度更大。
    """
    def compute_control(self, error, dt):
        # 针对末端关节加大自适应幅度
        alpha = np.array([0.5, 0.5, 0.5, 0.5, 1.2, 2.0])
        adaptive_kp = self.kp * (1 + alpha * np.abs(error))
        adaptive_ki = self.ki
        adaptive_kd = self.kd
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return adaptive_kp * error + adaptive_ki * self.integral + adaptive_kd * derivative

class UR5eTrajectoryTracking:
    """
    使用PID控制实现UR5e的轨迹跟踪。
    """
    def __init__(self, xml_path, kp, ki, kd, num_joints=6):
        self.xml_path = xml_path  # MJCF文件路径
        self.num_joints = num_joints
        # 初始化多关节PID控制器
        self.pid_controller = PIDController(kp, ki, kd, num_joints)

    def load_model(self):
        """
        从MJCF文件加载UR5e模型。
        """
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)  # 加载模型
        self.data = mujoco.MjData(self.model)  # 创建模拟数据对象
        # 地面已在MJCF文件中定义，无需动态添加

    def track_trajectory(self, trajectory, dt, save_dir=None):
        actual_positions = []
        target_positions = []
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            for target in trajectory:
                current_position = self.data.qpos[:self.num_joints].copy()
                actual_positions.append(current_position)
                target_positions.append(target.copy())
                error = target - current_position
                control_signal = self.pid_controller.compute_control(error, dt)
                self.data.ctrl[:self.num_joints] = control_signal
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.1)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.1)
        actual_positions = np.array(actual_positions)
        target_positions = np.array(target_positions)
        # 保存数据
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, 'trajectory_actual.npy'), actual_positions)
            np.save(os.path.join(save_dir, 'trajectory_target.npy'), target_positions)
        return actual_positions, target_positions

if __name__ == "__main__":
    # 定义ur5e_control目录下data文件夹
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # 定义UR5e MJCF文件的路径（加载带地面的scene.xml）
    xml_path = os.path.join(os.path.dirname(__file__), '../models/universal_robots_ur5e/scene.xml')

    num_joints = 6  # UR5e机械臂关节数
    # 针对末端关节优化PID参数
    kp = [5.0, 5.0, 5.0, 5.0, 8.0, 15.0]
    ki = [0, 0, 0, 0, 0, 0]
    kd = [0.5, 0.5, 0.5, 0.5, 1.8, 2.5]

    # 初始化轨迹跟踪系统（可切换为自适应PID）
    tracking_system = UR5eTrajectoryTracking(xml_path, kp, ki, kd, num_joints=num_joints)
    tracking_system.pid_controller = AdaptivePIDController(kp, ki, kd, num_joints)

    # 加载模型
    tracking_system.load_model()

    # 多段工业轨迹关键点（单位：度）
    key_points_deg = [
        [0, -90, 90, -90, -90, 0],         # Home
        [45, -60, 60, -60, 60, 90],        # Approach（末端关节大幅旋转）
        [45, -62, 62, -58, -60, -90],      # Operation（末端关节反向旋转）
        [45, -55, 55, -65, 120, 180]       # Retract（末端关节再大幅旋转）
    ]
    key_points = [np.deg2rad(p) for p in key_points_deg]
    num_segments = len(key_points) - 1
    num_steps_per_segment = 1000  # 每段插值步数
    all_trajectory = []
    for i in range(num_segments):
        interpolator = QuinticPolynomialInterpolator(key_points[i], key_points[i+1], num_steps_per_segment)
        segment_traj = interpolator.generate()
        if i > 0:
            segment_traj = segment_traj[1:]
        all_trajectory.append(segment_traj)
    trajectory = np.vstack(all_trajectory)
    # 设置初始位
    tracking_system.data.qpos[:num_joints] = key_points[0]

    # 只用五次多项式插值对比
    interpolators = {
        'quintic': lambda: trajectory
    }
    results = {}
    for name, traj_func in interpolators.items():
        traj = traj_func()
        tracking_system.data.qpos[:num_joints] = key_points[0]
        actual_positions, target_positions = tracking_system.track_trajectory(traj, dt=0.01)
        results[name] = (actual_positions, target_positions)
        errors = target_positions - actual_positions
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        np.save(os.path.join(data_dir, f'rmse_{name}.npy'), rmse)
        print(f'{name} 每个关节的RMSE:', rmse)
    # 对比图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    joint_names = [f'Joint {i+1}' for i in range(num_joints)]
    for i in range(num_joints):
        ax = axes[i // 3, i % 3]
        for name in interpolators.keys():
            actual, target = results[name]
            ax.plot(target[:, i], label=f'{name.capitalize()} Target', linestyle='--')
            ax.plot(actual[:, i], label=f'{name.capitalize()} Actual')
        ax.set_title(joint_names[i])
        ax.set_xlabel('Step')
        ax.set_ylabel('Angle (rad)')
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    img_path = os.path.join(data_dir, 'joint_trajectories_compare.png')
    plt.savefig(img_path)
    plt.close()
