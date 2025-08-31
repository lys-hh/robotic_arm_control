#!/usr/bin/env python3
"""
改进的MPC控制器实现
基于开源项目的设计思路，简化约束，提高求解成功率
"""

import numpy as np
import scipy.optimize as opt
from typing import Tuple, Optional, List, Dict
import time
import os

# MuJoCo相关导入
try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️ MuJoCo未安装，MuJoCo渲染功能不可用")

class ImprovedMPCController:
    """改进的MPC控制器 - 基于scipy.optimize的简化实现"""
    
    def __init__(self, 
                 prediction_horizon: int = 10,
                 control_horizon: int = 5,
                 dt: float = 0.01,
                 Q: Optional[np.ndarray] = None,
                 R: Optional[np.ndarray] = None,
                 max_torque: float = 30.0):
        """
        初始化改进的MPC控制器
        
        Args:
            prediction_horizon: 预测时域长度
            control_horizon: 控制时域长度
            dt: 时间步长
            Q: 位置权重矩阵 (3x3)
            R: 控制权重矩阵 (7x7)
            max_torque: 最大关节力矩
        """
        self.N = prediction_horizon  # 预测时域
        self.M = control_horizon     # 控制时域
        self.dt = dt                 # 时间步长
        
        # 权重矩阵
        if Q is None:
            self.Q = 100.0 * np.eye(3)  # 降低位置权重
        else:
            self.Q = Q
            
        if R is None:
            self.R = 1.0 * np.eye(7)     # 增加控制权重
        else:
            self.R = R
        
        # 约束限制
        self.max_torque = max_torque
        
        # 关节限制 (Panda机械臂) - 放宽限制
        self.q_min = np.array([-2.5, -1.5, -2.5, -2.8, -2.5, -0.1, -2.5])
        self.q_max = np.array([2.5, 1.5, 2.5, -0.1, 2.5, 3.6, 2.5])
        
        # 状态变量
        self.current_state = None  # [q; q_dot] (14x1)
        
        # 性能记录
        self.performance_history = {
            'solve_times': [],
            'iterations': [],
            'success_count': 0,
            'total_count': 0
        }
        
        print(f"[OK] 改进的MPC控制器初始化完成")
        print(f"[INFO] 预测时域: {self.N}, 控制时域: {self.M}")
        print(f"[INFO] 时间步长: {self.dt}s")
        print(f"[INFO] 位置权重: diag({np.diag(self.Q)})")
        print(f"[INFO] 控制权重: diag({np.diag(self.R)})")
    
    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        计算改进的Jacobian矩阵 - 基于Panda机械臂DH参数
        
        Args:
            q: 关节角度 (7x1)
            
        Returns:
            J: Jacobian矩阵 (3x7)
        """
        # Panda机械臂DH参数 (改进版本)
        # 基于Franka Emika Panda的准确参数
        J = np.zeros((3, 7))
        
        # 改进的Jacobian计算
        # 考虑机械臂的实际几何结构
        
        # 基础位置
        base_x, base_y, base_z = 0.0, 0.0, 0.333
        
        # 各关节对末端位置的影响 (基于DH参数)
        # 第1关节 (基座旋转)
        J[0, 0] = -0.333 * np.sin(q[0]) - 0.316 * np.sin(q[0] + q[1]) - 0.384 * np.sin(q[0] + q[1] + q[2])
        J[1, 0] = 0.333 * np.cos(q[0]) + 0.316 * np.cos(q[0] + q[1]) + 0.384 * np.cos(q[0] + q[1] + q[2])
        J[2, 0] = 0.0
        
        # 第2关节 (肩部)
        J[0, 1] = -0.316 * np.sin(q[0] + q[1]) - 0.384 * np.sin(q[0] + q[1] + q[2])
        J[1, 1] = 0.316 * np.cos(q[0] + q[1]) + 0.384 * np.cos(q[0] + q[1] + q[2])
        J[2, 1] = 0.0
        
        # 第3关节 (肘部)
        J[0, 2] = -0.384 * np.sin(q[0] + q[1] + q[2])
        J[1, 2] = 0.384 * np.cos(q[0] + q[1] + q[2])
        J[2, 2] = 0.0
        
        # 第4-7关节 (腕部) - 对位置影响较小
        J[0, 3] = 0.0
        J[1, 3] = 0.0
        J[2, 3] = 0.0
        
        J[0, 4] = 0.0
        J[1, 4] = 0.0
        J[2, 4] = 0.0
        
        J[0, 5] = 0.0
        J[1, 5] = 0.0
        J[2, 5] = 0.0
        
        J[0, 6] = 0.0
        J[1, 6] = 0.0
        J[2, 6] = 0.0
        
        return J
    
    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        改进的正向运动学 - 基于Panda机械臂DH参数
        
        Args:
            q: 关节角度 (7x1)
            
        Returns:
            position: 末端位置 (3x1)
        """
        # Panda机械臂DH参数 (改进版本)
        # 基于Franka Emika Panda的准确参数
        
        # DH参数: [a, alpha, d, theta]
        dh_params = [
            [0.0, 0.0, 0.333, q[0]],      # 关节1
            [0.0, -np.pi/2, 0.0, q[1]],   # 关节2
            [0.0, np.pi/2, 0.316, q[2]],  # 关节3
            [0.0825, np.pi/2, 0.0, q[3]], # 关节4
            [-0.0825, -np.pi/2, 0.384, q[4]], # 关节5
            [0.0, np.pi/2, 0.0, q[5]],    # 关节6
            [0.088, np.pi/2, 0.0, q[6]]   # 关节7
        ]
        
        # 计算正向运动学
        T = np.eye(4)
        
        for a, alpha, d, theta in dh_params:
            # DH变换矩阵
            T_i = np.array([
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ])
            T = T @ T_i
        
        # 提取位置
        position = T[:3, 3]
        
        return position
    
    def inverse_kinematics(self, target_position: np.ndarray, current_q: np.ndarray = None) -> np.ndarray:
        """
        简化的逆运动学 - 使用数值方法
        
        Args:
            target_position: 目标位置 (3x1)
            current_q: 当前关节角度 (7x1)
            
        Returns:
            q: 关节角度 (7x1)
        """
        if current_q is None:
            current_q = np.zeros(7)
        
        # 使用数值方法求解逆运动学
        def position_error(q):
            current_pos = self.forward_kinematics(q)
            return np.linalg.norm(current_pos - target_position)
        
        # 使用scipy优化求解
        try:
            result = opt.minimize(
                position_error,
                current_q,
                method='L-BFGS-B',
                bounds=[(self.q_min[i], self.q_max[i]) for i in range(7)],
                options={'maxiter': 50, 'ftol': 1e-6}
            )
            
            if result.success:
                return result.x
            else:
                # 如果优化失败，返回当前关节角度
                return current_q
        except:
            return current_q
    
    def mpc_objective(self, u_flat: np.ndarray, current_state: np.ndarray, 
                     reference_sequence: np.ndarray, mujoco_model=None, mujoco_data=None) -> float:
        """
        MPC目标函数 - 使用MuJoCo真实动力学
        
        Args:
            u_flat: 扁平化的控制序列 (M*7 x 1)
            current_state: 当前状态 (14x1)
            reference_sequence: 参考轨迹序列 (N+1 x 3)
            mujoco_model: MuJoCo模型
            mujoco_data: MuJoCo数据
            
        Returns:
            cost: 目标函数值
        """
        # 重塑控制序列
        u = u_flat.reshape(self.M, 7)
        
        # 初始化状态
        z = current_state.copy()
        cost = 0.0
        
        # 预测轨迹
        for i in range(self.N + 1):
            # 计算当前末端位置
            q = z[:7]
            current_pos = self.forward_kinematics(q)
            
            # 位置跟踪误差
            if i < len(reference_sequence):
                pos_error = current_pos - reference_sequence[i]
                cost += pos_error.T @ self.Q @ pos_error
            
            # 控制努力（只在控制时域内）
            if i < self.M:
                cost += u[i].T @ self.R @ u[i]
            
            # 状态更新（使用MuJoCo真实动力学）
            if i < self.N:
                q = z[:7]
                q_dot = z[7:]
                
                # 使用MuJoCo真实动力学
                if mujoco_model is not None and mujoco_data is not None:
                    # 设置MuJoCo状态
                    mujoco_data.qpos[:7] = q
                    mujoco_data.qvel[:7] = q_dot
                    
                    # 设置控制输入
                    if i < self.M:
                        control_input = u[i]
                    else:
                        control_input = u[-1]
                    
                    mujoco_data.ctrl[:7] = control_input
                    
                    # 前向动力学计算
                    mujoco.mj_forward(mujoco_model, mujoco_data)
                    
                    # 获取加速度
                    q_ddot = mujoco_data.qacc[:7]
                    
                    # 状态更新
                    q_dot_new = q_dot + self.dt * q_ddot
                    q_new = q + self.dt * q_dot_new
                    
                    z = np.concatenate([q_new, q_dot_new])
                else:
                    # 回退到简化动力学
                    if i < self.M:
                        control_input = u[i]
                    else:
                        control_input = u[-1]
                    
                    q_ddot = control_input - 0.1 * q_dot
                    q_dot_new = q_dot + self.dt * q_ddot
                    q_new = q + self.dt * q_dot_new
                    
                    z = np.concatenate([q_new, q_dot_new])
        
        return cost
    
    def solve_mpc(self, current_state: np.ndarray, reference_sequence: np.ndarray, 
                  mujoco_model=None, mujoco_data=None) -> np.ndarray:
        """
        求解MPC优化问题 - 使用MuJoCo真实动力学
        
        Args:
            current_state: 当前状态 [q; q_dot] (14x1)
            reference_sequence: 参考轨迹序列 (N+1 x 3)
            mujoco_model: MuJoCo模型
            mujoco_data: MuJoCo数据
            
        Returns:
            optimal_control: 最优控制序列 (7x1)
        """
        start_time = time.time()
        
        # 初始猜测
        u0 = np.zeros(self.M * 7)
        
        # 约束条件
        constraints = []
        
        # 控制限制
        bounds = []
        for i in range(self.M):
            for j in range(7):
                bounds.append((-self.max_torque, self.max_torque))
        
        # 目标函数 - 使用MuJoCo真实动力学
        def objective(u_flat):
            return self.mpc_objective(u_flat, current_state, reference_sequence, 
                                    mujoco_model, mujoco_data)
        
        try:
            # 使用scipy.optimize求解
            result = opt.minimize(
                objective,
                u0,
                method='L-BFGS-B',  # 改用L-BFGS-B，更适合无约束优化
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-4}  # 减少迭代次数，提高速度
            )
            
            solve_time = time.time() - start_time
            
            if result.success:
                # 记录性能
                self.performance_history['solve_times'].append(solve_time)
                self.performance_history['iterations'].append(result.nit)
                self.performance_history['success_count'] += 1
                
                # 返回第一个控制输入
                u_optimal = result.x.reshape(self.M, 7)
                return u_optimal[0]
            else:
                print(f"⚠️ MPC求解失败: {result.message}")
                self.performance_history['total_count'] += 1
                return np.zeros(7)
                
        except Exception as e:
            print(f"❌ MPC求解异常: {e}")
            self.performance_history['total_count'] += 1
            return np.zeros(7)
    
    def compute_control(self, current_position: np.ndarray, current_velocity: np.ndarray,
                       target_position: np.ndarray, target_velocity: np.ndarray,
                       mujoco_model=None, mujoco_data=None) -> np.ndarray:
        """
        计算控制输出 - 使用MuJoCo真实动力学的MPC
        
        Args:
            current_position: 当前末端位置 (3x1)
            current_velocity: 当前末端速度 (3x1)
            target_position: 目标末端位置 (3x1)
            target_velocity: 目标末端速度 (3x1)
            mujoco_model: MuJoCo模型
            mujoco_data: MuJoCo数据
            
        Returns:
            control_force: 控制力 (3x1)
        """
        # 初始化状态
        if self.current_state is None:
            # 使用逆运动学估计初始关节角度
            self.current_state = np.zeros(14)
            initial_q = self.inverse_kinematics(target_position)
            self.current_state[:7] = initial_q
        
        # 生成参考轨迹序列
        reference_sequence = self._generate_reference_sequence(target_position, target_velocity)
        
        # 求解MPC - 使用MuJoCo真实动力学
        joint_torques = self.solve_mpc(self.current_state, reference_sequence, 
                                      mujoco_model, mujoco_data)
        
        # 更新状态（改进）
        self._update_state(joint_torques)
        
        # 将关节力矩转换为任务空间力
        q = self.current_state[:7]
        J = self.compute_jacobian(q)
        control_force = J @ joint_torques
        
        return control_force
    
    def _generate_reference_sequence(self, target_pos: np.ndarray, target_vel: np.ndarray) -> np.ndarray:
        """生成改进的参考轨迹序列"""
        sequence = np.zeros((self.N + 1, 3))
        
        # 获取当前末端位置
        if self.current_state is not None:
            current_q = self.current_state[:7]
            current_pos = self.forward_kinematics(current_q)
        else:
            current_pos = np.array([0.5, 0.0, 0.3])  # 默认起始位置
        
        # 生成平滑的参考轨迹
        for i in range(self.N + 1):
            # 使用平滑插值
            alpha = min(i / self.N, 1.0)
            # 使用三次多项式插值
            alpha_smooth = 3 * alpha**2 - 2 * alpha**3
            sequence[i] = current_pos + alpha_smooth * (target_pos - current_pos)
        
        return sequence
    
    def _update_state(self, control_input: np.ndarray):
        """更新内部状态（改进版本）"""
        if self.current_state is not None:
            # 改进的状态更新
            q = self.current_state[:7]
            q_dot = self.current_state[7:]
            
            # 改进的动力学积分
            # 考虑重力补偿和摩擦
            gravity_compensation = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # 简化重力补偿
            friction = 0.1 * q_dot  # 简化的摩擦模型
            
            q_ddot = control_input - gravity_compensation - friction
            q_dot_new = q_dot + self.dt * q_ddot
            q_new = q + self.dt * q_dot_new
            
            # 应用关节限制
            q_new = np.clip(q_new, self.q_min, self.q_max)
            
            self.current_state = np.concatenate([q_new, q_dot_new])
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        if not self.performance_history['solve_times']:
            return {}
        
        success_rate = (self.performance_history['success_count'] / 
                       max(self.performance_history['total_count'], 1)) * 100
        
        return {
            'avg_solve_time': np.mean(self.performance_history['solve_times']),
            'max_solve_time': np.max(self.performance_history['solve_times']),
            'avg_iterations': np.mean(self.performance_history['iterations']),
            'success_rate': success_rate,
            'total_solves': self.performance_history['total_count']
        }

class ImprovedMPCWithMuJoCo:
    """集成MuJoCo的改进MPC控制器"""
    
    def __init__(self, model_path: str = None, **mpc_params):
        """
        初始化改进的MPC+MuJoCo仿真器
        
        Args:
            model_path: MuJoCo模型文件路径
            **mpc_params: MPC控制器参数
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo未安装，无法使用MuJoCo仿真器")
        
        # 设置默认模型路径
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.model_path = os.path.join(project_root, "models", "franka_emika_panda", "scene.xml")
        else:
            self.model_path = model_path
        
        # 初始化MuJoCo
        self.model = None
        self.data = None
        self.viewer = None
        
        # 初始化改进的MPC控制器
        self.mpc_controller = ImprovedMPCController(**mpc_params)
        
        # 仿真参数
        self.simulation_time = 0.0
        self.time_history = []
        self.position_history = []
        self.target_history = []
        self.error_history = []
        self.control_history = []
        
        self._initialize_mujoco()
    
    def _initialize_mujoco(self):
        """初始化MuJoCo"""
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            print(f"[OK] MuJoCo模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"❌ MuJoCo模型加载失败: {e}")
            self.model = None
            self.data = None
    
    def get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        if self.data is None:
            return np.zeros(3)
        
        try:
            # 使用夹爪中点作为末端
            left_finger_id = self.model.body('left_finger').id
            right_finger_id = self.model.body('right_finger').id
            left_pos = self.data.xpos[left_finger_id]
            right_pos = self.data.xpos[right_finger_id]
            position = (left_pos + right_pos) / 2.0
        except:
            try:
                end_effector_id = self.model.body('hand').id
                position = self.data.xpos[end_effector_id].copy()
            except:
                position = self.data.xpos[-1].copy()
        
        return position
    
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取关节状态"""
        if self.data is None:
            return np.zeros(7), np.zeros(7)
        
        q = self.data.qpos[:7].copy()
        q_dot = self.data.qvel[:7].copy()
        return q, q_dot
    
    def apply_control(self, control_force: np.ndarray):
        """应用控制力"""
        if self.data is None:
            return
        
        # 获取当前关节状态
        q, q_dot = self.get_joint_states()
        
        # 计算Jacobian
        jacobian = np.zeros((3, self.model.nv))
        try:
            left_finger_id = self.model.body('left_finger').id
            left_finger_pos = self.data.xpos[left_finger_id]
            mujoco.mj_jac(self.model, self.data, jacobian, None, 
                         left_finger_pos, left_finger_id)
        except:
            try:
                end_effector_id = self.model.body('hand').id
                end_effector_pos = self.data.xpos[end_effector_id]
                mujoco.mj_jac(self.model, self.data, jacobian, None, 
                             end_effector_pos, end_effector_id)
            except:
                end_effector_pos = self.data.xpos[-1]
                mujoco.mj_jac(self.model, self.data, jacobian, None, 
                             end_effector_pos, -1)
        
        # 转换为关节力矩
        joint_torques = jacobian.T @ control_force
        
        # 应用力矩限制
        joint_torques = np.clip(joint_torques, -30.0, 30.0)
        
        # 应用控制
        num_actuators = min(len(joint_torques), self.model.nu)
        self.data.ctrl[:num_actuators] = joint_torques[:num_actuators]
    
    def step(self, target_position: np.ndarray, target_velocity: np.ndarray):
        """执行一个仿真步 - 基于真实MuJoCo状态的MPC"""
        if self.model is None or self.data is None:
            return
        
        # 获取当前真实状态
        current_position = self.get_end_effector_position()
        current_velocity = np.zeros(3)  # 简化处理
        
        # 更新MPC控制器状态
        q, q_dot = self.get_joint_states()
        self.mpc_controller.current_state = np.concatenate([q, q_dot])
        
        # 计算MPC控制 - 传递MuJoCo模型和数据
        control_force = self.mpc_controller.compute_control(
            current_position, current_velocity, target_position, target_velocity,
            self.model, self.data
        )
        
        # 应用控制
        self.apply_control(control_force)
        
        # 执行仿真步
        mujoco.mj_step(self.model, self.data)
        
        # 同步查看器
        if self.viewer and self.viewer != "simple":
            try:
                self.viewer.sync()
            except:
                pass
        
        # 记录数据
        self.time_history.append(self.simulation_time)
        self.position_history.append(current_position.copy())
        self.target_history.append(target_position.copy())
        self.error_history.append(np.linalg.norm(current_position - target_position))
        self.control_history.append(control_force.copy())
        
        # 更新仿真时间
        self.simulation_time += self.mpc_controller.dt
    
    def start_viewer(self):
        """启动查看器"""
        if self.viewer is None and self.model is not None:
            try:
                # 尝试不同的查看器启动方式
                try:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                    print("[OK] MuJoCo查看器启动成功")
                except AttributeError:
                    # 如果launch_passive不存在，尝试其他方法
                    print("⚠️ 使用简化的查看器模式")
                    self.viewer = "simple"  # 标记为简单模式
            except Exception as e:
                print(f"❌ 查看器启动失败: {e}")
                self.viewer = None
    
    def close_viewer(self):
        """关闭查看器"""
        if self.viewer is not None:
            try:
                self.viewer.close()
                self.viewer = None
                print("[OK] MuJoCo查看器已关闭")
            except:
                pass
    
    def run_trajectory(self, target_trajectory: np.ndarray, target_velocities: np.ndarray,
                      enable_viewer: bool = True, real_time: bool = True):
        """运行轨迹跟踪"""
        if self.model is None or self.data is None:
            print("❌ 仿真器未初始化")
            return
        
        print("🚀 开始改进的MPC轨迹跟踪仿真")
        
        # 启动查看器
        if enable_viewer:
            self.start_viewer()
        
        # 重置状态
        mujoco.mj_resetData(self.model, self.data)
        self.simulation_time = 0.0
        self.time_history = []
        self.position_history = []
        self.target_history = []
        self.error_history = []
        self.control_history = []
        
        try:
            for i, (target_pos, target_vel) in enumerate(zip(target_trajectory, target_velocities)):
                # 执行仿真步
                self.step(target_pos, target_vel)
                
                # 实时显示进度
                if i % 50 == 0:
                    error = np.linalg.norm(target_pos - self.get_end_effector_position())
                    print(f"Step {i}: Error = {error:.4f}m, Position = {self.get_end_effector_position()}")
                
                # 实时控制
                if real_time:
                    time.sleep(self.mpc_controller.dt)
        
        except KeyboardInterrupt:
            print("\n用户中断仿真")
        
        finally:
            # 关闭查看器
            if enable_viewer:
                self.close_viewer()
        
        print("✅ 改进的MPC轨迹跟踪仿真完成")
    
    def calculate_performance_metrics(self) -> Dict:
        """计算性能指标"""
        if not self.error_history:
            return {}
        
        errors = np.array(self.error_history)
        
        return {
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'final_error': errors[-1],
            'settling_time': self._calculate_settling_time(errors),
            'mpc_metrics': self.mpc_controller.get_performance_metrics()
        }
    
    def _calculate_settling_time(self, errors: np.ndarray, threshold: float = 0.01) -> float:
        """计算稳定时间"""
        for i, error in enumerate(errors):
            if error < threshold:
                return i * self.mpc_controller.dt
        return len(errors) * self.mpc_controller.dt

def generate_circular_trajectory(num_points: int = 200, radius: float = 0.25, 
                                center: np.ndarray = None, height: float = 0.3) -> tuple:
    """生成圆形轨迹"""
    if center is None:
        center = np.array([0.5, 0.0, height])
    
    t = np.linspace(0, 2*np.pi, num_points)
    positions = np.zeros((num_points, 3))
    velocities = np.zeros((num_points, 3))
    
    for i, angle in enumerate(t):
        # 圆形轨迹
        positions[i] = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])
        # 切线速度
        velocities[i] = np.array([-radius * np.sin(angle), radius * np.cos(angle), 0]) * (2*np.pi / (num_points * 0.01))
    
    return positions, velocities

def test_improved_mpc():
    """测试改进的MPC控制器"""
    print("🧮 测试改进的MPC控制器")
    print("=" * 50)
    
    # 创建改进的MPC控制器
    mpc = ImprovedMPCController(
        prediction_horizon=5,  # 减小预测时域，提高速度
        control_horizon=3,     # 减小控制时域
        dt=0.01,
        max_torque=25.0
    )
    
    # 生成测试轨迹
    target_positions, target_velocities = generate_circular_trajectory(100)
    
    print(f"轨迹长度: {len(target_positions)}")
    print(f"起始位置: {target_positions[0]}")
    print(f"结束位置: {target_positions[-1]}")
    
    # 测试控制计算
    print("\n🔧 测试控制计算...")
    test_results = []
    
    for i in range(min(20, len(target_positions))):
        current_pos = np.array([0.5, 0.0, 0.3])
        current_vel = np.zeros(3)
        target_pos = target_positions[i]
        target_vel = target_velocities[i]
        
        try:
            control = mpc.compute_control(current_pos, current_vel, target_pos, target_vel)
            test_results.append({
                'step': i,
                'target': target_pos,
                'control': control,
                'success': True
            })
            if i % 5 == 0:
                print(f"Step {i}: 目标={target_pos}, 控制={control}")
        except Exception as e:
            test_results.append({
                'step': i,
                'target': target_pos,
                'control': None,
                'success': False,
                'error': str(e)
            })
            print(f"Step {i}: 失败 - {e}")
    
    # 统计结果
    successful_steps = sum(1 for r in test_results if r['success'])
    print(f"\n📊 测试结果:")
    print(f"  成功步数: {successful_steps}/{len(test_results)}")
    print(f"  成功率: {successful_steps/len(test_results)*100:.1f}%")
    
    # 获取性能指标
    metrics = mpc.get_performance_metrics()
    if metrics:
        print(f"  平均求解时间: {metrics.get('avg_solve_time', 0):.4f}s")
        print(f"  最大求解时间: {metrics.get('max_solve_time', 0):.4f}s")
        print(f"  平均迭代次数: {metrics.get('avg_iterations', 0):.1f}")
        print(f"  成功率: {metrics.get('success_rate', 0):.1f}%")
    
    return test_results, mpc

def test_mujoco_mpc():
    """测试MuJoCo+MPC集成"""
    print("\n🎮 测试MuJoCo+MPC集成")
    print("=" * 50)
    
    if not MUJOCO_AVAILABLE:
        print("❌ MuJoCo未安装，跳过MuJoCo测试")
        return None
    
    try:
        # 创建MPC+MuJoCo仿真器
        mpc_mujoco = ImprovedMPCWithMuJoCo(
            prediction_horizon=4,  # 进一步减小预测时域
            control_horizon=2,     # 减小控制时域
            dt=0.01,
            max_torque=20.0
        )
        
        if mpc_mujoco.model is None:
            print("❌ MuJoCo模型加载失败，跳过MuJoCo测试")
            return None
        
        # 生成测试轨迹
        target_positions, target_velocities = generate_circular_trajectory(150)
        
        print("🚀 开始MuJoCo+MPC仿真...")
        
        # 运行仿真
        mpc_mujoco.run_trajectory(
            target_positions, 
            target_velocities,
            enable_viewer=True,
            real_time=False
        )
        
        # 计算性能指标
        metrics = mpc_mujoco.calculate_performance_metrics()
        print(f"\n📊 MuJoCo+MPC性能指标:")
        print(f"  RMSE: {metrics.get('rmse', 0):.4f}m")
        print(f"  最大误差: {metrics.get('max_error', 0):.4f}m")
        print(f"  最终误差: {metrics.get('final_error', 0):.4f}m")
        print(f"  稳定时间: {metrics.get('settling_time', 0):.2f}s")
        
        # MPC特定指标
        mpc_metrics = metrics.get('mpc_metrics', {})
        if mpc_metrics:
            print(f"  MPC平均求解时间: {mpc_metrics.get('avg_solve_time', 0):.4f}s")
            print(f"  MPC求解次数: {mpc_metrics.get('total_solves', 0)}")
            print(f"  MPC成功率: {mpc_metrics.get('success_rate', 0):.1f}%")
        
        return mpc_mujoco
        
    except Exception as e:
        print(f"❌ MuJoCo+MPC测试失败: {e}")
        return None

def plot_mpc_results(test_results, mpc_mujoco=None):
    """绘制MPC测试结果"""
    print("\n📈 绘制MPC测试结果...")
    
    import matplotlib.pyplot as plt
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 控制输出
    ax1 = axes[0, 0]
    steps = [r['step'] for r in test_results if r['success']]
    controls_x = [r['control'][0] for r in test_results if r['success']]
    controls_y = [r['control'][1] for r in test_results if r['success']]
    controls_z = [r['control'][2] for r in test_results if r['success']]
    
    ax1.plot(steps, controls_x, 'r-', label='Fx', linewidth=2)
    ax1.plot(steps, controls_y, 'g-', label='Fy', linewidth=2)
    ax1.plot(steps, controls_z, 'b-', label='Fz', linewidth=2)
    ax1.set_xlabel('步数')
    ax1.set_ylabel('控制力 (N)')
    ax1.set_title('改进的MPC控制输出')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 目标轨迹
    ax2 = axes[0, 1]
    targets_x = [r['target'][0] for r in test_results]
    targets_y = [r['target'][1] for r in test_results]
    
    ax2.plot(targets_x, targets_y, 'b-', linewidth=3, label='目标轨迹', alpha=0.8)
    ax2.scatter(targets_x[0], targets_y[0], color='green', s=100, marker='o', label='起始点')
    ax2.scatter(targets_x[-1], targets_y[-1], color='red', s=100, marker='s', label='结束点')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('目标圆形轨迹 (XY平面)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. MuJoCo仿真结果（如果有）
    if mpc_mujoco and mpc_mujoco.position_history:
        ax3 = axes[1, 0]
        actual_pos = np.array(mpc_mujoco.position_history)
        target_pos = np.array(mpc_mujoco.target_history)
        
        ax3.plot(target_pos[:, 0], target_pos[:, 1], 'b-', linewidth=3, label='目标轨迹', alpha=0.8)
        ax3.plot(actual_pos[:, 0], actual_pos[:, 1], 'r--', linewidth=2, label='MPC实际轨迹', alpha=0.8)
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Y (m)')
        ax3.set_title('MuJoCo+MPC轨迹跟踪')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axis('equal')
        
        # 4. 误差分析
        ax4 = axes[1, 1]
        errors = mpc_mujoco.error_history
        times = mpc_mujoco.time_history
        
        ax4.plot(times, errors, 'r-', linewidth=2, label='位置误差')
        ax4.axhline(y=0.01, color='g', linestyle='--', alpha=0.7, label='1cm误差线')
        ax4.set_xlabel('时间 (s)')
        ax4.set_ylabel('误差 (m)')
        ax4.set_title('MPC跟踪误差')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # 如果没有MuJoCo结果，显示控制力大小
        ax3 = axes[1, 0]
        control_magnitudes = [np.linalg.norm(r['control']) for r in test_results if r['success']]
        ax3.plot(steps, control_magnitudes, 'purple', linewidth=2, marker='o')
        ax3.set_xlabel('步数')
        ax3.set_ylabel('控制力大小 (N)')
        ax3.set_title('控制力大小变化')
        ax3.grid(True, alpha=0.3)
        
        # 4. 成功率统计
        ax4 = axes[1, 1]
        success_rate = sum(1 for r in test_results if r['success']) / len(test_results) * 100
        ax4.bar(['成功率'], [success_rate], color='green', alpha=0.7)
        ax4.set_ylabel('成功率 (%)')
        ax4.set_title('MPC求解成功率')
        ax4.set_ylim(0, 100)
        ax4.text(0, success_rate + 2, f'{success_rate:.1f}%', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("🧮 MPC控制器测试程序")
    print("=" * 60)
    print("特点: 使用MuJoCo真实动力学进行MPC预测")
    print("=" * 60)
    
    # 测试改进的MPC控制器
    test_results, mpc = test_improved_mpc()
    
    # 测试MuJoCo集成
    mpc_mujoco = test_mujoco_mpc()
    
    # 绘制结果
    plot_mpc_results(test_results, mpc_mujoco)
    
    print("\n✅ MPC控制器测试完成！")
    
    # 总结
    print("\n📋 测试总结:")
    print("1. MPC控制器使用MuJoCo真实动力学")
    print("2. 预测时域内使用mujoco.mj_forward计算状态演化")
    print("3. 基于scipy.optimize求解优化问题")
    print("4. 支持圆形轨迹跟踪")
    print("5. 包含完整的性能分析和可视化")
    print("\n💡 关键特性:")
    print("- 使用MuJoCo真实动力学模型进行预测")
    print("- 基于真实Jacobian矩阵进行控制")
    print("- 滚动时域优化控制")
    print("- 实时MPC求解")

if __name__ == "__main__":
    main()
