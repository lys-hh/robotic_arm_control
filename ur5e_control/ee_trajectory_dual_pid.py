import numpy as np
if not hasattr(np, 'float'):
    np.float = float
from mujoco import MjModel, MjData
from mujoco import mj_name2id, mj_id2name
import mujoco.viewer
import os
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF_PATH = os.path.join(BASE_DIR, 'models', 'universal_robots_ur5e', 'scene.xml')

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

HOME_Q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
ZERO_Q = np.zeros(6)

print('加载mujoco模型...')
mj_model = MjModel.from_xml_path(MJCF_PATH)
mj_data = MjData(mj_model)

EE_SITE = 'ee_site'
site_id = mj_name2id(mj_model, 6, EE_SITE)  # 6=mjOBJ_SITE

# 轨迹生成函数（圆形轨迹）
def generate_circle_traj(center, radius, axis, n_points):
    t = np.linspace(0, 2*np.pi, n_points)
    traj = np.zeros((n_points, 3))
    if axis == 'xy':
        traj[:, 0] = center[0] + radius * np.cos(t)
        traj[:, 1] = center[1] + radius * np.sin(t)
        traj[:, 2] = center[2]
    elif axis == 'xz':
        traj[:, 0] = center[0] + radius * np.cos(t)
        traj[:, 1] = center[1]
        traj[:, 2] = center[2] + radius * np.sin(t)
    elif axis == 'yz':
        traj[:, 0] = center[0]
        traj[:, 1] = center[1] + radius * np.cos(t)
        traj[:, 2] = center[2] + radius * np.sin(t)
    return traj

# ========== 工具函数定义 ========== 
def get_site_quat(mj_data, site_id):
    """
    自动兼容mujoco的site_xquat和site_xmat，返回四元数（wxyz）。
    如果site_xmat全为零，自动报错提醒用户先调用mj_forward。
    """
    try:
        quat = mj_data.site_xquat[site_id].copy()
    except AttributeError:
        mat = mj_data.site_xmat[site_id].reshape(3, 3)
        if np.allclose(mat, 0):
            raise RuntimeError("site_xmat全为零，说明mj_forward未调用或qpos未初始化。请先设置关节角度并调用mj_forward。")
        quat = R.from_matrix(mat).as_quat()  # xyzw
        quat = np.roll(quat, 1)  # 转为wxyz
    return quat

# ========== Mujoco数值法逆运动学 ========== 
def mujoco_ik_solve(mj_model, mj_data, site_id, target_pos, q_init, use_orientation=False, target_quat=None, max_iter=100, tol=1e-6):
    """
    用Mujoco自带的雅可比和正运动学做数值法逆运动学。
    target_pos: 目标末端位置 (3,)
    q_init: 初始关节角度 (6,)
    use_orientation: 是否同时控制末端朝向
    target_quat: 目标朝向（四元数，wxyz）
    返回: 逆解关节角度 (6,)
    """
    q = q_init.copy()
    n = 6
    min_iter = 5  # 强制至少迭代5步，避免提前收敛
    for iter_idx in range(max_iter):
        mj_data.qpos[:n] = q
        from mujoco import mj_forward, mj_jacSite
        mj_forward(mj_model, mj_data)
        ee_pos = mj_data.site_xpos[site_id].copy()
        pos_error = target_pos - ee_pos
        if use_orientation:
            ee_quat = get_site_quat(mj_data, site_id)
            r1 = R.from_quat(ee_quat[[1,2,3,0]])
            r2 = R.from_quat(target_quat[[1,2,3,0]])
            rotvec_error = (r2 * r1.inv()).as_rotvec()
            err = np.concatenate([pos_error, rotvec_error])
            jacp = np.zeros((3, n))
            jacr = np.zeros((3, n))
            mj_jacSite(mj_model, mj_data, jacp, jacr, site_id)
            J = np.vstack([jacp, jacr])
            dq = np.linalg.pinv(J) @ err
            if np.linalg.norm(err) < tol and iter_idx >= min_iter:
                break
        else:
            jacp = np.zeros((3, n))
            mj_jacSite(mj_model, mj_data, jacp, None, site_id)
            dq = np.linalg.pinv(jacp) @ pos_error
            if np.linalg.norm(pos_error) < tol and iter_idx >= min_iter:
                break
        q += dq
    return q

# 关节PID控制器
class JointPID:
    def __init__(self, Kp, Ki, Kd, n_joints):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.n = n_joints
        self.integral = np.zeros(n_joints)
        self.prev_error = np.zeros(n_joints)
    def reset(self):
        self.integral[:] = 0
        self.prev_error[:] = 0
    def step(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        d_error = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * d_error

# 仿真参数
n_steps = 200
# 用home位末端为圆心，半径0.05，平面xy，生成圆弧轨迹
mj_data.qpos[:6] = HOME_Q
from mujoco import mj_forward
mj_forward(mj_model, mj_data)
home_ee = mj_data.site_xpos[site_id].copy()
center = home_ee
radius = 0.1
axis = 'xy'
traj = generate_circle_traj(center, radius, axis, n_steps)

# 1. 起点对齐：插值home位末端到直线起点（其实就是home_ee本身）
home_q = HOME_Q.copy()
q = home_q.copy()
q_traj = [home_q.copy()]
mj_data.qpos[:6] = home_q
mj_forward(mj_model, mj_data)
home_quat = get_site_quat(mj_data, site_id)
start_ee = traj[0]
pre_steps = 50
pre_traj = np.linspace(home_ee, start_ee, pre_steps)

# 2. 拼接完整轨迹
full_traj = np.vstack([pre_traj, traj])

# 关节空间插值（三次插值）前，打印full_traj的前后几个点
print("full_traj 前3个点:", full_traj[:3])
print("full_traj 后3个点:", full_traj[-3:])

# ========== 轨迹逆解主循环前，增加开关变量 ==========
use_orientation = False  # True=位置+姿态控制，False=仅位置控制
max_error = 10.0  # 临时调大，便于调试逆解

# ========== 轨迹逆解主循环 ========== 
q_traj = [home_q.copy()]
q = home_q + np.random.uniform(-0.05, 0.05, size=6)  # 只在第一步加扰动
for i in range(1, len(full_traj)):
    q_new = mujoco_ik_solve(
        mj_model, mj_data, site_id, full_traj[i], q,
        use_orientation=use_orientation,
        target_quat=home_quat,
        max_iter=200
    )
    mj_data.qpos[:6] = q_new
    mj_forward(mj_model, mj_data)
    fk_pos = mj_data.site_xpos[site_id].copy()
    pos_error = np.linalg.norm(fk_pos - full_traj[i])
    # 只在关键步打印pos_error和重置信息
    if i == 1 or i % 50 == 0 or i == len(full_traj)-1 or pos_error > max_error:
        print(f"第{i}步 pos_error: {pos_error:.6f}")
    if pos_error > max_error:
        print(f'第{i}步逆解误差过大({pos_error:.4f})，重置初值为home_q+扰动')
        q_new = mujoco_ik_solve(
            mj_model, mj_data, site_id, full_traj[i], home_q + np.random.uniform(-0.05, 0.05, size=6),
            use_orientation=use_orientation,
            target_quat=home_quat,
            max_iter=200
        )
        mj_data.qpos[:6] = q_new
        mj_forward(mj_model, mj_data)
        fk_pos = mj_data.site_xpos[site_id].copy()
        pos_error = np.linalg.norm(fk_pos - full_traj[i])
        print(f"[重置后] 第{i}步 pos_error: {pos_error:.6f}")
    q = q_new  # 关键：用上一步的逆解结果作为下一步初值
    q_traj.append(q.copy())
q_traj = np.array(q_traj)

# 关节空间插值（三次插值）
interp_steps = 10
n_dense = (len(q_traj)-1)*interp_steps + 1
x = np.arange(len(q_traj))
x_dense = np.linspace(0, len(q_traj)-1, n_dense)
q_traj_dense = []
for j in range(q_traj.shape[1]):
    cs = CubicSpline(x, q_traj[:,j])
    q_traj_dense.append(cs(x_dense))
q_traj_dense = np.stack(q_traj_dense, axis=1)
# 插值后，打印q_traj_dense的前后几个点
print("q_traj_dense 前3个点:", q_traj_dense[:3])
print("q_traj_dense 后3个点:", q_traj_dense[-3:])

# PID闭环控制仿真
pid = JointPID(Kp=2.0, Ki=0.0, Kd=0.1, n_joints=6)
dt = 0.01
actual_traj = []
q_sim = home_q.copy()
dq_norms = []  # 用于统计dq范数

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    for i in range(len(q_traj_dense)):
        target_q = q_traj_dense[i]
        dq = pid.step(target_q, q_sim, dt)
        dq_norms.append(np.linalg.norm(dq))
        q_sim += dq * dt  # 1. 更新关节角度
        mj_data.qpos[:6] = q_sim  # 2. 传递给仿真引擎
        from mujoco import mj_forward
        mj_forward(mj_model, mj_data)  # 3. 刷新仿真状态
        actual_traj.append(mj_data.site_xpos[site_id].copy())
        # 只在关键步打印详细信息
        # if i == 0 or i % 50 == 0 or i == len(q_traj_dense)-1 or np.allclose(dq, 0):
        #     print(f"Step {i}: q_sim = {q_sim}")
        #     print(f"         dq = {dq}")
        #     print(f"         target_q - q_sim = {target_q - q_sim}")
        #     if np.allclose(dq, 0):
        #         print(f"[警告] Step {i} dq 全为零，PID无输出，机械臂不会动！")
        viewer.sync()  # 5. 渲染同步

actual_traj = np.array(actual_traj)
# ========== 调试：检查实际轨迹结果 ==========
print("actual_traj unique rows:", np.unique(actual_traj, axis=0).shape[0])
print("actual_traj 前3个点:", actual_traj[:3])
print("actual_traj 后3个点:", actual_traj[-3:])
if np.unique(actual_traj, axis=0).shape[0] == 1:
    print("[警告] actual_traj 全部相同，机械臂实际没有动！请检查PID控制器、目标轨迹和逆解。")
# 汇总统计dq范数
if dq_norms:
    dq_norms = np.array(dq_norms)
    print(f"PID输出dq范数统计：最大={dq_norms.max():.4f}，最小={dq_norms.min():.4f}，平均={dq_norms.mean():.4f}")

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(full_traj[:,0], full_traj[:,1], full_traj[:,2], label='Target Traj', color='r')
ax.plot(actual_traj[:,0], actual_traj[:,1], actual_traj[:,2], label='Actual Traj', color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('End-Effector Trajectory Tracking (PID)')
plt.show()
