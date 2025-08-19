import time
import mujoco
import mujoco.viewer
import os
import matplotlib.pyplot as plt
import numpy as np

class TrajectoryPlanner:
    def __init__(self, start_position, end_position, num_steps=100):
        self.start_position = start_position
        self.end_position = end_position
        self.num_steps = num_steps

    def linear_interpolation(self):
        return np.linspace(self.start_position, self.end_position, self.num_steps)

def main():
    # 定义包含地面模型的XML文件路径
    xml_path = os.path.join(os.path.dirname(__file__), '../models/franka_emika_panda/mjx_single_cube.xml')

    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)

    # 创建模拟数据对象
    data = mujoco.MjData(model)

    # 获取地面中立方体和机械臂末端执行器的位置
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")

    # 强制设置立方体和末端执行器的位置
    data.geom_xpos[cube_geom_id] = [0.5, 0, 0.03]  # 立方体位置
    end_effector_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    end_effector_position = [0.0, 0.0, 0.0584]  # 设置末端执行器的完整三维坐标
    data.qpos[end_effector_joint_id:end_effector_joint_id+3] = end_effector_position  # 设置末端执行器初始位置

    cube_position = data.geom_xpos[cube_geom_id]  # 保留完整的三维坐标
    end_effector_position = [data.qpos[end_effector_joint_id]]

    print("强制设置后的立方体位置:", cube_position)
    print("强制设置后的末端执行器位置:", end_effector_position)

    # 使用 TrajectoryPlanner 进行轨迹规划
    planner = TrajectoryPlanner(end_effector_position, cube_position)
    trajectory = planner.linear_interpolation()

    # 启动渲染器（被动模式）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for step in trajectory:
            # 打印轨迹点调试信息
            print("轨迹点:", step)

            control_input = np.zeros(data.ctrl.shape)  # 初始化控制输入为零
            control_input[:3] = step[:3]  # 设置目标位置为轨迹点的X、Y、Z坐标
            data.ctrl[:] = control_input

            # 打印控制输入调试信息
            print("控制输入:", control_input)

            # 运行模拟
            mujoco.mj_step(model, data)

            # 同步渲染器与模拟状态
            viewer.sync()

            # 添加延迟以便观察渲染效果
            time.sleep(0.1)

    # 打印模型信息
    print("模型加载成功:", model)

if __name__ == "__main__":
    main()