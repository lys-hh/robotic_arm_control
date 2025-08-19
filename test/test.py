import mujoco
import mujoco.viewer
import os

def main():
    # 定义包含地面模型的XML文件路径
    xml_path = os.path.join(os.path.dirname(__file__), '../models/franka_emika_panda/mjx_single_cube.xml')

    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)

    # 创建模拟数据对象
    data = mujoco.MjData(model)

    # 启动渲染器（被动模式）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("渲染器已启动，您可以手动拖动关节。")
        while True:
            # 运行模拟
            mujoco.mj_step(model, data)

            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

            # 同步渲染器与模拟状态
            viewer.sync()

if __name__ == "__main__":
    main()
