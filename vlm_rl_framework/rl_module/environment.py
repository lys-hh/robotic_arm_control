"""
强化学习环境封装
"""

import gymnasium as gym
import numpy as np
import mujoco
import os
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PandaEnvironment(gym.Env):
    """
    Panda机械臂强化学习环境
    
    状态空间: [关节角度(7), 目标位置(3), 末端位置(3), 关节速度(7), 接触力(3)]
    动作空间: [关节力矩(7)]
    """
    
    def __init__(self, xml_path: str = None, max_steps: int = 1000):
        """
        初始化环境
        
        Args:
            xml_path: MuJoCo XML文件路径（如果为None，使用默认场景）
            max_steps: 最大步数
        """
        super().__init__()
        
        # 如果没有提供XML路径，使用默认场景
        if xml_path is None:
            xml_path = "models/franka_emika_panda/scene_with_camera.xml"
        
        self.xml_path = xml_path
        self.max_steps = max_steps
        self.current_step = 0
        
        # 检查文件是否存在
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MuJoCo XML文件不存在: {xml_path}")
        
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 定义动作空间 - 7个关节 + 1个夹爪
        self.action_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(8,), dtype=np.float32  # 7个关节 + 1个夹爪
        )
        
        # 定义观察空间 - 增加抓取相关状态
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32  # 23 + 4个抓取状态
        )
        
        # 目标位置（由VLM提供）
        self.target_position = np.array([0.5, 0.0, 0.3])
        
        # 抓取相关状态
        self.task_type = "move_to"  # "move_to" 或 "grasp_and_move"
        self.pickup_position = np.array([0.5, 0.0, 0.35])
        self.place_position = np.array([0.5, 0.0, 0.35])
        self.is_grasped = False
        self.grasp_target_reached = False
        
        # 相机设置
        self.camera_id = None
        self._setup_camera()
        
        logger.info("Panda环境初始化完成")
    
    def _setup_camera(self):
        """设置相机"""
        # 查找相机
        camera_names = ["top_view", "side_view", "front_view"]
        for name in camera_names:
            try:
                self.camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
                logger.info(f"找到相机: {name}")
                break
            except:
                continue
        
        if self.camera_id is None:
            logger.warning("未找到相机，将使用默认视角")
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Returns:
            Tuple[np.ndarray, Dict]: 初始状态和信息
        """
        super().reset(seed=seed)
        
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 随机初始化机械臂位置
        self._randomize_initial_position()
        
        # 重置步数
        self.current_step = 0
        
        # 获取初始状态
        state = self._get_state()
        
        return state, {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作
        
        Args:
            action: 关节力矩 [τ₁, τ₂, ..., τ₇, gripper] (8维)
            
        Returns:
            Tuple[np.ndarray, float, bool, bool, Dict]: 状态、奖励、终止、截断、信息
        """
        # 应用关节动作（前7维）
        self.data.ctrl[:7] = action[:7]
        
        # 应用夹爪动作（第8维）
        if len(action) > 7:
            gripper_action = action[7]
            # 夹爪控制：正值关闭，负值打开
            self._set_gripper_action(gripper_action)
        
        # 步进仿真
        mujoco.mj_step(self.model, self.data)
        
        # 获取新状态
        state = self._get_state()
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查是否终止
        done = self._is_done()
        
        # 检查是否截断
        truncated = self.current_step >= self.max_steps
        
        # 更新步数
        self.current_step += 1
        
        return state, reward, done, truncated, {}
    
    def _get_state(self) -> np.ndarray:
        """获取当前状态"""
        # 关节角度 (7维)
        joint_positions = self.data.qpos[:7].copy()
        
        # 关节速度 (7维)
        joint_velocities = self.data.qvel[:7].copy()
        
        # 末端执行器位置 (3维)
        end_effector_pos = self._get_end_effector_position()
        
        # 接触力 (3维)
        contact_force = self._get_contact_force()
        
        # 目标位置 (3维)
        target_pos = self.target_position.copy()
        
        # 抓取相关状态 (4维)
        # 获取夹爪位置 - 查找finger_joint1
        gripper_position = 0.0
        try:
            finger_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
            if finger_joint_id >= 0 and finger_joint_id < len(self.data.qpos):
                gripper_position = self.data.qpos[finger_joint_id]
        except:
            # 如果找不到夹爪关节，使用默认值
            pass
        
        grasp_states = np.array([
            float(self.is_grasped),  # 是否已抓取
            float(self.grasp_target_reached),  # 是否到达抓取目标
            float(self.task_type == "grasp_and_move"),  # 是否为抓取任务
            gripper_position  # 夹爪位置
        ])
        
        # 组合状态
        state = np.concatenate([
            joint_positions,      # 7维
            target_pos,           # 3维
            end_effector_pos,     # 3维
            joint_velocities,     # 7维
            contact_force,        # 3维
            grasp_states          # 4维
        ])
        
        return state.astype(np.float32)
    
    def _get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        # 使用Panda模型的标准命名：hand是夹爪主体
        end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        position = self.data.xpos[end_effector_id].copy()
        return position
    
    def _get_contact_force(self) -> np.ndarray:
        """获取接触力"""
        # 简化实现，实际应该从力传感器读取
        # 这里暂时返回零向量
        return np.zeros(3)
    
    def _get_gripper_action(self) -> float:
        """获取夹爪控制动作"""
        try:
            # 查找actuator8（夹爪执行器）
            actuator8_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
            if actuator8_id >= 0 and actuator8_id < len(self.data.ctrl):
                return self.data.ctrl[actuator8_id]
            else:
                return 0.0
        except:
            return 0.0
    
    def _set_gripper_action(self, action: float):
        """设置夹爪控制动作"""
        try:
            # 查找actuator8（夹爪执行器）
            actuator8_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
            if actuator8_id >= 0 and actuator8_id < len(self.data.ctrl):
                # 将0-0.04的夹爪动作映射到0-255的控制范围
                gripper_control = int(action * 255 / 0.04)
                self.data.ctrl[actuator8_id] = gripper_control
        except:
            # 如果找不到夹爪执行器，忽略
            pass
    
    def _compute_reward(self) -> float:
        """计算奖励函数"""
        # 获取当前状态
        end_effector_pos = self._get_end_effector_position()
        joint_velocities = self.data.qvel[:7]
        
        # 根据任务类型计算奖励
        if self.task_type == "move_to":
            # 移动任务奖励
            position_error = np.linalg.norm(end_effector_pos - self.target_position)
            position_reward = -0.1 * position_error
            
            # 运动平滑性奖励
            smoothness_reward = -0.01 * np.linalg.norm(joint_velocities)
            
            # 任务完成奖励
            completion_reward = 50.0 if position_error < 0.05 else 0.0
            
            # 接近奖励
            approach_reward = 5.0 if position_error < 0.1 else 0.0
            
            total_reward = position_reward + smoothness_reward + completion_reward + approach_reward
            
        elif self.task_type == "grasp_and_move":
            # 抓取任务奖励
            if not self.is_grasped:
                # 阶段1：移动到抓取位置
                pickup_error = np.linalg.norm(end_effector_pos - self.pickup_position)
                position_reward = -0.1 * pickup_error
                
                # 到达抓取位置奖励
                if pickup_error < 0.05:
                    self.grasp_target_reached = True
                    approach_reward = 20.0
                    
                    # 检查夹爪动作 - 模拟抓取
                    gripper_action = self._get_gripper_action()
                    if abs(gripper_action) > 0.5:  # 夹爪有动作
                        # 模拟抓取成功
                        self.is_grasped = True
                        grasp_reward = 50.0  # 抓取成功奖励
                        logger.info("抓取成功！")
                    else:
                        grasp_reward = 0.0
                else:
                    approach_reward = 5.0 if pickup_error < 0.1 else 0.0
                    grasp_reward = 0.0
                
                # 运动平滑性奖励
                smoothness_reward = -0.01 * np.linalg.norm(joint_velocities)
                
                total_reward = position_reward + smoothness_reward + approach_reward + grasp_reward
                
            else:
                # 阶段2：移动到放置位置
                place_error = np.linalg.norm(end_effector_pos - self.place_position)
                position_reward = -0.1 * place_error
                
                # 运动平滑性奖励
                smoothness_reward = -0.01 * np.linalg.norm(joint_velocities)
                
                # 任务完成奖励
                completion_reward = 100.0 if place_error < 0.05 else 0.0
                
                # 接近奖励
                approach_reward = 10.0 if place_error < 0.1 else 0.0
                
                # 放置奖励 - 检查夹爪释放
                gripper_action = self._get_gripper_action()
                if place_error < 0.05 and abs(gripper_action) < 0.1:  # 到达位置且夹爪释放
                    place_reward = 30.0  # 放置成功奖励
                    logger.info("放置成功！")
                else:
                    place_reward = 0.0
                
                total_reward = position_reward + smoothness_reward + completion_reward + approach_reward + place_reward
        else:
            # 默认奖励
            position_error = np.linalg.norm(end_effector_pos - self.target_position)
            position_reward = -0.1 * position_error
            smoothness_reward = -0.01 * np.linalg.norm(joint_velocities)
            completion_reward = 50.0 if position_error < 0.05 else 0.0
            total_reward = position_reward + smoothness_reward + completion_reward
        
        return total_reward
    
    def _is_done(self) -> bool:
        """检查是否完成"""
        end_effector_pos = self._get_end_effector_position()
        
        if self.task_type == "move_to":
            # 移动任务：到达目标位置
            position_error = np.linalg.norm(end_effector_pos - self.target_position)
            return position_error < 0.05
            
        elif self.task_type == "grasp_and_move":
            # 抓取任务：完成抓取并到达放置位置
            if not self.is_grasped:
                # 阶段1：检查是否到达抓取位置并执行抓取
                pickup_error = np.linalg.norm(end_effector_pos - self.pickup_position)
                if pickup_error < 0.05:
                    # 检查夹爪动作
                    gripper_action = self._get_gripper_action()
                    if abs(gripper_action) > 0.5:  # 夹爪有动作，模拟抓取
                        self.is_grasped = True
                        # 更新目标位置为放置位置
                        self.target_position = self.place_position
                        logger.info("抓取完成，开始移动到放置位置")
                        return False
                    return False
                return False
            else:
                # 阶段2：检查是否到达放置位置并释放物体
                place_error = np.linalg.norm(end_effector_pos - self.place_position)
                if place_error < 0.05:
                    # 检查夹爪是否释放
                    gripper_action = self._get_gripper_action()
                    if abs(gripper_action) < 0.1:  # 夹爪释放
                        logger.info("放置完成，任务结束")
                        return True
                    return False
                return False
        else:
            # 默认：到达目标位置
            position_error = np.linalg.norm(end_effector_pos - self.target_position)
            return position_error < 0.05
    
    def _check_collision(self) -> bool:
        """检查碰撞"""
        # 简化实现，实际应该检查接触点
        return False
    
    def _randomize_initial_position(self):
        """随机初始化机械臂位置"""
        # 设置安全的初始位置
        safe_positions = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        self.data.qpos[:7] = safe_positions + np.random.normal(0, 0.1, 7)
    
    def set_target_position(self, target: np.ndarray):
        """设置目标位置"""
        self.target_position = target.copy()
    
    def set_task_config(self, task_type: str, pickup_position: np.ndarray = None, place_position: np.ndarray = None):
        """设置任务配置"""
        self.task_type = task_type
        if pickup_position is not None:
            self.pickup_position = np.array(pickup_position)
        if place_position is not None:
            self.place_position = np.array(place_position)
        
        # 重置抓取状态
        self.is_grasped = False
        self.grasp_target_reached = False
        
        # 根据任务类型设置初始目标
        if task_type == "move_to":
            self.target_position = self.pickup_position.copy()
        elif task_type == "grasp_and_move":
            self.target_position = self.pickup_position.copy()
    
    def reset_grasp_state(self):
        """重置抓取状态"""
        self.is_grasped = False
        self.grasp_target_reached = False
    
    def get_camera_image(self) -> np.ndarray:
        """获取相机图像 - 先尝试MuJoCo渲染，失败时使用模拟图像"""
        # 首先尝试MuJoCo渲染
        if self.camera_id is not None:
            try:
                # 渲染图像
                width, height = 640, 480
                
                # 使用MuJoCo渲染器
                mujoco.mj_forward(self.model, self.data)
                
                # 创建渲染上下文（使用with语句确保正确释放）
                with mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150) as context:
                    # 创建视口
                    viewport = mujoco.MjrRect(0, 0, width, height)
                    
                    # 创建图像缓冲区
                    image = np.empty((height, width, 3), dtype=np.uint8)
                    
                    # 渲染图像
                    mujoco.mjr_render(viewport, self.model, context, self.camera_id, mujoco.mjtCatBit.mjCAT_ALL)
                    
                    # 读取像素数据
                    mujoco.mjr_readPixels(image, None, viewport, self.model, context)
                    
                    logger.info("MuJoCo渲染成功")
                    return image
                    
            except Exception as e:
                logger.error(f"MuJoCo渲染失败: {e}")
                # 继续使用模拟图像
        
        # 使用模拟图像作为备选方案
        try:
            # 创建模拟图像
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 定义多个物体的位置（世界坐标）
            objects = [
                {"type": "red_cube", "world_pos": [0.5, 0.0, 0.3], "color": [255, 0, 0]},
                {"type": "blue_circle", "world_pos": [0.7, 0.2, 0.3], "color": [0, 0, 255]},
                {"type": "green_block", "world_pos": [0.3, -0.2, 0.3], "color": [0, 255, 0]},
                {"type": "yellow_sphere", "world_pos": [0.6, -0.1, 0.3], "color": [255, 255, 0]}
            ]
            
            # 绘制所有物体
            for obj in objects:
                world_pos = obj["world_pos"]
                color = obj["color"]
                
                # 将世界坐标转换为像素坐标
                pixel_x = int(100 + (world_pos[0] - 0.2) * 440 / 0.6)
                pixel_y = int(100 + (world_pos[1] + 0.3) * 280 / 0.6)
                
                # 确保像素坐标在图像范围内
                pixel_x = max(50, min(590, pixel_x))
                pixel_y = max(50, min(430, pixel_y))
                
                # 绘制物体
                size = 25
                image[pixel_y-size:pixel_y+size, pixel_x-size:pixel_x+size] = color
            
            # 绘制机器人基座（灰色矩形）
            image[400:450, 250:390] = [128, 128, 128]
            
            # 添加一些噪声使图像更真实
            noise = np.random.randint(0, 15, image.shape, dtype=np.uint8)
            image = np.clip(image + noise, 0, 255)
            
            logger.info("使用模拟图像")
            return image
            
        except Exception as e:
            logger.error(f"模拟图像生成失败: {e}")
            # 返回默认图像
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def render(self):
        """渲染环境"""
        # 这里可以添加可视化代码
        pass
    
    def close(self):
        """关闭环境"""
        pass


