"""
强化学习抓取环境
基于MuJoCo的Panda机械臂抓取任务环境
包含肌腱控制
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging
import os
import time

from config import GraspingConfig, RewardConfig
from singularity_handler import SingularityHandler
from action_wrapper import SafeActionWrapper
from state import get_proprioceptive_state
from reward import calculate_reward

class PandaGraspingEnv(gym.Env):
    """
    Panda机械臂抓取环境
    
    任务: 控制机械臂到达并抓取指定位置的物体
    观察空间: 本体感知状态 + 肌腱状态
    动作空间: 关节角度 + 肌腱控制
    """
    
    def __init__(self, grasping_config: GraspingConfig, reward_config: RewardConfig):
        super().__init__()
        
        self.grasping_config = grasping_config
        self.reward_config = reward_config
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 检查XML文件是否存在
        if not os.path.exists(grasping_config.xml_path):
            raise FileNotFoundError(f"MuJoCo XML文件不存在: {grasping_config.xml_path}")
        
        # 初始化MuJoCo模型
        self._init_mujoco()
        
        # 设置观察空间和动作空间
        self._setup_spaces()
        
        # 初始化奇异点处理器
        self.singularity_handler = SingularityHandler()
        
        # 初始化安全动作包装器
        self.action_wrapper = SafeActionWrapper(self)
        
        # 目标信息
        self.target_pos = np.array([0.5, 0.0, 0.3])
        self.target_quat = np.array([1, 0, 0, 0])
        
        # 任务状态
        self.task_state = {
            'object_position': None,
            'is_grasped': False,
            'grasp_success': False,
            'episode_steps': 0,
            'previous_joint_pos': None,
            'was_singular': False
        }
        
        # 前一步状态（用于奖励计算）
        self.prev_state = None
        
        # 初始化训练监控器
        self.training_monitor = None
        self.episode_start_time = None
        self.episode_singularity_count = 0
        
        # 重置警告控制
        self.singularity_handler.reset_warning_control()
        
        # 重置环境
        self.reset()
    
    def _init_mujoco(self):
        """初始化MuJoCo环境"""
        try:
            # 设置无头模式环境变量
            os.environ['MUJOCO_GL'] = 'egl'  # 使用EGL而不是OpenGL
            os.environ['DISPLAY'] = ':0'     # 设置虚拟显示
            
            # 加载模型
            self.model = mujoco.MjModel.from_xml_path(self.grasping_config.xml_path)
            self.data = mujoco.MjData(self.model)
            
            # 查找关键组件
            self._find_components()
            
            # 尝试创建渲染器，如果失败则使用无头模式
            try:
                self.renderer = mujoco.Renderer(self.model)
                self.headless = False
                print("✅ 图形渲染模式已启用")
            except Exception as e:
                print(f"⚠️  图形渲染失败，使用无头模式: {e}")
                self.renderer = None
                self.headless = True
                
        except Exception as e:
            print(f"❌ MuJoCo初始化失败: {e}")
            raise
    
    def _find_components(self):
        """查找关键组件"""
        try:
            # 查找机械臂关节
            self.arm_joint_names = []
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and 'panda' in joint_name.lower():
                    self.arm_joint_names.append(joint_name)
            
            # 查找夹爪关节
            self.gripper_joint_names = []
            self.gripper_joint_ids = []  # 添加这个属性
            for i in range(self.model.njnt):
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                if joint_name and ('finger' in joint_name.lower() or 'gripper' in joint_name.lower()):
                    self.gripper_joint_names.append(joint_name)
                    self.gripper_joint_ids.append(i)  # 记录关节ID
            
            # 获取所有body名称，用于调试
            all_body_names = []
            for i in range(self.model.nbody):
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    all_body_names.append((i, body_name))
            
            self.logger.info(f"模型中的所有body: {[name for _, name in all_body_names]}")
            
            # 查找目标物体 - 更加灵活的方法
            self.target_body_id = -1
            target_candidates = [self.grasping_config.target_object] + ["red_cube", "blue_cube", "green_cube", "target_cube", "cube"]
            
            for target_name in target_candidates:
                self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
                if self.target_body_id != -1:
                    self.logger.info(f"找到目标物体: {target_name} (ID: {self.target_body_id})")
                    break
            
            if self.target_body_id == -1:
                # 如果还是找不到，尝试查找包含关键词的body
                for i, name in all_body_names:
                    if any(keyword in name.lower() for keyword in ['cube', 'box', 'target', 'object', 'ball']):
                        self.target_body_id = i
                        self.logger.info(f"通过关键词找到目标物体: {name} (ID: {i})")
                        break
                
                if self.target_body_id == -1:
                    self.logger.warning(f"找不到目标物体，使用默认")
                    self.target_body_id = None
            
            # 查找末端执行器 - 更加灵活的方法
            self.end_effector_id = -1
            end_effector_candidates = ["panda_hand", "panda_gripper", "end_effector", "gripper", "hand", "gripper_hand", "ee", "end_effector_link"]
            
            for name in end_effector_candidates:
                self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if self.end_effector_id != -1:
                    self.logger.info(f"找到末端执行器: {name} (ID: {self.end_effector_id})")
                    break
            
            if self.end_effector_id == -1:
                # 如果还是找不到，尝试查找包含关键词的body
                for i, name in all_body_names:
                    if any(keyword in name.lower() for keyword in ['hand', 'gripper', 'ee', 'end', 'finger']):
                        self.end_effector_id = i
                        self.logger.info(f"通过关键词找到末端执行器: {name} (ID: {i})")
                        break
            
            if self.end_effector_id == -1:
                # 如果还是找不到，使用最后一个body作为末端执行器
                if all_body_names:
                    self.end_effector_id = all_body_names[-1][0]
                    self.logger.warning(f"找不到末端执行器，使用最后一个body: {all_body_names[-1][1]} (ID: {self.end_effector_id})")
                else:
                    raise ValueError("模型中没有找到任何body")
            
            # 查找物体ID（用于设置物体位置）
            self.object_id = self.target_body_id  # 使用相同的ID
            
            self.logger.info(f"找到 {len(self.arm_joint_names)} 个机械臂关节")
            self.logger.info(f"找到 {len(self.gripper_joint_names)} 个夹爪关节")
            self.logger.info(f"目标物体ID: {self.target_body_id}")
            self.logger.info(f"末端执行器ID: {self.end_effector_id}")
            self.logger.info(f"物体ID: {self.object_id}")
            
        except Exception as e:
            self.logger.error(f"组件查找失败: {e}")
            raise
    
    def _setup_spaces(self):
        """设置观察空间和动作空间"""
        # 动作空间: 7个关节角度 + 1个肌腱控制
        self.action_space = spaces.Box(
            low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0]),
            high=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0]),
            dtype=np.float32
        )
        
        # 观察空间: 本体感知状态 + 肌腱状态
        # 7个关节位置 + 7个关节速度 + 7个关节力矩 + 1个肌腱位置 + 1个肌腱速度 + 1个肌腱张力 +
        # 3个末端位置 + 4个末端方向 + 6个末端速度 + 1个夹爪状态 + 3个目标位置 + 4个目标方向 +
        # 1个可操作度 + 1个接触力 + 6个手指力
        obs_dim = 7 + 7 + 7 + 1 + 1 + 1 + 3 + 4 + 6 + 1 + 3 + 4 + 1 + 1 + 6  # 52维
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 使用安全的初始配置
        joint_positions = self.singularity_handler.generate_safe_initial_config()
        self.data.qpos[:7] = joint_positions
        
        # 初始化肌腱位置
        if len(self.data.qpos) > 7:
            self.data.qpos[7] = 0.02  # 肌腱初始位置
        
        # 初始化夹爪位置
        if hasattr(self, 'gripper_joint_ids') and self.gripper_joint_ids:
            for joint_id in self.gripper_joint_ids:
                if joint_id < len(self.data.qpos):
                    self.data.qpos[joint_id] = 0.02  # 半开状态
        
        # 设置物体位置（固定位置）
        self._set_object_position()
        
        # 重置episode监控
        self.episode_start_time = time.time()
        self.episode_singularity_count = 0
        
        # 重置任务状态
        self.task_state = {
            'object_position': self._get_object_position(),
            'is_grasped': False,
            'grasp_success': False,
            'episode_steps': 0,
            'previous_joint_pos': joint_positions.copy()
        }
        
        # 重置前一步状态
        self.prev_state = None
        
        # 前向动力学
        mujoco.mj_forward(self.model, self.data)
        
        # 获取初始观察
        observation = self._get_observation()
        info = self._get_info()
        
        self.logger.debug(f"环境重置完成，物体位置: {self.task_state['object_position']}")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """执行一步动作"""
        # 获取当前状态信息
        current_state = get_proprioceptive_state(self.data, self.model, self)
        
        # 应用安全动作包装器
        safe_action = self.action_wrapper.apply(action, current_state)
        
        # 设置关节控制
        self.data.ctrl[:7] = safe_action['joint_commands']
        
        # 设置肌腱控制
        if len(self.data.ctrl) > 7:
            # 将肌腱命令映射到控制范围
            tendon_control = int(safe_action['tendon_command'] * 255)
            self.data.ctrl[7] = tendon_control
        
        # 检查当前关节位置是否奇异（使用警告控制）
        current_joint_pos = self.data.qpos[:7].copy()
        current_time = time.time()
        is_singular, singularity_type, singularity_score, should_warn = self.singularity_handler.detect_singularity_with_warning_control(current_joint_pos, current_time)
        
        if is_singular:
            safe_joint_action = self.singularity_handler.get_safe_config(current_joint_pos)
            self.data.ctrl[:7] = safe_joint_action
            self.episode_singularity_count += 1  # 记录奇异点次数
            if should_warn:
                self.logger.warning(f"检测到奇异点 {singularity_type} (程度: {singularity_score:.3f})，使用渐进安全配置")
        
        # 模拟一步
        mujoco.mj_step(self.model, self.data)
        
        # 更新任务状态
        self.task_state['episode_steps'] += 1
        self.task_state['object_position'] = self._get_object_position()
        
        # 检查奇异点恢复
        if self.task_state['previous_joint_pos'] is not None:
            recovered = self.singularity_handler.check_singularity_recovery(
                self.data.qpos[:7], self.task_state['previous_joint_pos']
            )
            if recovered:
                # 只在调试模式下输出恢复信息
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("从奇异点恢复")
        
        # 更新前一步关节位置
        self.task_state['previous_joint_pos'] = self.data.qpos[:7].copy()
        
        # 检查抓取状态
        self._update_grasp_state()
        
        # 获取新的状态
        new_state = get_proprioceptive_state(self.data, self.model, self)
        
        # 计算奖励
        reward = calculate_reward(new_state, self.prev_state, safe_action)
        
        # 更新前一步状态
        self.prev_state = current_state
        
        # 获取观察和信息
        observation = self._get_observation()
        terminated = self._is_done()
        truncated = self.task_state['episode_steps'] >= self.grasping_config.max_steps
        info = self._get_info()
        
        # 如果episode结束且有监控器，记录episode信息
        if (terminated or truncated) and self.training_monitor is not None:
            episode_time = time.time() - self.episode_start_time
            self.training_monitor.log_episode(
                episode=self.task_state.get('episode_number', 0),
                reward=reward,  # 使用实际计算的奖励
                length=self.task_state['episode_steps'],
                success=self.task_state['is_grasped'],
                singularity_count=self.episode_singularity_count,
                episode_time=episode_time
            )
        
        return observation, reward, terminated, truncated, info
    
    def _set_object_position(self):
        """设置物体位置（固定位置）"""
        # 使用XML文件中定义的固定位置，确保一致性
        # XML中定义的位置: pos="0.5 0.0 0.1"
        object_pos = np.array([0.5, 0.0, 0.1])
        
        # 设置物体位置
        if self.object_id is not None:
            # 找到物体的自由关节
            for i in range(self.model.njnt):
                if self.model.jnt_bodyid[i] == self.object_id:
                    self.data.qpos[i] = object_pos
                    break
        
        self.logger.info(f"设置目标物体位置: {object_pos}")
    
    def _get_object_position(self) -> np.ndarray:
        """获取物体位置"""
        if self.object_id is not None:
            return self.data.xpos[self.object_id].copy()
        else:
            # 如果没有找到物体，返回XML中定义的固定位置
            return np.array([0.5, 0.0, 0.1])
    
    def _update_grasp_state(self):
        """更新抓取状态"""
        end_effector_pos = self._get_end_effector_position()
        object_pos = self._get_object_position()
        
        # 检查是否接近物体
        distance_to_object = np.linalg.norm(end_effector_pos - object_pos)
        
        # 检查夹爪状态
        gripper_width = 0.0
        if hasattr(self, 'gripper_joint_ids') and self.gripper_joint_ids:
            for joint_id in self.gripper_joint_ids:
                if joint_id < len(self.data.qpos):
                    gripper_width += self.data.qpos[joint_id]
        
        # 判断是否抓取成功
        if (distance_to_object < self.grasping_config.grasp_distance_threshold and 
            gripper_width < self.grasping_config.gripper_closed_threshold):
            self.task_state['is_grasped'] = True
            
            # 检查抓取是否稳定
            if distance_to_object < self.grasping_config.grasp_success_threshold:
                self.task_state['grasp_success'] = True
    
    def _get_observation(self) -> np.ndarray:
        """获取观察"""
        # 获取本体感知状态
        state = get_proprioceptive_state(self.data, self.model, self)
        
        # 组合观察向量
        observation = np.concatenate([
            state['joint_positions'],      # 7: 关节位置
            state['joint_velocities'],     # 7: 关节速度
            state['joint_torques'],        # 7: 关节力矩
            [state['tendon_position']],    # 1: 肌腱位置
            [state['tendon_velocity']],    # 1: 肌腱速度
            [state['tendon_tension']],     # 1: 肌腱张力
            state['ee_position'],          # 3: 末端位置
            state['ee_orientation'],       # 4: 末端方向
            state['ee_velocity'],          # 6: 末端速度
            [state['gripper_state']],      # 1: 夹爪状态
            state['target_position'],      # 3: 目标位置
            state['target_orientation'],   # 4: 目标方向
            [state['manipulability']],     # 1: 可操作度
            [state['contact_force']],      # 1: 接触力
            state['left_finger_force'],    # 3: 左手指力
            state['right_finger_force']    # 3: 右手指力
        ])
        
        return observation.astype(np.float32)
    
    def _get_end_effector_position(self) -> np.ndarray:
        """获取末端执行器位置"""
        if self.end_effector_id is not None:
            # 确保前向动力学已经计算
            if len(self.data.xpos) > self.end_effector_id:
                return self.data.xpos[self.end_effector_id].copy()
            else:
                # 如果xpos数组还没有计算，返回机械臂前方的默认位置
                return np.array([0.4, 0.0, 0.2])
        else:
            # 如果没有找到末端执行器，使用最后一个关节的位置
            if len(self.data.xpos) > 0:
                return self.data.xpos[-1].copy()
            else:
                # 如果xpos为空，返回机械臂前方的默认位置
                return np.array([0.4, 0.0, 0.2])
    
    def _is_done(self) -> bool:
        """判断episode是否结束"""
        # 检查是否成功抓取
        if self.task_state['is_grasped']:
            return True
        
        # 检查是否超时
        if self.task_state['episode_steps'] >= self.grasping_config.max_steps:
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """获取信息"""
        end_effector_pos = self._get_end_effector_position()
        object_pos = self._get_object_position()
        
        info = {
            'episode_steps': self.task_state['episode_steps'],
            'is_grasped': self.task_state['is_grasped'],
            'grasp_success': self.task_state['grasp_success'],
            'end_effector_pos': end_effector_pos.copy(),
            'object_pos': object_pos.copy(),
            'distance_to_object': np.linalg.norm(end_effector_pos - object_pos),
            'gripper_width': sum(self.data.qpos[joint_id] for joint_id in self.gripper_joint_ids 
                                if hasattr(self, 'gripper_joint_ids') and joint_id < len(self.data.qpos)) if hasattr(self, 'gripper_joint_ids') and self.gripper_joint_ids else 0.0
        }
        
        return info
    
    def render(self, mode='human'):
        """渲染环境"""
        if self.headless or self.renderer is None:
            # 无头模式，跳过渲染
            return None
        
        try:
            # 更新渲染器
            self.renderer.update_scene(self.data)
            image = self.renderer.render()
            return image
        except Exception as e:
            self.logger.warning(f"渲染失败: {e}")
            return None
    
    def close(self):
        """关闭环境"""
        pass
