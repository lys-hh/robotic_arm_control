"""
VLM+RL系统集成模块
"""

import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from .vlm_module import VLMProcessor
from .rl_module import RLAgent, PandaEnvironment

logger = logging.getLogger(__name__)

class VLMRLSystem:
    """
    VLM+RL集成系统
    
    功能:
    - 整合VLM和RL模块
    - 提供统一的控制接口
    - 支持实时指令执行
    """
    
    def __init__(self, 
                 robot_type: str = "panda",
                 xml_path: str = None,
                 vlm_model: str = "Qwen/Qwen-VL-Chat-1.8B",
                 rl_algorithm: str = "ppo"):
        """
        初始化VLM+RL系统
        
        Args:
            robot_type: 机器人类型 ("panda" 或 "ur5e")
            xml_path: MuJoCo XML文件路径
            vlm_model: VLM模型名称
            rl_algorithm: RL算法 ("ppo", "td3", "sac")
        """
        self.robot_type = robot_type
        self.xml_path = xml_path or self._get_default_xml_path(robot_type)
        self.vlm_model = vlm_model
        self.rl_algorithm = rl_algorithm
        
        # 初始化组件
        self.vlm_processor = None
        self.rl_agent = None
        self.environment = None
        
        # 性能监控
        self.performance_stats = {
            "vlm_latency": [],
            "rl_latency": [],
            "total_latency": [],
            "success_rate": 0.0,
            "total_instructions": 0
        }
        
        # 初始化系统
        self._initialize_system()
        
        logger.info(f"VLM+RL系统初始化完成: {robot_type} + {vlm_model} + {rl_algorithm}")
    
    def _get_default_xml_path(self, robot_type: str) -> str:
        """获取默认XML文件路径"""
        if robot_type.lower() == "panda":
            return "../models/franka_emika_panda/panda.xml"
        elif robot_type.lower() == "ur5e":
            return "../models/universal_robots_ur5e/ur5e.xml"
        else:
            raise ValueError(f"不支持的机器人类型: {robot_type}")
    
    def _initialize_system(self):
        """初始化系统组件"""
        try:
            # 1. 初始化VLM处理器
            logger.info("正在初始化VLM处理器...")
            self.vlm_processor = VLMProcessor(
                model_name=self.vlm_model,
                device="auto"
            )
            
            # 2. 初始化RL环境
            logger.info("正在初始化RL环境...")
            if self.robot_type.lower() == "panda":
                self.environment = PandaEnvironment(self.xml_path)
           
            # 3. 初始化RL智能体
            logger.info("正在初始化RL智能体...")
            self.rl_agent = self._create_rl_agent()
            
            logger.info("系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"系统初始化失败: {e}")
            raise
    
    def _create_rl_agent(self) -> RLAgent:
        """创建RL智能体"""
        # 这里应该根据选择的算法创建相应的智能体
        # 暂时返回一个占位符
        from .rl_module.rl_agent import PPOPandaAgent
        
        if self.robot_type.lower() == "panda":
            return PPOPandaAgent(
                input_dim=23,  # 状态空间维度
                output_dim=7,  # 动作空间维度
                hidden_dim=256
            )

    
    def execute_instruction(self, instruction: str, max_steps: int = 1000) -> Dict:
        """
        执行自然语言指令
        
        Args:
            instruction: 自然语言指令
            max_steps: 最大执行步数
            
        Returns:
            Dict: 执行结果
        """
        start_time = time.time()
        
        try:
            # 1. VLM处理阶段
            vlm_start = time.time()
            image = self.environment.get_camera_image()
            target_info = self.vlm_processor.process_instruction(image, instruction)
            vlm_latency = time.time() - vlm_start
            
            logger.info(f"VLM处理结果: {target_info}")
            
            # 2. 设置目标位置
            target_position = np.array(target_info["position"])
            self.environment.set_target_position(target_position)
            
            # 3. RL控制阶段
            rl_start = time.time()
            success = self._execute_rl_control(max_steps)
            rl_latency = time.time() - rl_start
            
            # 4. 记录性能统计
            total_latency = time.time() - start_time
            self._update_performance_stats(vlm_latency, rl_latency, total_latency, success)
            
            # 5. 返回结果
            result = {
                "success": success,
                "target_info": target_info,
                "performance": {
                    "vlm_latency": vlm_latency,
                    "rl_latency": rl_latency,
                    "total_latency": total_latency
                },
                "final_position": self.environment._get_end_effector_position().tolist()
            }
            
            logger.info(f"指令执行完成: {instruction} -> {'成功' if success else '失败'}")
            return result
            
        except Exception as e:
            logger.error(f"指令执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "performance": {
                    "vlm_latency": 0.0,
                    "rl_latency": 0.0,
                    "total_latency": time.time() - start_time
                }
            }
    
    def _execute_rl_control(self, max_steps: int) -> bool:
        """执行RL控制"""
        # 重置环境
        state, _ = self.environment.reset()
        
        for step in range(max_steps):
            # 获取动作
            action = self.rl_agent.get_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, _ = self.environment.step(action)
            
            # 更新状态
            state = next_state
            
            # 检查是否完成
            if done:
                return True
            
            # 检查是否超时
            if truncated:
                return False
        
        return False
    
    def _update_performance_stats(self, vlm_latency: float, rl_latency: float, 
                                 total_latency: float, success: bool):
        """更新性能统计"""
        self.performance_stats["vlm_latency"].append(vlm_latency)
        self.performance_stats["rl_latency"].append(rl_latency)
        self.performance_stats["total_latency"].append(total_latency)
        
        self.performance_stats["total_instructions"] += 1
        if success:
            self.performance_stats["success_rate"] = (
                self.performance_stats["success_rate"] * (self.performance_stats["total_instructions"] - 1) + 1
            ) / self.performance_stats["total_instructions"]
    
    def batch_execute(self, instructions: List[str]) -> List[Dict]:
        """
        批量执行指令
        
        Args:
            instructions: 指令列表
            
        Returns:
            List[Dict]: 执行结果列表
        """
        results = []
        
        for i, instruction in enumerate(instructions):
            logger.info(f"执行指令 {i+1}/{len(instructions)}: {instruction}")
            result = self.execute_instruction(instruction)
            results.append(result)
            
            # 添加延迟避免过快执行
            time.sleep(0.5)
        
        return results
    
    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        if not self.performance_stats["total_latency"]:
            return {"message": "暂无性能数据"}
        
        return {
            "total_instructions": self.performance_stats["total_instructions"],
            "success_rate": self.performance_stats["success_rate"],
            "average_latencies": {
                "vlm": np.mean(self.performance_stats["vlm_latency"]),
                "rl": np.mean(self.performance_stats["rl_latency"]),
                "total": np.mean(self.performance_stats["total_latency"])
            },
            "latency_std": {
                "vlm": np.std(self.performance_stats["vlm_latency"]),
                "rl": np.std(self.performance_stats["rl_latency"]),
                "total": np.std(self.performance_stats["total_latency"])
            }
        }
    
    def plot_performance(self, save_path: Optional[str] = None):
        """绘制性能图表"""
        if not self.performance_stats["total_latency"]:
            logger.warning("暂无性能数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 延迟分布
        axes[0, 0].hist(self.performance_stats["vlm_latency"], bins=20, alpha=0.7, label="VLM")
        axes[0, 0].hist(self.performance_stats["rl_latency"], bins=20, alpha=0.7, label="RL")
        axes[0, 0].set_xlabel("延迟 (秒)")
        axes[0, 0].set_ylabel("频次")
        axes[0, 0].set_title("延迟分布")
        axes[0, 0].legend()
        
        # 总延迟趋势
        axes[0, 1].plot(self.performance_stats["total_latency"])
        axes[0, 1].set_xlabel("指令序号")
        axes[0, 1].set_ylabel("总延迟 (秒)")
        axes[0, 1].set_title("总延迟趋势")
        
        # 成功率
        axes[1, 0].bar(["成功率"], [self.performance_stats["success_rate"]])
        axes[1, 0].set_ylabel("成功率")
        axes[1, 0].set_title("任务成功率")
        axes[1, 0].set_ylim(0, 1)
        
        # 平均延迟对比
        avg_latencies = [
            np.mean(self.performance_stats["vlm_latency"]),
            np.mean(self.performance_stats["rl_latency"])
        ]
        axes[1, 1].bar(["VLM", "RL"], avg_latencies)
        axes[1, 1].set_ylabel("平均延迟 (秒)")
        axes[1, 1].set_title("平均延迟对比")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能图表已保存到: {save_path}")
        
        plt.show()
    
    def save_system_state(self, filepath: str):
        """保存系统状态"""
        import pickle
        
        state = {
            "robot_type": self.robot_type,
            "vlm_model": self.vlm_model,
            "rl_algorithm": self.rl_algorithm,
            "performance_stats": self.performance_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"系统状态已保存到: {filepath}")
    
    def load_system_state(self, filepath: str):
        """加载系统状态"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.performance_stats = state["performance_stats"]
        logger.info(f"系统状态已从 {filepath} 加载")
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            "robot_type": self.robot_type,
            "vlm_model": self.vlm_model,
            "rl_algorithm": self.rl_algorithm,
            "vlm_info": self.vlm_processor.get_model_info() if self.vlm_processor else None,
            "environment_info": {
                "observation_space": str(self.environment.observation_space) if self.environment else None,
                "action_space": str(self.environment.action_space) if self.environment else None
            }
        }
