#!/usr/bin/env python3
"""
VLM+RL联合训练脚本
结合视觉语言模型和强化学习的机械臂智能控制训练
"""

import os
import sys
import numpy as np
import torch
import logging
from typing import Dict, List, Tuple
import time

# 设置环境变量
cache_dir = "./vlm_models"
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VLMRLTrainer:
    """
    VLM+RL联合训练器
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化训练器
        
        Args:
            config: 训练配置
        """
        self.config = config or self._get_default_config()
        self.vlm_processor = None
        self.rl_agent = None
        self.environment = None
        
        logger.info("VLM+RL训练器初始化完成")
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            "vlm": {
                "model_name": "Qwen/Qwen-VL-Chat",
                "device": "cuda" if torch.cuda.is_available() else "cpu",  # 明确指定设备
                "cache_dir": cache_dir
            },
            "rl": {
                "algorithm": "PPO",
                "learning_rate": 1e-4,        # 降低学习率，更稳定
                "total_timesteps": 500000,    # 增加训练步数
                "batch_size": 256,            # 增加批次大小
                "n_steps": 2048,              # 保持不变
                "n_epochs": 8,                # 减少训练轮数
                "gamma": 0.99,                # 折扣因子
                "gae_lambda": 0.95,           # GAE参数
                "clip_range": 0.2             # PPO裁剪范围
            },
            "training": {
                "max_episodes": 2000,         # 增加episode数量
                "episode_length": 300,        # 增加episode长度
                "save_interval": 50           # 增加保存间隔
            }
        }
    
    def setup_components(self):
        """设置VLM和RL组件"""
        try:
            logger.info("正在设置VLM组件...")
            from vlm_module import VLMProcessor
            self.vlm_processor = VLMProcessor(
                model_name=self.config["vlm"]["model_name"],
                device=self.config["vlm"]["device"]
            )
            logger.info("✅ VLM组件设置完成")
            
            logger.info("正在设置RL组件...")
            from rl_module import RLAgent, PandaEnvironment
            self.environment = PandaEnvironment()
            self.rl_agent = RLAgent(
                input_dim=self.environment.observation_space.shape[0],
                output_dim=self.environment.action_space.shape[0],
                algorithm=self.config["rl"]["algorithm"]
            )
            self.rl_agent.set_environment(self.environment)
            logger.info("✅ RL组件设置完成")
            
        except Exception as e:
            logger.error(f"组件设置失败: {e}")
            raise
    
    def train_single_episode(self, episode: int) -> Dict:
        """
        训练单个episode
        
        Args:
            episode: episode编号
            
        Returns:
            Dict: episode结果
        """
        logger.info(f"开始训练Episode {episode}")
        
        # 重置环境
        obs, info = self.environment.reset()
        episode_reward = 0
        episode_steps = 0
        episode_success = False
        
        # 获取相机图像
        camera_image = self.environment.get_camera_image()
        
        # 随机选择指令
        import random
        instructions = [
            "移动到红色立方体上方",
            "移动到蓝色圆形上方", 
            "移动到绿色方块上方",
            "移动到黄色球体上方",
            "抓取红色立方体到右边",
            "抓取蓝色圆形到左边",
            "抓取绿色方块到上边",
            "抓取黄色球体到前面"
        ]
        instruction = random.choice(instructions)
        
        # 使用VLM处理图像和指令
        try:
            vlm_result = self.vlm_processor.process_instruction(camera_image, instruction)
            logger.info(f"VLM处理结果: {vlm_result}")
        except Exception as e:
            logger.error(f"VLM处理失败: {e}")
            # 不再使用备用方案，直接抛出异常以便诊断问题
            raise ValueError(f"VLM处理失败，需要诊断问题: {e}")
        
        # 根据任务类型设置目标位置
        if "task_type" in vlm_result:
            task_type = vlm_result["task_type"]
            pickup_position = vlm_result.get("pickup_position", [0.5, 0.0, 0.35])
            place_position = vlm_result.get("place_position", [0.5, 0.0, 0.35])
            
            # 设置任务配置
            self.environment.set_task_config(
                task_type=task_type,
                pickup_position=pickup_position,
                place_position=place_position
            )
            
            if task_type == "move_to":
                logger.info(f"VLM识别移动任务，目标位置: {pickup_position}")
            elif task_type == "grasp_and_move":
                logger.info(f"VLM识别抓取任务，抓取位置: {pickup_position}, 放置位置: {place_position}")
        else:
            # 兼容旧格式
            if "world_position" in vlm_result:
                target_position = vlm_result["world_position"]
            else:
                target_position = [0.5, 0.0, 0.35]
            
            # 设置为移动任务
            self.environment.set_task_config(
                task_type="move_to",
                pickup_position=target_position
            )
            logger.info(f"使用兼容模式，目标位置: {target_position}")
        
        # 执行episode
        for step in range(self.config["training"]["episode_length"]):
            # RL智能体选择动作
            action, _ = self.rl_agent.predict(obs, deterministic=False)
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.environment.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            # 检查是否成功
            if info.get("success", False):
                episode_success = True
                logger.info(f"Episode {episode} 在第{step}步成功完成")
                break
            
            if done:
                break
        
        episode_result = {
            "episode": episode,
            "reward": episode_reward,
            "steps": episode_steps,
            "success": episode_success,
            "vlm_result": vlm_result,
            "instruction": instruction
        }
        
        logger.info(f"Episode {episode} 完成: 指令='{instruction}', 奖励={episode_reward:.2f}, 步数={episode_steps}, 成功={episode_success}")
        return episode_result
    
    def train(self):
        """执行完整训练"""
        logger.info("开始VLM+RL联合训练")
        
        # 设置组件
        self.setup_components()
        
        # 训练统计
        training_stats = {
            "episodes": [],
            "total_rewards": [],
            "success_rate": [],
            "avg_steps": []
        }
        
        # 开始训练
        for episode in range(self.config["training"]["max_episodes"]):
            try:
                # 训练单个episode
                episode_result = self.train_single_episode(episode)
                
                # 记录统计
                training_stats["episodes"].append(episode)
                training_stats["total_rewards"].append(episode_result["reward"])
                training_stats["success_rate"].append(episode_result["success"])
                training_stats["avg_steps"].append(episode_result["steps"])
                
                # 定期保存模型
                if (episode + 1) % self.config["training"]["save_interval"] == 0:
                    self.save_models(episode + 1)
                    self.print_training_summary(training_stats)
                
                # 短暂休息
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Episode {episode} 训练失败: {e}")
                continue
        
        # 训练完成
        logger.info("VLM+RL联合训练完成")
        self.save_models("final")
        self.print_final_summary(training_stats)
    
    def save_models(self, suffix: str):
        """保存模型"""
        try:
            # 保存RL模型
            rl_model_path = f"models/rl_agent_{suffix}.zip"
            os.makedirs("models", exist_ok=True)
            self.rl_agent.save(rl_model_path)
            logger.info(f"RL模型已保存到: {rl_model_path}")
            
            # 保存训练配置
            config_path = f"models/training_config_{suffix}.json"
            import json
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"训练配置已保存到: {config_path}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {e}")
    
    def print_training_summary(self, stats: Dict):
        """打印训练摘要"""
        recent_episodes = stats["episodes"][-10:]  # 最近10个episode
        recent_rewards = stats["total_rewards"][-10:]
        recent_success = stats["success_rate"][-10:]
        
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_success) * 100
        
        logger.info(f"训练摘要 (最近10个episode):")
        logger.info(f"  平均奖励: {avg_reward:.2f}")
        logger.info(f"  成功率: {success_rate:.1f}%")
    
    def print_final_summary(self, stats: Dict):
        """打印最终训练摘要"""
        total_episodes = len(stats["episodes"])
        avg_reward = np.mean(stats["total_rewards"])
        success_rate = np.mean(stats["success_rate"]) * 100
        avg_steps = np.mean(stats["avg_steps"])
        
        logger.info("=" * 50)
        logger.info("VLM+RL联合训练最终摘要")
        logger.info("=" * 50)
        logger.info(f"总episode数: {total_episodes}")
        logger.info(f"平均奖励: {avg_reward:.2f}")
        logger.info(f"成功率: {success_rate:.1f}%")
        logger.info(f"平均步数: {avg_steps:.1f}")
        logger.info("=" * 50)

def main():
    """主函数"""
    print("=" * 60)
    print("VLM+RL联合训练系统")
    print("=" * 60)
    
    try:
        # 创建训练器
        trainer = VLMRLTrainer()
        
        # 开始训练
        trainer.train()
        
        print("\n🎉 训练完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
