"""
云端专用训练脚本
完全避免图形界面问题，适用于无头服务器环境
"""

import os
import sys
import logging
import numpy as np

# 设置环境变量，避免图形界面问题
os.environ['MUJOCO_GL'] = 'egl'
os.environ['DISPLAY'] = ':0'
os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__)) + ':' + os.environ.get('PYTHONPATH', '')

# 禁用matplotlib的图形界面
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from agent import GraspingAgent
from environment import PandaGraspingEnv
from config import get_config
from training_monitor import TrainingMonitor

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('cloud_training.log')
        ]
    )

def main():
    """主训练函数"""
    print("=" * 80)
    print("☁️  Panda机械臂强化学习训练系统 (云端版)")
    print("=" * 80)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 获取配置
        config = get_config()
        logger.info("配置加载完成")
        
        # 创建环境
        logger.info("创建训练环境...")
        env = PandaGraspingEnv(
            grasping_config=config.grasping,
            reward_config=config.reward
        )
        logger.info("环境创建成功")
        
        # 创建智能体
        logger.info("创建智能体...")
        agent = GraspingAgent(
            network_config=config.network,
            training_config=config.training
        )
        agent.set_environment(env)
        logger.info("智能体创建成功")
        
        # 开始训练
        logger.info("开始训练...")
        total_timesteps = config.training.total_timesteps
        
        print(f"\n🎯 训练目标: {total_timesteps:,} 总步数")
        print(f"📊 监控指标: 奖励、成功率、奇异点、训练趋势")
        print(f"📈 图表保存: logs/ 目录")
        print(f"📝 日志文件: cloud_training.log")
        print(f"☁️  运行模式: 云端无头模式")
        print(f"🔧 归一化技术: 自定义VecNormalize包装器 (观察 + 奖励)")
        
        # 训练
        agent.train(total_timesteps=total_timesteps)
        
        # 保存最终模型
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(models_dir, exist_ok=True)
        final_model_path = os.path.join(models_dir, "cloud_final_model.zip")
        agent.save(final_model_path)
        logger.info(f"最终模型已保存: {final_model_path}")
        
        # 打印训练摘要
        if hasattr(agent, 'training_monitor') and agent.training_monitor:
            agent.training_monitor.print_final_summary()
        
        print("\n✅ 云端训练完成！")
        print("📊 查看训练图表: logs/ 目录")
        print("🤖 模型文件: models/cloud_final_model.zip")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
