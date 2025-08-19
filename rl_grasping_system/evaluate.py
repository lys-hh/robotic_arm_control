"""
强化学习抓取评估脚本
用于测试训练好的模型性能
"""

import os
import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from environment import PandaGraspingEnv
from agent import GraspingAgent

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_model(model_path: str, config):
    """加载模型"""
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 创建环境
    env = PandaGraspingEnv(
        grasping_config=config.grasping,
        reward_config=config.reward
    )
    
    # 创建智能体
    agent = GraspingAgent(
        network_config=config.network,
        training_config=config.training,
        model_path=model_path
    )
    
    # 设置环境
    agent.set_environment(env)
    
    logger.info(f"模型加载成功: {model_path}")
    return agent, env

def run_episode(agent, env, render: bool = False, max_steps: int = 500):
    """运行单个episode"""
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_info = []
    
    while episode_length < max_steps:
        # 预测动作
        action, _ = agent.predict(obs, deterministic=True)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 记录信息
        episode_reward += reward
        episode_length += 1
        episode_info.append({
            'step': episode_length,
            'reward': reward,
            'cumulative_reward': episode_reward,
            'distance_to_object': info.get('distance_to_object', 0),
            'gripper_width': info.get('gripper_width', 0),
            'is_grasped': info.get('is_grasped', False),
            'grasp_success': info.get('grasp_success', False)
        })
        
        # 渲染
        if render:
            env.render()
        
        # 检查是否结束
        if terminated or truncated:
            break
    
    return {
        'episode_reward': episode_reward,
        'episode_length': episode_length,
        'grasp_success': info.get('grasp_success', False),
        'final_distance': info.get('distance_to_object', 0),
        'episode_info': episode_info
    }

def evaluate_model(agent, env, n_episodes: int = 50, render: bool = False):
    """评估模型性能"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始评估，episode数量: {n_episodes}")
    
    results = []
    success_count = 0
    
    for episode in range(n_episodes):
        logger.info(f"运行episode {episode + 1}/{n_episodes}")
        
        episode_result = run_episode(agent, env, render=render)
        results.append(episode_result)
        
        if episode_result['grasp_success']:
            success_count += 1
        
        logger.info(f"Episode {episode + 1} - 奖励: {episode_result['episode_reward']:.3f}, "
                   f"步数: {episode_result['episode_length']}, "
                   f"成功: {episode_result['grasp_success']}")
    
    # 计算统计信息
    success_rate = success_count / n_episodes
    avg_reward = np.mean([r['episode_reward'] for r in results])
    avg_length = np.mean([r['episode_length'] for r in results])
    avg_final_distance = np.mean([r['final_distance'] for r in results])
    
    # 成功episode的统计
    successful_episodes = [r for r in results if r['grasp_success']]
    if successful_episodes:
        avg_success_reward = np.mean([r['episode_reward'] for r in successful_episodes])
        avg_success_length = np.mean([r['episode_length'] for r in successful_episodes])
    else:
        avg_success_reward = 0
        avg_success_length = 0
    
    evaluation_results = {
        'n_episodes': n_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_episode_length': avg_length,
        'avg_final_distance': avg_final_distance,
        'avg_success_reward': avg_success_reward,
        'avg_success_length': avg_success_length,
        'episode_results': results
    }
    
    # 打印结果
    logger.info("评估完成")
    logger.info(f"成功率: {success_rate:.3f} ({success_count}/{n_episodes})")
    logger.info(f"平均奖励: {avg_reward:.3f}")
    logger.info(f"平均步数: {avg_length:.1f}")
    logger.info(f"平均最终距离: {avg_final_distance:.3f}")
    if successful_episodes:
        logger.info(f"成功episode平均奖励: {avg_success_reward:.3f}")
        logger.info(f"成功episode平均步数: {avg_success_length:.1f}")
    
    return evaluation_results

def plot_results(results, save_path: str = None):
    """绘制评估结果"""
    logger = logging.getLogger(__name__)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('抓取任务评估结果', fontsize=16)
    
    # 1. 奖励分布
    rewards = [r['episode_reward'] for r in results['episode_results']]
    axes[0, 0].hist(rewards, bins=20, alpha=0.7, color='blue')
    axes[0, 0].set_title('奖励分布')
    axes[0, 0].set_xlabel('奖励')
    axes[0, 0].set_ylabel('频次')
    axes[0, 0].axvline(results['avg_reward'], color='red', linestyle='--', 
                       label=f'平均: {results["avg_reward"]:.3f}')
    axes[0, 0].legend()
    
    # 2. 步数分布
    lengths = [r['episode_length'] for r in results['episode_results']]
    axes[0, 1].hist(lengths, bins=20, alpha=0.7, color='green')
    axes[0, 1].set_title('步数分布')
    axes[0, 1].set_xlabel('步数')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].axvline(results['avg_episode_length'], color='red', linestyle='--',
                       label=f'平均: {results["avg_episode_length"]:.1f}')
    axes[0, 1].legend()
    
    # 3. 成功率随时间变化
    success_rates = []
    window_size = 10
    for i in range(window_size, len(results['episode_results']) + 1):
        window = results['episode_results'][i-window_size:i]
        success_rate = sum(1 for r in window if r['grasp_success']) / len(window)
        success_rates.append(success_rate)
    
    axes[1, 0].plot(range(window_size, len(results['episode_results']) + 1), success_rates)
    axes[1, 0].set_title(f'成功率变化 (窗口大小: {window_size})')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('成功率')
    axes[1, 0].axhline(results['success_rate'], color='red', linestyle='--',
                       label=f'总体成功率: {results["success_rate"]:.3f}')
    axes[1, 0].legend()
    
    # 4. 距离分布
    distances = [r['final_distance'] for r in results['episode_results']]
    axes[1, 1].hist(distances, bins=20, alpha=0.7, color='orange')
    axes[1, 1].set_title('最终距离分布')
    axes[1, 1].set_xlabel('距离')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].axvline(results['avg_final_distance'], color='red', linestyle='--',
                       label=f'平均: {results["avg_final_distance"]:.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"结果图已保存到: {save_path}")
    
    plt.show()

def save_results(results, save_path: str):
    """保存评估结果"""
    logger = logging.getLogger(__name__)
    
    import json
    
    # 准备保存的数据
    save_data = {
        'evaluation_time': datetime.now().isoformat(),
        'n_episodes': results['n_episodes'],
        'success_rate': results['success_rate'],
        'avg_reward': results['avg_reward'],
        'avg_episode_length': results['avg_episode_length'],
        'avg_final_distance': results['avg_final_distance'],
        'avg_success_reward': results['avg_success_reward'],
        'avg_success_length': results['avg_success_length'],
        'episode_summaries': [
            {
                'episode_reward': r['episode_reward'],
                'episode_length': r['episode_length'],
                'grasp_success': r['grasp_success'],
                'final_distance': r['final_distance']
            }
            for r in results['episode_results']
        ]
    }
    
    # 保存到文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"评估结果已保存到: {save_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="强化学习抓取评估")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--n_episodes", type=int, default=50, help="评估episode数量")
    parser.add_argument("--render", action="store_true", help="是否渲染")
    parser.add_argument("--save_results", type=str, help="结果保存路径")
    parser.add_argument("--save_plot", type=str, help="图表保存路径")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    try:
        # 获取配置
        config = get_config()
        
        # 设置默认保存路径
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.save_results is None:
            args.save_results = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
        if args.save_plot is None:
            args.save_plot = os.path.join(results_dir, f"evaluation_plot_{timestamp}.png")
        
        # 加载模型
        agent, env = load_model(args.model_path, config)
        
        # 评估模型
        results = evaluate_model(agent, env, args.n_episodes, args.render)
        
        # 绘制结果
        plot_results(results, args.save_plot)
        
        # 保存结果
        save_results(results, args.save_results)
        
        logger.info("评估完成")
        
    except Exception as e:
        logger.error(f"评估失败: {e}")
        raise

if __name__ == "__main__":
    main()
