"""
VecNormalize包装器
使用Stable-Baselines3的VecNormalize进行观察和奖励归一化
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import logging

class VecNormalizeWrapper(VecEnvWrapper):
    """简单的归一化包装器"""
    
    def __init__(self, venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0):
        super().__init__(venv)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        
        # 观察归一化统计量
        if self.norm_obs:
            self.obs_mean = np.zeros(self.observation_space.shape[0])
            self.obs_var = np.ones(self.observation_space.shape[0])
            self.obs_count = 0
            
        # 奖励归一化统计量
        if self.norm_reward:
            self.reward_mean = 0.0
            self.reward_var = 1.0
            self.reward_count = 0
            
        self.logger = logging.getLogger(__name__)
        
    def update_obs_stats(self, obs):
        """更新观察统计量"""
        if not self.norm_obs:
            return
            
        batch_mean = np.mean(obs, axis=0)
        batch_var = np.var(obs, axis=0)
        batch_count = obs.shape[0]
        
        if self.obs_count == 0:
            self.obs_mean = batch_mean
            self.obs_var = batch_var
            self.obs_count = batch_count
        else:
            delta = batch_mean - self.obs_mean
            tot_count = self.obs_count + batch_count
            
            new_mean = self.obs_mean + delta * batch_count / tot_count
            m_a = self.obs_var * self.obs_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.obs_count * batch_count / tot_count
            new_var = M2 / tot_count
            
            self.obs_mean = new_mean
            self.obs_var = new_var
            self.obs_count = tot_count
            
    def update_reward_stats(self, rewards):
        """更新奖励统计量"""
        if not self.norm_reward:
            return
            
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)
        batch_count = len(rewards)
        
        if self.reward_count == 0:
            self.reward_mean = batch_mean
            self.reward_var = batch_var
            self.reward_count = batch_count
        else:
            delta = batch_mean - self.reward_mean
            tot_count = self.reward_count + batch_count
            
            new_mean = self.reward_mean + delta * batch_count / tot_count
            m_a = self.reward_var * self.reward_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / tot_count
            new_var = M2 / tot_count
            
            self.reward_mean = new_mean
            self.reward_var = new_var
            self.reward_count = tot_count
            
    def normalize_obs(self, obs):
        """归一化观察"""
        if not self.norm_obs or self.obs_count == 0:
            return obs
            
        normalized = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)
        
    def normalize_reward(self, reward):
        """归一化奖励"""
        if not self.norm_reward or self.reward_count == 0:
            return reward
            
        normalized = (reward - self.reward_mean) / np.sqrt(self.reward_var + 1e-8)
        return np.clip(normalized, -self.clip_reward, self.clip_reward)
        
    def reset(self):
        """重置环境"""
        obs = self.venv.reset()
        if self.norm_obs:
            obs = self.normalize_obs(obs)
        return obs  # 只返回观察数组
        
    def step_wait(self):
        """执行一步"""
        obs, rewards, dones, infos = self.venv.step_wait()
        
        # 更新统计量
        if self.norm_obs:
            self.update_obs_stats(obs)
            obs = self.normalize_obs(obs)
            
        if self.norm_reward:
            self.update_reward_stats(rewards)
            rewards = self.normalize_reward(rewards)
            
        return obs, rewards, dones, infos

def create_normalized_env(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0):
    """创建归一化环境"""
    vec_env = DummyVecEnv([lambda: env])
    normalized_env = VecNormalizeWrapper(
        vec_env, 
        norm_obs=norm_obs, 
        norm_reward=norm_reward,
        clip_obs=clip_obs,
        clip_reward=clip_reward
    )
    return normalized_env
