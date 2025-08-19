"""
检查当前模型配置并尝试下载正确的Qwen-VL-Chat模型
"""

import os
import json
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM
import torch

def check_current_model():
    """检查当前模型配置"""
    print("=" * 60)
    print("检查当前模型配置")
    print("=" * 60)
    
    try:
        # 1. 检查模型配置文件
        print("\n1. 检查模型配置文件...")
        config_path = "./vlm_models/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8/config.json"
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            print(f"✅ 找到配置文件")
            print(f"   模型类型: {config_data.get('model_type', 'Unknown')}")
            print(f"   架构: {config_data.get('architectures', 'Unknown')}")
            print(f"   是否有视觉配置: {'vision_config' in config_data}")
            
            if 'vision_config' in config_data:
                print(f"   视觉配置: {config_data['vision_config']}")
            else:
                print(f"   ❌ 没有视觉配置")
                
            # 检查其他关键配置
            print(f"   词汇表大小: {config_data.get('vocab_size', 'Unknown')}")
            print(f"   隐藏层大小: {config_data.get('hidden_size', 'Unknown')}")
            print(f"   层数: {config_data.get('num_hidden_layers', 'Unknown')}")
            
        else:
            print(f"❌ 配置文件不存在: {config_path}")
        
        # 2. 尝试加载配置
        print("\n2. 尝试加载配置...")
        try:
            config = AutoConfig.from_pretrained(
                "Qwen/Qwen-VL-Chat",
                cache_dir="./vlm_models",
                local_files_only=True,
                trust_remote_code=True
            )
            print(f"✅ 配置加载成功")
            print(f"   配置类型: {type(config)}")
            print(f"   模型类型: {config.model_type}")
            print(f"   是否有视觉配置: {hasattr(config, 'vision_config')}")
            
            if hasattr(config, 'vision_config'):
                print(f"   视觉配置: {config.vision_config}")
            else:
                print(f"   ❌ 没有视觉配置")
                
        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
        
        # 3. 检查模型文件
        print("\n3. 检查模型文件...")
        model_dir = "./vlm_models/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
        
        if os.path.exists(model_dir):
            files = os.listdir(model_dir)
            print(f"✅ 模型目录存在，文件数量: {len(files)}")
            print(f"   文件列表: {files}")
            
            # 检查是否有视觉相关文件
            vision_files = [f for f in files if 'vision' in f.lower() or 'image' in f.lower()]
            print(f"   视觉相关文件: {vision_files}")
            
            # 检查模型权重文件
            model_files = [f for f in files if f.endswith('.bin') or f.endswith('.safetensors')]
            print(f"   模型权重文件: {model_files}")
            
        else:
            print(f"❌ 模型目录不存在: {model_dir}")
        
        # 4. 尝试下载正确的模型
        print("\n4. 尝试下载正确的模型...")
        print("   注意：这需要网络连接")
        
        # 检查网络连接
        try:
            import requests
            response = requests.get("https://huggingface.co", timeout=5)
            print(f"   ✅ 网络连接正常")
            
            # 尝试下载正确的模型
            print("\n   尝试下载Qwen-VL-Chat-7B...")
            try:
                # 先尝试下载配置
                config = AutoConfig.from_pretrained(
                    "Qwen/Qwen-VL-Chat-7B",
                    trust_remote_code=True
                )
                print(f"   ✅ 配置下载成功")
                print(f"   模型类型: {config.model_type}")
                print(f"   是否有视觉配置: {hasattr(config, 'vision_config')}")
                
                if hasattr(config, 'vision_config'):
                    print(f"   ✅ 这个模型支持视觉！")
                    print(f"   视觉配置: {config.vision_config}")
                    
                    # 尝试下载处理器
                    print("\n   尝试下载处理器...")
                    processor = AutoProcessor.from_pretrained(
                        "Qwen/Qwen-VL-Chat-7B",
                        trust_remote_code=True
                    )
                    print(f"   ✅ 处理器下载成功")
                    print(f"   处理器类型: {type(processor)}")
                    
                    # 检查是否有图像处理器
                    if hasattr(processor, 'image_processor'):
                        print(f"   ✅ 有图像处理器")
                    else:
                        print(f"   ❌ 没有图像处理器")
                        
                else:
                    print(f"   ❌ 这个模型也不支持视觉")
                    
            except Exception as e:
                print(f"   ❌ 下载失败: {e}")
                
        except Exception as e:
            print(f"   ❌ 网络连接失败: {e}")
            print(f"   无法下载新模型")
        
        print("\n" + "=" * 60)
        print("检查完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 检查过程中出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_current_model()
