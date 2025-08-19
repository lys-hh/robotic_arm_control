"""
VLM诊断工具 - 帮助诊断VLM处理失败的具体原因
"""

import numpy as np
import torch
import logging
from vlm_module.vlm_processor import VLMProcessor

logger = logging.getLogger(__name__)

class VLMDiagnostic:
    """VLM诊断工具"""
    
    def __init__(self, cache_dir: str = "./vlm_models"):
        self.cache_dir = cache_dir
        self.vlm_processor = None
    
    def diagnose_vlm_loading(self):
        """诊断VLM模型加载问题"""
        print("=" * 50)
        print("VLM模型加载诊断")
        print("=" * 50)
        
        try:
            print("1. 检查CUDA可用性...")
            if torch.cuda.is_available():
                print(f"   ✅ CUDA可用，设备数量: {torch.cuda.device_count()}")
                print(f"   GPU名称: {torch.cuda.get_device_name(0)}")
                print(f"   GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                print("   ❌ CUDA不可用")
            
            print("\n2. 检查模型缓存目录...")
            import os
            if os.path.exists(self.cache_dir):
                print(f"   ✅ 缓存目录存在: {self.cache_dir}")
                cache_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                               for dirpath, dirnames, filenames in os.walk(self.cache_dir)
                               for filename in filenames)
                print(f"   缓存大小: {cache_size / 1024**3:.1f}GB")
            else:
                print(f"   ❌ 缓存目录不存在: {self.cache_dir}")
            
            print("\n3. 尝试加载VLM处理器...")
            self.vlm_processor = VLMProcessor(cache_dir=self.cache_dir)
            print("   ✅ VLM处理器加载成功")
            
            print("\n4. 检查模型信息...")
            model_info = self.vlm_processor.get_model_info()
            print(f"   模型名称: {model_info['model_name']}")
            print(f"   使用设备: {model_info['device']}")
            print(f"   参数数量: {model_info['parameters']:,}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ VLM加载失败: {e}")
            return False
    
    def diagnose_image_processing(self, test_image: np.ndarray):
        """诊断图像处理问题"""
        print("\n" + "=" * 50)
        print("图像处理诊断")
        print("=" * 50)
        
        try:
            print(f"1. 检查输入图像...")
            print(f"   图像尺寸: {test_image.shape}")
            print(f"   数据类型: {test_image.dtype}")
            print(f"   数值范围: [{test_image.min()}, {test_image.max()}]")
            
            print("\n2. 尝试图像预处理...")
            processed_image = self.vlm_processor._preprocess_image(test_image)
            print(f"   预处理后尺寸: {processed_image.shape}")
            print(f"   预处理后数据类型: {processed_image.dtype}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ 图像处理失败: {e}")
            return False
    
    def diagnose_vlm_inference(self, test_image: np.ndarray, test_instruction: str):
        """诊断VLM推理问题"""
        print("\n" + "=" * 50)
        print("VLM推理诊断")
        print("=" * 50)
        
        try:
            print(f"1. 测试指令: {test_instruction}")
            
            print("\n2. 构建提示词...")
            prompt = self.vlm_processor._build_prompt(test_instruction)
            print(f"   提示词长度: {len(prompt)} 字符")
            print(f"   提示词前200字符: {prompt[:200]}...")
            
            print("\n3. 尝试VLM推理...")
            result = self.vlm_processor.process_instruction(test_image, test_instruction)
            print(f"   ✅ VLM推理成功")
            print(f"   返回结果: {result}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ VLM推理失败: {e}")
            return False
    
    def create_test_image(self) -> np.ndarray:
        """创建测试图像"""
        # 创建一个简单的测试图像，包含多个物体
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加红色立方体
        image[200:250, 250:300] = [255, 0, 0]
        
        # 添加蓝色圆形
        image[150:200, 400:450] = [0, 0, 255]
        
        # 添加绿色方块
        image[250:300, 150:200] = [0, 255, 0]
        
        # 添加机器人基座
        image[400:450, 250:390] = [128, 128, 128]
        
        return image
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("开始VLM完整诊断...")
        
        # 1. 诊断模型加载
        if not self.diagnose_vlm_loading():
            print("❌ VLM模型加载失败，无法继续诊断")
            return False
        
        # 2. 创建测试图像
        test_image = self.create_test_image()
        
        # 3. 诊断图像处理
        if not self.diagnose_image_processing(test_image):
            print("❌ 图像处理失败，无法继续诊断")
            return False
        
        # 4. 诊断VLM推理
        test_instructions = [
            "移动到红色立方体上方",
            "抓取蓝色圆形到左边"
        ]
        
        for instruction in test_instructions:
            if not self.diagnose_vlm_inference(test_image, instruction):
                print(f"❌ VLM推理失败，指令: {instruction}")
                return False
        
        print("\n" + "=" * 50)
        print("✅ 诊断完成，VLM工作正常")
        print("=" * 50)
        return True

def main():
    """主函数"""
    diagnostic = VLMDiagnostic()
    diagnostic.run_full_diagnosis()

if __name__ == "__main__":
    main()
