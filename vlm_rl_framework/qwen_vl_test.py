"""
测试修复后的VLM处理器
"""

import numpy as np
from PIL import Image
from vlm_module.vlm_processor import VLMProcessor

def test_fixed_vlm():
    """测试修复后的VLM处理器"""
    print("=" * 60)
    print("测试修复后的VLM处理器")
    print("=" * 60)
    
    try:
        # 1. 加载VLM处理器
        print("\n1. 加载VLM处理器...")
        vlm = VLMProcessor()
        print(f"✅ VLM处理器加载成功")
        
        # 2. 创建测试图像
        print("\n2. 创建测试图像...")
        image_array = np.zeros((480, 640, 3), dtype=np.uint8)
        image_array[200:250, 250:300] = [255, 0, 0]  # 红色方块
        print(f"✅ 测试图像创建成功，形状: {image_array.shape}")
        
        # 3. 测试VLM处理
        print("\n3. 测试VLM处理...")
        try:
            result = vlm.process_instruction(image_array, "移动到红色立方体上方")
            print(f"✅ VLM处理成功")
            print(f"   返回结果: {result}")
        except Exception as e:
            print(f"❌ VLM处理失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_vlm()
