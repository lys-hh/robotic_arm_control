"""
VLM处理器 - 视觉语言模型的核心处理类
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoProcessor, AutoModelForCausalLM
import logging
import json

logger = logging.getLogger(__name__)

class VLMProcessor:
    """
    视觉语言模型处理器
    
    功能:
    - 加载和初始化VLM模型
    - 处理图像和文本输入
    - 输出目标检测结果
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen-VL-Chat", device: str = "auto", cache_dir: str = "./vlm_models"):
        """
        初始化VLM处理器
        
        Args:
            model_name: VLM模型名称
            device: 设备选择 ("auto", "cuda", "cpu")
            cache_dir: 模型缓存目录
        """
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.cache_dir = cache_dir
        
        # 初始化模型
        self.processor = None
        self.model = None
        self._load_model()
        
        logger.info(f"VLM处理器初始化完成，使用模型: {model_name}")
    
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("检测到CUDA，使用GPU加速")
            else:
                device = "cpu"
                logger.info("未检测到CUDA，使用CPU")
        return device
    
    def _load_model(self):
        """加载VLM模型"""
        try:
            logger.info("正在加载VLM模型...")
            
            # 设置环境变量
            import os
            os.environ['HF_HOME'] = self.cache_dir
            os.environ['TRANSFORMERS_CACHE'] = self.cache_dir
            
            # 检查GPU内存
            if self.device == "cuda" and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU显存: {gpu_memory:.1f}GB")
                if gpu_memory < 6:
                    logger.warning(f"GPU显存不足({gpu_memory:.1f}GB)，建议使用CPU模式")
            
            # 加载处理器
            logger.info("加载VLM处理器...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                local_files_only=True
            )
            logger.info("VLM处理器加载成功")
            
            # 加载模型
            logger.info("加载VLM模型...")
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,  # 使用BF16精度
                    device_map="auto",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="cpu",
                    cache_dir=self.cache_dir,
                    trust_remote_code=True,
                    local_files_only=True
                )
            
            logger.info(f"VLM模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"VLM模型加载失败: {e}")
            raise
    
    def process_instruction(self, image: np.ndarray, instruction: str) -> dict:
        """
        处理图像和指令，返回VLM分析结果
        
        Args:
            image: 输入图像 (numpy array)
            instruction: 指令文本
            
        Returns:
            dict: 包含分析结果的字典
        """
        logger.info(f"VLM处理开始 - 图像尺寸: {image.shape if image is not None else 'None'}, 指令: {instruction}")
        
        try:
            # 1. 图像预处理
            processed_image = self._preprocess_image(image)
            
            # 2. 构建包含图像标记的提示词
            prompt = self._build_prompt_with_image_marker(instruction)
            
            # 3. 使用AutoProcessor处理多模态输入（关键修复）
            logger.info("开始VLM多模态输入处理...")
            logger.info(f"提示词: {prompt}")
            logger.info(f"图像类型: {type(processed_image)}")
            logger.info(f"图像形状: {processed_image.shape if processed_image is not None else 'None'}")
            
            # 转换为PIL Image
            from PIL import Image
            pil_image = Image.fromarray(processed_image)
            logger.info(f"转换为PIL图像: {pil_image.size}")
            
            # 使用AutoProcessor处理多模态输入
            inputs = self.processor(
                text=prompt,
                images=pil_image,
                return_tensors="pt"
            )
            
            logger.info("VLM多模态输入处理成功")
            logger.info(f"输入类型: {type(inputs)}")
            logger.info(f"输入键: {list(inputs.keys())}")
            
            # 详细检查每个输入
            for key, value in inputs.items():
                logger.info(f"输入 {key}: 类型={type(value)}, 形状={value.shape if hasattr(value, 'shape') else 'No shape'}")
                if value is None:
                    raise ValueError(f"输入 {key} 为None")
            
            # 检查是否包含图像信息（关键检查）
            if 'pixel_values' in inputs:
                logger.info("✅ 包含pixel_values（图像信息）")
                logger.info(f"pixel_values形状: {inputs['pixel_values'].shape}")
            else:
                logger.error("❌ 缺少pixel_values（图像信息）")
                raise ValueError("VLM输入缺少图像信息，无法进行视觉推理")
                
        except Exception as e:
            logger.error(f"VLM多模态输入处理失败: {e}")
            raise ValueError(f"VLM多模态输入处理失败: {e}")
        
        # 4. 移动到设备
        try:
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logger.info(f"输入已移动到设备: {self.device}")
        except Exception as e:
            logger.error(f"设备移动失败: {e}")
            raise ValueError(f"设备移动失败: {e}")
        
        # 5. 生成回复
        with torch.no_grad():
            try:
                # 使用更简单的生成参数
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    use_cache=True
                )
                logger.info("VLM生成成功")
                logger.info(f"输出形状: {outputs.shape}")
            except Exception as e:
                logger.error(f"VLM生成失败: {e}")
                raise ValueError(f"VLM生成失败: {e}")
        
        # 6. 解码回复
        try:
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"VLM解码成功，原始回复: {response}")
        except Exception as e:
            logger.error(f"VLM解码失败: {e}")
            raise ValueError(f"VLM解码失败: {e}")
        
        # 7. 解析JSON响应
        try:
            result = self._parse_json_response(response)
            if result is None:
                raise ValueError("VLM返回的响应无法解析为有效的JSON格式")
            logger.info(f"VLM解析成功: {result}")
            return result
        except Exception as e:
            logger.error(f"VLM解析失败: {e}")
            raise ValueError(f"VLM解析失败: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """预处理图像用于VLM输入"""
        logger.info(f"预处理图像，原始尺寸: {image.shape}, 数据类型: {image.dtype}")
        
        # 检查输入图像
        if image is None:
            raise ValueError("输入图像为None")
        
        if not isinstance(image, np.ndarray):
            raise ValueError(f"图像类型错误: {type(image)}")
        
        if image.size == 0:
            raise ValueError("图像为空")
        
        logger.info(f"图像数值范围: [{image.min()}, {image.max()}]")
        
        # 确保图像是RGB格式
        if len(image.shape) == 3 and image.shape[2] == 3:
            processed_image = image
            logger.info("图像已经是RGB格式")
        elif len(image.shape) == 3 and image.shape[2] == 4:
            processed_image = image[:, :, :3]
            logger.info("从RGBA转换为RGB")
        elif len(image.shape) == 2:
            processed_image = np.stack([image] * 3, axis=2)
            logger.info("从灰度转换为RGB")
        else:
            raise ValueError(f"不支持的图像格式: {image.shape}")
        
        logger.info(f"格式转换后形状: {processed_image.shape}")
        
        # 确保数据类型为uint8
        if processed_image.dtype != np.uint8:
            if processed_image.max() <= 1.0:
                processed_image = (processed_image * 255).astype(np.uint8)
            else:
                processed_image = processed_image.astype(np.uint8)
            logger.info("数据类型转换为uint8")
        
        logger.info(f"数据类型转换后: {processed_image.dtype}")
        
        # 调整图像尺寸
        target_size = (224, 224)  # VLM标准输入尺寸
        if processed_image.shape[:2] != target_size:
            try:
                import cv2
                processed_image = cv2.resize(processed_image, target_size)
                logger.info(f"使用OpenCV调整图像尺寸为: {target_size}")
            except ImportError:
                from PIL import Image
                pil_image = Image.fromarray(processed_image)
                pil_image = pil_image.resize(target_size)
                processed_image = np.array(pil_image)
                logger.info(f"使用PIL调整图像尺寸为: {target_size}")
        
        logger.info(f"图像预处理完成，最终尺寸: {processed_image.shape}")
        logger.info(f"最终数据类型: {processed_image.dtype}")
        logger.info(f"最终数值范围: [{processed_image.min()}, {processed_image.max()}]")
        
        # 最终检查
        if processed_image is None:
            raise ValueError("图像预处理后为None")
        
        return processed_image
    
    def _build_prompt(self, instruction: str) -> str:
        """构建VLM提示词"""
        prompt = f"""
你是一个机器人控制助手。请根据图像和指令，识别目标物体的位置和任务要求。

指令: {instruction}

请分析图像中的物体，识别指定物体的位置，并根据指令确定任务目标。

请以JSON格式返回结果，包含以下字段:
- object_type: 物体类型 (如: red_cube, blue_circle, green_block, yellow_sphere)
- object_position: 物体在图像中的像素坐标 [x, y] (图像中心为[320, 240])
- object_world_position: 物体在世界坐标系中的位置 [x, y, z] (米为单位)
- task_type: 任务类型 ("move_to" 或 "grasp_and_move")
- pickup_position: 抓取位置 [x, y, z] (物体上方0.05米)
- place_position: 放置位置 [x, y, z] (根据指令确定)
- confidence: 置信度 (0-1)
- description: 任务描述

任务类型说明:
1. "move_to": 移动到物体上方
2. "grasp_and_move": 抓取物体并移动到指定位置

放置位置规则:
- "左边": 在物体左侧0.3米
- "右边": 在物体右侧0.3米  
- "上边": 在物体上方0.2米
- "前面": 在物体前方0.3米
- "后面": 在物体后方0.3米

注意:
1. 图像尺寸为640x480像素
2. 世界坐标系: X轴向右, Y轴向前, Z轴向上
3. 相机位于Z=1.5米高度，朝下观察
4. 请尽可能准确地估计物体位置
5. 必须返回有效的JSON格式

请直接返回JSON，不要包含其他文字。
"""
        return prompt
    
    def _build_prompt_with_image_marker(self, instruction: str) -> str:
        """
        构建包含图像标记的提示词
        
        Args:
            instruction: 原始指令
            
        Returns:
            str: 包含图像标记的提示词
        """
        prompt = f"""你是一个机器人控制助手。请根据图像和指令，识别目标物体的位置和任务要求。

指令: {instruction}

请分析图像中的物体，识别指定物体的位置，并根据指令确定任务目标。

请以JSON格式返回结果，包含以下字段:
- object_type: 物体类型 (如: red_cube, blue_circle, green_block, yellow_sphere)
- object_world_position: 物体在世界坐标系中的位置 [x, y, z]
- task_type: 任务类型 (move_to, grasp_and_move)
- pickup_position: 抓取位置 [x, y, z] (如果是抓取任务)
- place_position: 放置位置 [x, y, z] (如果是抓取任务)

请直接返回JSON，不要包含其他文字。"""
        
        # 添加图像标记（关键：告诉模型图像在哪里）
        prompt = f"<img></img> {prompt}"
        
        return prompt
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """解析VLM的JSON回复"""
        try:
            # 查找JSON部分
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                logger.error(f"未找到JSON格式: {response}")
                return None
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # 验证必要字段
            required_fields = ['object_type', 'object_world_position', 'task_type', 'pickup_position', 'place_position']
            for field in required_fields:
                if field not in result:
                    logger.error(f"缺少必要字段: {field}")
                    return None
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}, 原始回复: {response}")
            return None
        except Exception as e:
            logger.error(f"结果解析失败: {e}")
            return None
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
