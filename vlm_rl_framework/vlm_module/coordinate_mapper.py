"""
坐标映射模块
"""

import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

class CoordinateMapper:
    """
    坐标映射器
    
    功能:
    - 像素坐标到世界坐标的转换
    - 相机标定和坐标变换
    - 多相机坐标融合
    """
    
    def __init__(self, camera_matrix: np.ndarray = None, 
                 dist_coeffs: np.ndarray = None,
                 camera_height: float = 1.5):
        """
        初始化坐标映射器
        
        Args:
            camera_matrix: 相机内参矩阵
            dist_coeffs: 畸变系数
            camera_height: 相机高度
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_height = camera_height
        
        # 如果没有提供相机参数，使用默认值
        if self.camera_matrix is None:
            self.camera_matrix = np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ])
        
        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)
        
        logger.info("坐标映射器初始化完成")
    
    def pixel_to_world(self, pixel_coords: Tuple[int, int], 
                      depth: float = None) -> np.ndarray:
        """
        像素坐标转换为世界坐标
        
        Args:
            pixel_coords: 像素坐标 (u, v)
            depth: 深度值（如果为None，使用默认深度）
            
        Returns:
            np.ndarray: 世界坐标 [x, y, z]
        """
        u, v = pixel_coords
        
        if depth is None:
            # 使用默认深度（基于相机高度）
            depth = self.camera_height * 0.7
        
        # 反投影到相机坐标系
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 计算相机坐标系下的坐标
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        
        # 转换到世界坐标系（假设相机朝下，Z轴向上）
        x_world = x_cam
        y_world = -y_cam
        z_world = self.camera_height - z_cam
        
        world_coords = np.array([x_world, y_world, z_world])
        
        logger.debug(f"像素坐标({u}, {v}) -> 世界坐标{x_world:.3f}, {y_world:.3f}, {z_world:.3f}")
        
        return world_coords
    
    def world_to_pixel(self, world_coords: np.ndarray) -> Tuple[int, int]:
        """
        世界坐标转换为像素坐标
        
        Args:
            world_coords: 世界坐标 [x, y, z]
            
        Returns:
            Tuple[int, int]: 像素坐标 (u, v)
        """
        x_world, y_world, z_world = world_coords
        
        # 转换到相机坐标系
        x_cam = x_world
        y_cam = -y_world
        z_cam = self.camera_height - z_world
        
        # 投影到像素坐标系
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        u = int(x_cam * fx / z_cam + cx)
        v = int(y_cam * fy / z_cam + cy)
        
        return (u, v)
    
    def estimate_object_position(self, bounding_box: List[int], 
                                image_size: Tuple[int, int]) -> np.ndarray:
        """
        基于边界框估计物体位置
        
        Args:
            bounding_box: 边界框 [x1, y1, x2, y2]
            image_size: 图像尺寸 (width, height)
            
        Returns:
            np.ndarray: 估计的世界坐标
        """
        x1, y1, x2, y2 = bounding_box
        img_width, img_height = image_size
        
        # 计算边界框中心
        center_u = (x1 + x2) // 2
        center_v = (y1 + y2) // 2
        
        # 计算边界框大小（用于估计深度）
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_size = max(bbox_width, bbox_height)
        
        # 基于边界框大小估计深度（物体越大，距离越近）
        estimated_depth = self._estimate_depth_from_bbox_size(bbox_size, image_size)
        
        # 转换为世界坐标
        world_coords = self.pixel_to_world((center_u, center_v), estimated_depth)
        
        logger.info(f"边界框{bounding_box} -> 世界坐标{world_coords}")
        
        return world_coords
    
    def _estimate_depth_from_bbox_size(self, bbox_size: int, 
                                     image_size: Tuple[int, int]) -> float:
        """
        基于边界框大小估计深度
        
        Args:
            bbox_size: 边界框大小
            image_size: 图像尺寸
            
        Returns:
            float: 估计的深度
        """
        img_width, img_height = image_size
        max_size = max(img_width, img_height)
        
        # 简单的线性关系：边界框越大，深度越小
        # 这里使用启发式方法，实际应用中需要更精确的标定
        normalized_size = bbox_size / max_size
        
        # 深度范围：0.3m - 1.0m
        min_depth = 0.3
        max_depth = 1.0
        
        estimated_depth = max_depth - (max_depth - min_depth) * normalized_size
        
        return estimated_depth
    
    def calibrate_camera(self, calibration_data: List[Dict]) -> bool:
        """
        相机标定
        
        Args:
            calibration_data: 标定数据列表，每个元素包含：
                - 'pixel_coords': 像素坐标
                - 'world_coords': 对应的世界坐标
                
        Returns:
            bool: 标定是否成功
        """
        try:
            # 提取数据
            pixel_points = []
            world_points = []
            
            for data in calibration_data:
                pixel_points.append(data['pixel_coords'])
                world_points.append(data['world_coords'])
            
            pixel_points = np.array(pixel_points, dtype=np.float32)
            world_points = np.array(world_points, dtype=np.float32)
            
            # 使用OpenCV进行标定（如果可用）
            try:
                import cv2
                
                # 假设世界坐标在Z=0平面上
                world_points_2d = world_points[:, :2]
                
                # 计算单应性矩阵
                H, status = cv2.findHomography(pixel_points, world_points_2d)
                
                if H is not None:
                    # 更新相机参数
                    self.homography_matrix = H
                    logger.info("相机标定成功")
                    return True
                else:
                    logger.error("相机标定失败：无法计算单应性矩阵")
                    return False
                    
            except ImportError:
                logger.warning("OpenCV未安装，使用简化标定方法")
                return self._simple_calibration(pixel_points, world_points)
                
        except Exception as e:
            logger.error(f"相机标定失败: {e}")
            return False
    
    def _simple_calibration(self, pixel_points: np.ndarray, 
                           world_points: np.ndarray) -> bool:
        """
        简化标定方法
        
        Args:
            pixel_points: 像素坐标点
            world_points: 世界坐标点
            
        Returns:
            bool: 标定是否成功
        """
        try:
            # 使用最小二乘法拟合线性变换
            # 这里使用简化的方法，实际应用中需要更复杂的标定
            
            # 计算像素坐标和世界坐标的均值
            pixel_mean = np.mean(pixel_points, axis=0)
            world_mean = np.mean(world_points, axis=0)
            
            # 计算缩放因子
            pixel_std = np.std(pixel_points, axis=0)
            world_std = np.std(world_points, axis=0)
            
            # 简单的线性变换参数
            self.scale_x = world_std[0] / pixel_std[0] if pixel_std[0] > 0 else 0.001
            self.scale_y = world_std[1] / pixel_std[1] if pixel_std[1] > 0 else 0.001
            self.offset_x = world_mean[0] - pixel_mean[0] * self.scale_x
            self.offset_y = world_mean[1] - pixel_mean[1] * self.scale_y
            
            logger.info("简化标定完成")
            return True
            
        except Exception as e:
            logger.error(f"简化标定失败: {e}")
            return False
    
    def get_calibration_info(self) -> Dict:
        """获取标定信息"""
        info = {
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "camera_height": self.camera_height
        }
        
        if hasattr(self, 'homography_matrix'):
            info["homography_matrix"] = self.homography_matrix.tolist()
        
        if hasattr(self, 'scale_x'):
            info["scale_x"] = self.scale_x
            info["scale_y"] = self.scale_y
            info["offset_x"] = self.offset_x
            info["offset_y"] = self.offset_y
        
        return info
