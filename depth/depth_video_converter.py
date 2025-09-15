#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度视频转换程序
将普通视频转换为包含深度信息的视频

作者: AI Assistant
日期: 2024
"""

import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple, Union
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DepthVideoConverter:
    """深度视频转换器"""
    
    def __init__(self, 
                 model_name: str = "Intel/dpt-large",
                 device: str = "auto",
                 output_format: str = "side_by_side",
                 black_threshold: int = 30):
        """
        初始化深度视频转换器
        
        Args:
            model_name: 深度估计模型名称
            device: 计算设备 ("auto", "cuda", "cpu")
            output_format: 输出格式 ("side_by_side", "depth_only", "overlay")
        """
        self.model_name = model_name
        self.output_format = output_format
        self.black_threshold = black_threshold
        
        # 设置设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        
        # 初始化深度估计模型
        self._load_depth_model()
    
    def _load_depth_model(self):
        """加载深度估计模型"""
        try:
            from transformers import pipeline
            logger.info(f"正在加载深度估计模型: {self.model_name}")
            
            self.depth_estimator = pipeline(
                "depth-estimation", 
                model=self.model_name,
                device=0 if self.device.type == "cuda" else -1
            )
            logger.info("深度估计模型加载成功")
            
        except ImportError:
            logger.error("需要安装 transformers 库: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"加载深度估计模型失败: {e}")
            raise
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        估计单张图像的深度
        
        Args:
            image: 输入图像 (H, W, 3)
            
        Returns:
            depth_map: 深度图 (H, W)
        """
        try:
            # 转换图像格式
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # 检测黑色背景区域（假设黑色背景是抠图后的区域）
            # 直接判断RGB值，使用容差判断
            black_mask = (image_rgb[:, :, 0] < self.black_threshold) & \
                        (image_rgb[:, :, 1] < self.black_threshold) & \
                        (image_rgb[:, :, 2] < self.black_threshold)
            
            # 转换为PIL图像
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # 使用深度估计模型
            result = self.depth_estimator(pil_image)
            depth_map = np.array(result["depth"])
            
            # 归一化深度图到0-255
            depth_map = ((depth_map - depth_map.min()) / 
                        (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            
            # 将黑色背景区域设为0（黑色）
            depth_map[black_mask] = 0
            
            return depth_map
            
        except Exception as e:
            logger.error(f"深度估计失败: {e}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    def process_video(self, 
                     input_path: str, 
                     output_path: str,
                     start_frame: int = 0,
                     max_frames: Optional[int] = None,
                     fps: Optional[float] = None) -> bool:
        """
        处理视频文件
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            start_frame: 起始帧
            max_frames: 最大处理帧数
            fps: 输出视频帧率
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 打开输入视频
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"无法打开输入视频: {input_path}")
                return False
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps is None:
                fps = original_fps
            
            logger.info(f"视频信息: {width}x{height}, {total_frames}帧, {original_fps:.2f}fps")
            
            # 设置起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 计算实际处理帧数
            if max_frames is None:
                max_frames = total_frames - start_frame
            else:
                max_frames = min(max_frames, total_frames - start_frame)
            
            # 设置输出视频参数
            if self.output_format == "side_by_side":
                output_width = width * 2
                output_height = height
            elif self.output_format == "depth_only":
                output_width = width
                output_height = height
            else:  # overlay
                output_width = width
                output_height = height
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
            
            if not out.isOpened():
                logger.error(f"无法创建输出视频: {output_path}")
                cap.release()
                return False
            
            # 处理每一帧
            logger.info("开始处理视频帧...")
            processed_frames = 0
            
            with tqdm(total=max_frames, desc="处理进度") as pbar:
                while processed_frames < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 估计深度
                    depth_map = self.estimate_depth(frame)
                    
                    # 根据输出格式组合图像
                    output_frame = self._combine_frames(frame, depth_map)
                    
                    # 写入输出视频
                    out.write(output_frame)
                    
                    processed_frames += 1
                    pbar.update(1)
            
            # 释放资源
            cap.release()
            out.release()
            
            logger.info(f"视频处理完成: {processed_frames}帧已处理")
            return True
            
        except Exception as e:
            logger.error(f"视频处理失败: {e}")
            return False
    
    def _combine_frames(self, original_frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        根据输出格式组合原始图像和深度图
        
        Args:
            original_frame: 原始图像
            depth_map: 深度图
            
        Returns:
            combined_frame: 组合后的图像
        """
        if self.output_format == "side_by_side":
            # 并排显示
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            
            # 确保黑色背景区域在深度图中也保持为黑色
            original_black_mask = (original_frame[:, :, 0] < self.black_threshold) & \
                                (original_frame[:, :, 1] < self.black_threshold) & \
                                (original_frame[:, :, 2] < self.black_threshold)
            depth_colored[original_black_mask] = [0, 0, 0]
            
            combined = np.hstack([original_frame, depth_colored])
            
        elif self.output_format == "depth_only":
            # 只显示深度图，保持黑色背景
            # 创建彩色深度图
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            
            # 将深度值为0的区域（黑色背景）保持为黑色
            black_mask = depth_map == 0
            depth_colored[black_mask] = [0, 0, 0]  # 设为黑色
            
            # 确保黑色背景区域在彩色深度图中也保持为黑色
            # 重新检测原始图像的黑色背景，使用RGB容差判断
            original_black_mask = (original_frame[:, :, 0] < self.black_threshold) & \
                                (original_frame[:, :, 1] < self.black_threshold) & \
                                (original_frame[:, :, 2] < self.black_threshold)
            depth_colored[original_black_mask] = [0, 0, 0]
            
            combined = depth_colored
            
        else:  # overlay
            # 叠加显示
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            # 创建透明度混合
            alpha = 0.6
            combined = cv2.addWeighted(original_frame, 1-alpha, depth_colored, alpha, 0)
        
        return combined
    
    def process_image(self, input_path: str, output_path: str) -> bool:
        """
        处理单张图像
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            bool: 处理是否成功
        """
        try:
            # 读取图像
            image = cv2.imread(input_path)
            if image is None:
                logger.error(f"无法读取图像: {input_path}")
                return False
            
            # 估计深度
            depth_map = self.estimate_depth(image)
            
            # 根据输出格式组合图像
            output_image = self._combine_frames(image, depth_map)
            
            # 保存结果
            cv2.imwrite(output_path, output_image)
            logger.info(f"图像处理完成: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"图像处理失败: {e}")
            return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="深度视频转换程序")
    parser.add_argument("input", help="输入视频或图像路径")
    parser.add_argument("output", help="输出路径")
    parser.add_argument("--model", default="Intel/dpt-large", 
                       help="深度估计模型名称 (默认: Intel/dpt-large)")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "cpu"],
                       help="计算设备 (默认: auto)")
    parser.add_argument("--format", default="side_by_side",
                       choices=["side_by_side", "depth_only", "overlay"],
                       help="输出格式 (默认: side_by_side)")
    parser.add_argument("--start-frame", type=int, default=0,
                       help="起始帧 (默认: 0)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="最大处理帧数 (默认: 全部)")
    parser.add_argument("--fps", type=float, default=None,
                       help="输出视频帧率 (默认: 保持原帧率)")
    parser.add_argument("--black-threshold", type=int, default=30,
                       help="黑色背景检测阈值 (默认: 30)")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 初始化转换器
    converter = DepthVideoConverter(
        model_name=args.model,
        device=args.device,
        output_format=args.format,
        black_threshold=args.black_threshold
    )
    
    # 判断输入类型
    input_ext = os.path.splitext(args.input)[1].lower()
    
    if input_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']:
        # 处理视频
        success = converter.process_video(
            input_path=args.input,
            output_path=args.output,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            fps=args.fps
        )
    elif input_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # 处理图像
        success = converter.process_image(
            input_path=args.input,
            output_path=args.output
        )
    else:
        logger.error(f"不支持的文件格式: {input_ext}")
        return
    
    if success:
        logger.info("转换完成!")
    else:
        logger.error("转换失败!")


if __name__ == "__main__":
    main()
