# 深度视频转换程序

一个专门用于将普通视频转换为包含深度信息的视频的Python程序。

## 功能特点

- 🎥 **视频深度估计**: 使用先进的深度学习模型估计视频中每一帧的深度信息
- 🖼️ **图像深度估计**: 支持单张图像的深度估计
- 🎨 **多种输出格式**: 支持并排显示、深度图单独显示、叠加显示等多种输出格式
- ⚡ **GPU加速**: 支持CUDA加速，大幅提升处理速度
- 🔧 **灵活配置**: 支持多种深度估计模型和自定义参数
- 📊 **进度显示**: 实时显示处理进度
- 🎯 **批量处理**: 支持批量处理多个文件

## 安装依赖

```bash
pip install torch torchvision opencv-python transformers tqdm numpy
```

## 快速开始

### 基本使用

```bash
# 转换视频
python depth_video_converter.py input_video.mp4 output_depth_video.mp4

# 转换图像
python depth_video_converter.py input_image.jpg output_depth_image.jpg
```

### 命令行参数

```bash
python depth_video_converter.py [输入文件] [输出文件] [选项]

选项:
  --model MODEL         深度估计模型名称 (默认: Intel/dpt-large)
  --device DEVICE       计算设备 (auto/cuda/cpu, 默认: auto)
  --format FORMAT       输出格式 (side_by_side/depth_only/overlay, 默认: side_by_side)
  --start-frame N       起始帧 (默认: 0)
  --max-frames N        最大处理帧数 (默认: 全部)
  --fps FPS            输出视频帧率 (默认: 保持原帧率)
```

## 支持的模型

程序支持多种深度估计模型：

- `Intel/dpt-large`: Intel DPT大模型（推荐，精度高）
- `Intel/dpt-hybrid-midas`: Intel DPT混合模型（平衡精度和速度）
- `facebook/dpt-dinov2-small-kitti`: Facebook DPT小模型（速度快）

## 输出格式

### 1. side_by_side（并排显示）
原始图像和深度图并排显示，便于对比。

### 2. depth_only（深度图单独显示）
只显示深度图，使用彩色映射显示深度信息。

### 3. overlay（叠加显示）
将深度图叠加到原始图像上，透明度可调。

## 使用示例

### 示例1：基本视频转换

```python
from depth_video_converter import DepthVideoConverter

# 初始化转换器
converter = DepthVideoConverter(
    model_name="Intel/dpt-large",
    device="auto",
    output_format="side_by_side"
)

# 转换视频
success = converter.process_video(
    input_path="input.mp4",
    output_path="output.mp4"
)
```

### 示例2：处理视频片段

```python
# 从第100帧开始，处理200帧
success = converter.process_video(
    input_path="long_video.mp4",
    output_path="segment.mp4",
    start_frame=100,
    max_frames=200,
    fps=24
)
```

### 示例3：图像处理

```python
# 处理单张图像
success = converter.process_image(
    input_path="image.jpg",
    output_path="depth_image.jpg"
)
```

### 示例4：批量处理

```python
import os

converter = DepthVideoConverter()

video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]

for i, video_file in enumerate(video_files):
    if os.path.exists(video_file):
        output_file = f"depth_video_{i+1}.mp4"
        converter.process_video(video_file, output_file)
```

## 高级功能

### 自定义深度估计

```python
# 使用不同的深度估计模型
converter = DepthVideoConverter(
    model_name="facebook/dpt-dinov2-small-kitti",  # 更快的模型
    device="cpu",                                  # 强制使用CPU
    output_format="depth_only"                    # 只输出深度图
)
```

### 处理特定帧范围

```python
# 只处理视频的前100帧
converter.process_video(
    input_path="input.mp4",
    output_path="output.mp4",
    start_frame=0,
    max_frames=100
)
```

## 性能优化建议

1. **使用GPU**: 如果有NVIDIA GPU，使用`--device cuda`可以大幅提升处理速度
2. **选择合适的模型**: 
   - 高精度：`Intel/dpt-large`
   - 平衡：`Intel/dpt-hybrid-midas`
   - 高速度：`facebook/dpt-dinov2-small-kitti`
3. **限制处理帧数**: 对于长视频，可以先用`--max-frames`测试效果
4. **调整输出格式**: `depth_only`格式处理速度最快

## 常见问题

### Q: 处理速度很慢怎么办？
A: 
- 使用GPU加速：`--device cuda`
- 选择更快的模型：`--model facebook/dpt-dinov2-small-kitti`
- 减少处理帧数：`--max-frames 100`

### Q: 内存不足怎么办？
A: 
- 使用CPU：`--device cpu`
- 减少处理帧数
- 选择更小的模型

### Q: 深度图质量不好怎么办？
A: 
- 使用更高精度的模型：`--model Intel/dpt-large`
- 确保输入视频质量良好
- 尝试不同的输出格式

## 技术原理

程序使用基于Transformer的深度估计模型（如DPT）来估计视频中每一帧的深度信息：

1. **特征提取**: 使用预训练的视觉Transformer提取图像特征
2. **深度回归**: 通过深度回归头预测每个像素的深度值
3. **后处理**: 对深度图进行归一化和可视化处理
4. **视频合成**: 将深度信息与原始视频结合生成最终输出

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 更新日志

- v1.0.0: 初始版本，支持基本的视频和图像深度转换功能
