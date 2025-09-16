import gradio as gr
import subprocess
import os
import shutil
import tempfile
from pathlib import Path

def repair_fragmented_mp4(input_path, output_path, input_fps=30, output_fps=None, video_codec='h264'):
    """
    通过提取裸流并重新封装来修复损坏的流式MP4文件。

    :param input_path: 输入的损坏MP4文件路径。
    :param output_path: 输出的修复后MP4文件路径。
    :param input_fps: 输入视频的原始帧率。
    :param output_fps: 输出视频的目标帧率，如果为None则与输入帧率相同。
    :param video_codec: 视频编码 ('h264' 或 'h265')。
    """
    # 如果没有指定输出帧率，则使用输入帧率
    if output_fps is None or output_fps == 0:
        output_fps = input_fps
    
    # 检查系统中是否安装了 ffmpeg
    if not shutil.which("ffmpeg"):
        return None, "错误：未能找到 FFmpeg 程序。请确保您已正确安装 FFmpeg 并将其添加到了系统的 PATH 环境变量中。"

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        return None, f"错误：输入文件不存在 -> {input_path}"

    # 根据视频编码确定比特流过滤器和临时文件名
    if video_codec.lower() == 'h264':
        bsf = 'h264_mp4toannexb'
        raw_stream_path = "temp_raw_video.h264"
    elif video_codec.lower() == 'h265':
        bsf = 'hevc_mp4toannexb'
        raw_stream_path = "temp_raw_video.h265"
    else:
        return None, f"错误：不支持的视频编码 '{video_codec}'。目前只支持 'h264' 或 'h265'。"

    try:
        # --- 第一步：从损坏的MP4中提取视频裸流 ---
        command1 = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'copy',
            '-bsf:v', bsf,
            '-an',              # 忽略音频，专注于视频
            '-y',               # 如果临时文件已存在，则覆盖
            raw_stream_path
        ]

        result1 = subprocess.run(
            command1,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        # --- 第二步：将裸流重新封装成正常的MP4 ---
        command2 = [
            'ffmpeg',
            '-framerate', str(input_fps),  # 使用输入帧率
            '-i', raw_stream_path,
            '-y'
        ]
        
        # 如果输入和输出帧率不同，需要进行帧率转换
        if input_fps != output_fps:
            command2.extend([
                '-r', str(output_fps),  # 设置输出帧率
                '-c:v', 'libx264' if video_codec.lower() == 'h264' else 'libx265'  # 重新编码
            ])
        else:
            # 帧率相同，直接复制
            command2.extend(['-c', 'copy'])
        
        command2.append(output_path)
        
        result2 = subprocess.run(
            command2,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        return output_path, "视频修复成功！"

    except FileNotFoundError:
        return None, "错误: 'ffmpeg' 命令未找到。请再次确认其安装和环境变量设置。"
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg 执行失败！返回码: {e.returncode}\n"
        error_msg += f"错误信息: {e.stderr}"
        return None, error_msg
    finally:
        # --- 第三步：清理临时生成的裸流文件 ---
        if os.path.exists(raw_stream_path):
            os.remove(raw_stream_path)

def process_video(video_file, input_fps, output_fps, video_codec):
    """
    处理上传的视频文件
    """
    if video_file is None:
        return None, "请先上传视频文件"
    
    # 创建临时输出文件
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        # 调用修复函数
        result_path, message = repair_fragmented_mp4(
            input_path=video_file.name,
            output_path=output_path,
            input_fps=input_fps,
            output_fps=output_fps,
            video_codec=video_codec
        )
        
        if result_path and os.path.exists(result_path):
            return result_path, message
        else:
            return None, message
            
    except Exception as e:
        return None, f"处理过程中发生错误: {str(e)}"

# 创建Gradio界面
def create_interface():
    with gr.Blocks(title="视频修复工具", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 视频修复工具")
        gr.Markdown("上传损坏的MP4文件，通过重新封装来修复视频流问题")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📁 文件上传")
                video_input = gr.File(
                    label="上传MP4视频文件",
                    file_types=[".mp4"],
                    type="filepath"
                )
                
                gr.Markdown("## ⚙️ 参数设置")
                input_fps = gr.Number(
                    label="输入帧率 (fps)",
                    value=30.0,
                    minimum=0.1,
                    maximum=120.0,
                    step=0.1,
                    info="原始视频的帧率"
                )
                
                output_fps = gr.Number(
                    label="输出帧率 (fps)",
                    value=None,
                    minimum=0,
                    maximum=120.0,
                    step=0.1,
                    info="目标帧率，留空则与输入帧率相同"
                )
                
                video_codec = gr.Dropdown(
                    label="视频编码",
                    choices=["h264", "h265"],
                    value="h264",
                    info="选择视频编码格式"
                )
                
                process_btn = gr.Button(
                    "🔧 开始修复",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("## 📺 处理结果")
                status_text = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    lines=3
                )
                
                video_output = gr.Video(
                    label="修复后的视频",
                    height=400
                )
                
                download_btn = gr.DownloadButton(
                    label="📥 下载视频",
                    variant="secondary"
                )
        
        # 处理按钮点击事件
        process_btn.click(
            fn=process_video,
            inputs=[video_input, input_fps, output_fps, video_codec],
            outputs=[video_output, status_text]
        )
        
        # 当视频输出更新时，更新下载按钮
        video_output.change(
            fn=lambda x: x if x else None,
            inputs=[video_output],
            outputs=[download_btn]
        )
        
        # 添加使用说明
        with gr.Accordion("📖 使用说明", open=False):
            gr.Markdown("""
            ### 功能说明
            - **文件上传**: 支持上传损坏的MP4视频文件
            - **参数设置**: 
              - 输入帧率：原始视频的帧率
              - 输出帧率：目标帧率（留空则与输入帧率相同）
              - 视频编码：选择H.264或H.265编码
            - **处理结果**: 显示修复后的视频并提供下载
            
            ### 使用步骤
            1. 上传需要修复的MP4文件
            2. 设置合适的参数（通常使用默认值即可）
            3. 点击"开始修复"按钮
            4. 等待处理完成
            5. 预览修复后的视频并下载
            
            ### 注意事项
            - 确保系统已安装FFmpeg
            - 支持H.264和H.265编码的视频
            - 处理时间取决于视频大小和复杂度
            """)
    
    return demo

if __name__ == "__main__":
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=False,
        show_error=True
    )
