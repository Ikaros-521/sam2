import subprocess
import os
import shutil
import argparse

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
    if output_fps is None:
        output_fps = input_fps
    # 检查系统中是否安装了 ffmpeg
    if not shutil.which("ffmpeg"):
        print("错误：未能找到 FFmpeg 程序。")
        print("请确保您已正确安装 FFmpeg 并将其添加到了系统的 PATH 环境变量中。")
        print("下载地址: https://ffmpeg.org/download.html")
        return

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在 -> {input_path}")
        return

    # 根据视频编码确定比特流过滤器和临时文件名
    if video_codec.lower() == 'h264':
        bsf = 'h264_mp4toannexb'
        raw_stream_path = "temp_raw_video.h264"
    elif video_codec.lower() == 'h265':
        bsf = 'hevc_mp4toannexb'
        raw_stream_path = "temp_raw_video.h265"
    else:
        print(f"错误：不支持的视频编码 '{video_codec}'。目前只支持 'h264' 或 'h265'。")
        return

    # --- 第一步：从损坏的MP4中提取视频裸流 ---
    print("--- 步骤 1: 正在从损坏的MP4中提取视频裸流... ---")
    command1 = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'copy',
        '-bsf:v', bsf,
        '-an',              # 忽略音频，专注于视频
        '-y',               # 如果临时文件已存在，则覆盖
        raw_stream_path
    ]

    try:
        # 执行FFmpeg命令
        # check=True 表示如果命令执行失败（返回非零退出码），则会抛出异常
        # capture_output=True 和 text=True 用于捕获命令的输出信息，方便调试
        print(f"执行命令: {' '.join(command1)}")
        result1 = subprocess.run(
            command1,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8' # 指定编码以避免Windows下乱码
        )
        print("视频裸流提取成功！")

        # --- 第二步：将裸流重新封装成正常的MP4 ---
        print("\n--- 步骤 2: 正在将裸流重新封装为正常的MP4... ---")
        
        # 构建FFmpeg命令
        command2 = [
            'ffmpeg',
            '-framerate', str(input_fps),  # 使用输入帧率
            '-i', raw_stream_path,
            '-y'
        ]
        
        # 如果输入和输出帧率不同，需要进行帧率转换
        if input_fps != output_fps:
            print(f"检测到帧率转换需求: {input_fps} fps -> {output_fps} fps")
            command2.extend([
                '-r', str(output_fps),  # 设置输出帧率
                '-c:v', 'libx264' if video_codec.lower() == 'h264' else 'libx265'  # 重新编码
            ])
        else:
            # 帧率相同，直接复制
            command2.extend(['-c', 'copy'])
        
        command2.append(output_path)
        
        print(f"执行命令: {' '.join(command2)}")
        result2 = subprocess.run(
            command2,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(f"视频修复成功！输出文件位于: {output_path}")

    except FileNotFoundError:
        # 这个异常理论上在脚本开始时已经被 shutil.which 捕获，但作为双重保障
        print("错误: 'ffmpeg' 命令未找到。请再次确认其安装和环境变量设置。")
    except subprocess.CalledProcessError as e:
        # 捕获FFmpeg执行失败的错误
        print(f"FFmpeg 执行失败！返回码: {e.returncode}")
        print("-------------------- FFmpeg 错误信息 --------------------")
        print(e.stderr)
        print("---------------------------------------------------------")
        print("\n请检查您的输入文件路径、视频编码格式 (`video_codec`) 和帧率 (`frame_rate`) 是否设置正确。")
    finally:
        # --- 第三步：清理临时生成的裸流文件 ---
        if os.path.exists(raw_stream_path):
            os.remove(raw_stream_path)
            print(f"\n临时文件 '{raw_stream_path}' 已被成功删除。")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='修复损坏的流式MP4文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python video_stream_change.py input.mp4 output.mp4
  python video_stream_change.py input.mp4 output.mp4 --input-fps 25 --output-fps 30
  python video_stream_change.py input.mp4 output.mp4 --codec h265 --input-fps 29.97
        """
    )
    
    # 必需参数
    parser.add_argument('input_file', 
                       help='输入的损坏MP4文件路径')
    parser.add_argument('output_file', 
                       help='输出的修复后MP4文件路径')
    
    # 可选参数
    parser.add_argument('--input-fps', '--ifps', 
                       type=float, 
                       default=30.0,
                       help='输入视频的原始帧率 (默认: 30.0)')
    parser.add_argument('--output-fps', '--ofps', 
                       type=float, 
                       default=None,
                       help='输出视频的目标帧率 (默认: 与输入帧率相同)')
    parser.add_argument('--codec', 
                       choices=['h264', 'h265'], 
                       default='h264',
                       help='视频编码格式 (默认: h264)')
    
    return parser.parse_args()

if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果没有指定输出帧率，则使用输入帧率
    output_fps = args.output_fps if args.output_fps is not None else args.input_fps
    
    # ============================
    # --- 显示参数信息 ---
    # ============================
    print(f"准备修复视频: {args.input_file}")
    print(f"输出文件: {args.output_file}")
    print(f"输入帧率: {args.input_fps}")
    print(f"输出帧率: {output_fps}")
    print(f"视频编码: {args.codec}")
    print("-" * 40)

    # 调用主函数
    repair_fragmented_mp4(
        input_path=args.input_file,
        output_path=args.output_file,
        input_fps=args.input_fps,
        output_fps=output_fps,
        video_codec=args.codec
    )