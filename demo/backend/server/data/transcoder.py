# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import av
from app_conf import FFMPEG_NUM_THREADS
from dataclasses_json import dataclass_json

TRANSCODE_VERSION = 1


@dataclass_json
@dataclass
class VideoMetadata:
    duration_sec: Optional[float]
    video_duration_sec: Optional[float]
    container_duration_sec: Optional[float]
    fps: Optional[float]
    width: Optional[int]
    height: Optional[int]
    num_video_frames: int
    num_video_streams: int
    video_start_time: float


# 定义一个transcode函数，用于将输入的视频文件转换为指定格式
def transcode(
    # 输入文件路径
    in_path: str,
    # 输出文件路径
    out_path: str,
    # 输入视频元数据
    in_metadata: Optional[VideoMetadata],
    # 搜索时间
    seek_t: float,
    # 视频时长（秒）
    duration_time_sec: float,
    # 保持原始质量
    keep_original_quality: bool = True,  
):
    # 获取视频编码器
    codec = os.environ.get("VIDEO_ENCODE_CODEC", "libx264")
    # 获取视频质量参数
    crf = int(os.environ.get("VIDEO_ENCODE_CRF", "23"))
    # 获取视频帧率
    fps = int(os.environ.get("VIDEO_ENCODE_FPS", "24"))
    # 获取视频最大宽度
    max_w = int(os.environ.get("VIDEO_ENCODE_MAX_WIDTH", "1280"))
    # 获取视频最大高度
    max_h = int(os.environ.get("VIDEO_ENCODE_MAX_HEIGHT", "720"))
    # 获取是否输出详细信息
    verbose = ast.literal_eval(os.environ.get("VIDEO_ENCODE_VERBOSE", "False"))

    # 新增：如果保持原画质，使用原视频参数
    if keep_original_quality and in_metadata is not None:
        crf = 18  # 高画质
        fps = int(in_metadata.fps) if in_metadata.fps else fps
        max_w = int(in_metadata.width) if in_metadata.width else max_w
        max_h = int(in_metadata.height) if in_metadata.height else max_h

    # 调用normalize_video函数，将输入视频转换为指定格式
    normalize_video(
        # 输入文件路径
        in_path=in_path,
        # 输出文件路径
        out_path=out_path,
        # 视频最大宽度
        max_w=max_w,
        # 视频最大高度
        max_h=max_h,
        # 搜索时间
        seek_t=seek_t,
        # 视频时长
        max_time=duration_time_sec,
        # 输入视频元数据
        in_metadata=in_metadata,
        # 视频编码器
        codec=codec,
        # 视频质量参数
        crf=crf,
        # 视频帧率
        fps=fps,
        # 是否输出详细信息
        verbose=verbose,
        keep_original_quality=keep_original_quality,  # 传递参数
    )


def get_video_metadata(path: str) -> VideoMetadata:
    # 打开视频文件
    with av.open(path) as cont:
        # 获取视频流数量
        num_video_streams = len(cont.streams.video)
        # 初始化视频流的宽、高、帧率、视频时长、容器时长、视频开始时间、旋转角度、视频帧数
        width, height, fps = None, None, None
        video_duration_sec = 0
        container_duration_sec = float((cont.duration or 0) / av.time_base)
        video_start_time = 0.0
        rotation_deg = 0
        num_video_frames = 0
        # 如果视频流数量大于0
        if num_video_streams > 0:
            # 获取第一个视频流
            video_stream = cont.streams.video[0]
            # 断言视频流的时间基数不为空
            assert video_stream.time_base is not None

            # for rotation, see: https://github.com/PyAV-Org/PyAV/pull/1249
            # 兼容 PyAV 新旧版本，优先用 side_data_objects
            if hasattr(video_stream, "side_data_objects"):
                # 遍历视频流的 side_data_objects
                for side_data in video_stream.side_data_objects:
                    # 如果 side_data 的类型为 displaymatrix
                    if getattr(side_data, "type", None) == "displaymatrix":
                        # 新版 PyAV 有 to_degrees 方法
                        to_degrees = getattr(side_data, "to_degrees", None)
                        if callable(to_degrees):
                            # 获取旋转角度
                            rotation_deg = side_data.to_degrees()
                        else:
                            # 兼容性兜底
                            rotation_deg = 0
            # 兼容极老版本 PyAV
            elif hasattr(video_stream, "side_data"):
                # 获取旋转角度
                rotation_deg = video_stream.side_data.get("DISPLAYMATRIX", 0)
            else:
                # 默认旋转角度为0
                rotation_deg = 0

            # 获取视频帧数
            num_video_frames = video_stream.frames
            # 获取视频开始时间
            video_start_time = float(video_stream.start_time * video_stream.time_base)
            # 获取视频流的宽、高
            width, height = video_stream.width, video_stream.height
            # 获取视频流的帧率
            fps = float(video_stream.guessed_rate)
            fps_avg = video_stream.average_rate
            # 如果视频流有时长
            if video_stream.duration is not None:
                # 获取视频时长
                video_duration_sec = float(
                    video_stream.duration * video_stream.time_base
                )
            # 如果帧率为空
            if fps is None:
                # 使用平均帧率
                fps = float(fps_avg)

            # 如果旋转角度不为空且为90、-90、270、-270
            if not math.isnan(rotation_deg) and int(rotation_deg) in (
                90,
                -90,
                270,
                -270,
            ):
                # 交换宽、高
                width, height = height, width

        # 获取视频时长
        duration_sec = max(container_duration_sec, video_duration_sec)

        # 返回视频元数据
        return VideoMetadata(
            duration_sec=duration_sec,
            container_duration_sec=container_duration_sec,
            video_duration_sec=video_duration_sec,
            video_start_time=video_start_time,
            fps=fps,
            width=width,
            height=height,
            num_video_streams=num_video_streams,
            num_video_frames=num_video_frames,
        )


# 定义一个函数，用于标准化视频
def normalize_video(
    in_path: str,  # 输入视频路径
    out_path: str,  # 输出视频路径
    max_w: int,  # 最大宽度
    max_h: int,  # 最大高度
    seek_t: float,  # 跳过时间
    max_time: float,  # 最大时间
    in_metadata: Optional[VideoMetadata],  # 输入视频元数据
    codec: str = "libx264",  # 编码器
    crf: int = 23,  # 压缩等级
    fps: int = 24,  # 帧率
    verbose: bool = False,  # 是否输出详细信息
    keep_original_quality: bool = False,  # 新增参数
):
    # 如果输入视频元数据为空，则获取输入视频的元数据
    if in_metadata is None:
        in_metadata = get_video_metadata(in_path)

    # 断言输入视频元数据中包含视频流
    assert in_metadata.num_video_streams > 0, "no video stream present"

    # 获取输入视频的宽度和高度
    w, h = in_metadata.width, in_metadata.height
    # 断言宽度和高度不为空
    assert w is not None, "width not available"
    assert h is not None, "height not available"

    # 新增：如果保持原画质，直接用原始宽高和帧率
    if keep_original_quality:
        w = int(w)
        h = int(h)
    else:
        r = w / h
        if r < 1:
            h = min(720, h)
            w = h * r
        else:
            w = min(1280, w)
            h = w / r
        w = int(w)
        h = int(h)
        if w % 2 != 0:
            w += 1
        if h % 2 != 0:
            h += 1

    ffmpeg = shutil.which("ffmpeg")
    cmd = [
        ffmpeg,
        "-threads",
        f"{FFMPEG_NUM_THREADS}",  # global threads
        "-ss",
        f"{seek_t:.2f}",
        "-t",
        f"{max_time:.2f}",
        "-i",
        in_path,
        "-threads",
        f"{FFMPEG_NUM_THREADS}",  # decode (or filter..?) threads
        "-vf",
        f"fps={fps},scale={w}:{h},setsar=1:1",
        "-c:v",
        codec,
        "-crf",
        f"{crf}",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        f"{FFMPEG_NUM_THREADS}",  # encode threads
        out_path,
        "-y",
    ]
    if verbose:
        print(" ".join(cmd))

    subprocess.call(
        cmd,
        stdout=None if verbose else subprocess.DEVNULL,
        stderr=None if verbose else subprocess.DEVNULL,
    )
