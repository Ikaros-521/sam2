import argparse
# pip install opencv-python
import cv2
import numpy as np
import time
from pathlib import Path
from inference.predictor import InferenceAPI
from inference.data_types import AddPointsRequest, StartSessionRequest, PropagateInVideoRequest
from pycocotools.mask import decode as decode_mask

# ========== 命令行参数 ===========
parser = argparse.ArgumentParser(description='视频打点遮罩追踪Demo')
parser.add_argument('--video_path', type=str, default='../../data/gallery/05_default_juggle.mp4', help='视频路径')
parser.add_argument('--keep', type=str, choices=['foreground', 'background'], default='foreground', help='保留前景还是背景')
parser.add_argument('--green_screen', action='store_true', help='扣完转绿幕')
parser.add_argument('--save_output', action='store_true', help='是否保存输出视频')
parser.add_argument('--output_path', type=str, default='output_masked.mp4', help='输出视频路径')
parser.add_argument('--no_fps', action='store_true', help='不在画面上显示FPS（默认显示）')
args = parser.parse_args()

VIDEO_PATH = args.video_path
OBJECT_ID = 1
KEEP_FOREGROUND = args.keep == 'foreground'
GREEN_SCREEN = args.green_screen
SAVE_OUTPUT = args.save_output
OUTPUT_PATH = args.output_path
NO_FPS = args.no_fps

# ========== 初始化API和Session ===========
api = InferenceAPI()
video_abspath = str(Path(VIDEO_PATH).absolute())
session_id = api.start_session(StartSessionRequest(type='start_session', path=video_abspath)).session_id

# ========== 打点与遮罩交互 ===========
points = []
labels = []  # 新增labels，支持前景/背景点
mask = None
frame = None

cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame = cap.read()
if not ret:
    print('无法读取视频')
    exit()


def update_mask():
    global mask, points, labels, frame, api, session_id
    if not points:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        return
    h, w = frame.shape[:2]
    norm_points = [[x / w, y / h] for (x, y) in points]
    request = AddPointsRequest(
        type='add_points',
        session_id=session_id,
        frame_index=0,
        object_id=OBJECT_ID,
        points=norm_points,  # 归一化后的点
        labels=labels,
        clear_old_points=True
    )
    response = api.add_points(request)
    # 合并所有目标的mask（如果有多个object_id）
    mask_sum = np.zeros(frame.shape[:2], dtype=np.uint8)
    for r in response.results:
        mask_rle = {
            'counts': r.mask.counts,
            'size': r.mask.size
        }
        m = decode_mask(mask_rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        mask_sum = np.logical_or(mask_sum, m)
    mask = mask_sum.astype(np.uint8)


def mouse_callback(event, x, y, flags, param):
    global points, labels
    # 鼠标左键点击添加前景点
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        labels.append(1)  # 前景点
        update_mask()
    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append((x, y))
        labels.append(0)  # 背景点
        update_mask()


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_callback)
update_mask()

while True:
    temp = frame.copy()
    # 画点
    for pt, lb in zip(points, labels):
        color = (0, 0, 255) if lb == 1 else (255, 0, 0)  # 红色前景，蓝色背景
        cv2.circle(temp, pt, 5, color, -1)
    # 画遮罩
    if mask is not None and mask.shape == temp.shape[:2]:
        colored_mask = np.zeros_like(temp)
        colored_mask[mask > 0] = (0, 255, 0)
        temp = cv2.addWeighted(temp, 0.7, colored_mask, 0.3, 0)
    cv2.putText(temp, '左键前景 右键背景 c清空 回车确认', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow('frame', temp)
    key = cv2.waitKey(1)
    if key == 13:  # 回车
        break
    elif key == ord('c'):
        points = []
        labels = []
        update_mask()
    elif key == 27:
        exit()
cv2.destroyAllWindows()

# ========== 遮罩确认后，propagate全视频追踪并实时显示 ========== 
print('正在进行全视频追踪并实时显示...')
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0
start_time = time.time()

# 视频保存相关
out_writer = None
if SAVE_OUTPUT:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

for response in api.propagate_in_video(PropagateInVideoRequest(
    type='propagate_in_video',
    session_id=session_id,
    start_frame_index=0
)):
    ret, frame = cap.read()
    if not ret:
        break
    # 合并所有目标体的mask
    mask_sum = np.zeros(frame.shape[:2], dtype=np.uint8)
    for r in response.results:
        mask_rle = {
            'counts': r.mask.counts,
            'size': r.mask.size
        }
        m = decode_mask(mask_rle)
        if m.ndim == 3:
            m = m[:, :, 0]
        mask_sum = np.logical_or(mask_sum, m)
    mask = mask_sum.astype(np.uint8)

    # 保留前景还是背景
    if KEEP_FOREGROUND:
        show_mask = mask > 0
    else:
        show_mask = mask == 0

    # 绿幕处理
    if GREEN_SCREEN:
        masked = frame.copy()
        if KEEP_FOREGROUND:
            masked[~show_mask] = (0, 255, 0)
        else:
            masked[~show_mask] = (0, 255, 0)
    else:
        masked = np.zeros_like(frame)
        masked[show_mask] = frame[show_mask]

    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    if not NO_FPS:
        cv2.putText(masked, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('masked', masked)
    if SAVE_OUTPUT and out_writer is not None:
        out_writer.write(masked)
    if cv2.waitKey(1) == 27:
        break

cap.release()
if out_writer is not None:
    out_writer.release()
cv2.destroyAllWindows() 