# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Generator

from app_conf import (
    GALLERY_PATH,
    GALLERY_PREFIX,
    POSTERS_PATH,
    POSTERS_PREFIX,
    UPLOADS_PATH,
    UPLOADS_PREFIX,
)
from data.loader import preload_data
from data.schema import schema
from data.store import set_videos
from flask import Flask, make_response, Request, request, Response, send_from_directory
from flask_cors import CORS
from inference.data_types import PropagateDataResponse, PropagateInVideoRequest
from inference.multipart import MultipartResponseBuilder
from inference.predictor import InferenceAPI
from strawberry.flask.views import GraphQLView

logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
# 创建CORS对象，允许跨域请求
cors = CORS(app, supports_credentials=True, origins="*")

# 预加载数据
videos = preload_data()
# 设置视频数据
set_videos(videos)

# 创建InferenceAPI对象
inference_api = InferenceAPI()


# 定义健康检查路由
@app.route("/healthy")
def healthy() -> Response:
    # 返回OK状态码
    return make_response("OK", 200)


# 定义发送画廊视频的路由
@app.route(f"/{GALLERY_PREFIX}/<path:path>", methods=["GET"])
def send_gallery_video(path: str) -> Response:
    try:
        # 从GALLERY_PATH目录下发送文件
        return send_from_directory(
            GALLERY_PATH,
            path,
        )
    except:
        # 如果文件不存在，抛出ValueError异常
        raise ValueError("resource not found")


# 定义发送海报图片的路由
@app.route(f"/{POSTERS_PREFIX}/<path:path>", methods=["GET"])
def send_poster_image(path: str) -> Response:
    try:
        # 从POSTERS_PATH目录下发送文件
        return send_from_directory(
            POSTERS_PATH,
            path,
        )
    except:
        # 如果文件不存在，抛出ValueError异常
        raise ValueError("resource not found")


# 定义发送上传视频的路由
@app.route(f"/{UPLOADS_PREFIX}/<path:path>", methods=["GET"])
def send_uploaded_video(path: str):
    try:
        # 从UPLOADS_PATH目录下发送文件
        return send_from_directory(
            UPLOADS_PATH,
            path,
        )
    except:
        # 如果文件不存在，抛出ValueError异常
        raise ValueError("resource not found")


# TOOD: Protect route with ToS permission check
@app.route("/propagate_in_video", methods=["POST"])
def propagate_in_video() -> Response:
    # 获取请求中的json数据
    data = request.json
    # 构造参数字典
    args = {
        "session_id": data["session_id"],
        "start_frame_index": data.get("start_frame_index", 0),
    }

    # 定义边界
    boundary = "frame"
    # 生成带有掩码的跟踪流
    frame = gen_track_with_mask_stream(boundary, **args)
    # 返回响应
    return Response(frame, mimetype="multipart/x-savi-stream; boundary=" + boundary)


def gen_track_with_mask_stream(
    boundary: str,
    session_id: str,
    start_frame_index: int,
) -> Generator[bytes, None, None]:
    # 使用autocast_context上下文
    with inference_api.autocast_context():
        # 构造请求
        request = PropagateInVideoRequest(
            type="propagate_in_video",
            session_id=session_id,
            start_frame_index=start_frame_index,
        )

        # 遍历请求的块
        for chunk in inference_api.propagate_in_video(request=request):
            # 构造响应
            yield MultipartResponseBuilder.build(
                boundary=boundary,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "Frame-Current": "-1",
                    # Total frames minus the reference frame
                    "Frame-Total": "-1",
                    "Mask-Type": "RLE[]",
                },
                body=chunk.to_json().encode("UTF-8"),
            ).get_message()


class MyGraphQLView(GraphQLView):
    def get_context(self, request: Request, response: Response) -> Any:
        return {"inference_api": inference_api}


# Add GraphQL route to Flask app.
app.add_url_rule(
    "/graphql",
    view_func=MyGraphQLView.as_view(
        "graphql_view",
        schema=schema,
        # Disable GET queries
        # https://strawberry.rocks/docs/operations/deployment
        # https://strawberry.rocks/docs/integrations/flask
        allow_queries_via_get=False,
        # Strawberry recently changed multipart request handling, which now
        # requires enabling support explicitly for views.
        # https://github.com/strawberry-graphql/strawberry/issues/3655
        multipart_uploads_enabled=True,
    ),
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
