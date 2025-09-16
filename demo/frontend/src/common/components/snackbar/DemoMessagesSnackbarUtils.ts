/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import {EnqueueOption} from '@/common/components/snackbar/useMessagesSnackbar';

export type MessageOptions = EnqueueOption & {
  repeat?: boolean;
};

type MessageEvent = {
  text: string;
  shown: boolean;
  action?: Element;
  options?: MessageOptions;
};

export interface MessagesEventMap {
  startSession: MessageEvent;
  firstClick: MessageEvent;
  pointClick: MessageEvent;
  addObjectClick: MessageEvent;
  trackAndPlayClick: MessageEvent;
  trackAndPlayComplete: MessageEvent;
  trackAndPlayThrottlingWarning: MessageEvent;
  effectsMessage: MessageEvent;
}

export const defaultMessageMap: MessagesEventMap = {
  startSession: {
    text: '正在启动会话',
    shown: false,
    options: {type: 'loading', showClose: false, repeat: true, duration: 2000},
  },
  firstClick: {
    text: '提示：点击视频中的任意对象开始操作。',
    shown: false,
    options: {expire: false, repeat: false},
  },
  pointClick: {
    text: '提示：不是您想要的结果？继续点击直到选中您想要的完整对象。',
    shown: false,
    options: {expire: false, repeat: false},
  },
  addObjectClick: {
    text: '提示：通过点击视频中的对象来添加新对象。',
    shown: false,
    options: {expire: false, repeat: false},
  },
  trackAndPlayClick: {
    text: '当你的物体被追踪时，请抓紧！在下一步中，您将能够应用视觉效果。如果跟踪看起来不正确，请随时停止跟踪以调整您的选择。',
    shown: false,
    options: {expire: false, repeat: false},
  },
  trackAndPlayComplete: {
    text: '提示：您可以通过返回跟踪不太正确的帧并添加或删除点击来修复跟踪问题。',
    shown: false,
    options: {expire: false, repeat: false},
  },
  trackAndPlayThrottlingWarning: {
    text: '看起来您点击跟踪按钮过于频繁！为了保持运行顺畅，我们暂时禁用了该按钮。',
    shown: false,
    options: {repeat: true},
  },
  effectsMessage: {
    text: '提示：如果您不确定从哪里开始，请点击"给我惊喜"为您的视频应用惊喜效果。',
    shown: false,
    options: {expire: false, repeat: false},
  },
};
