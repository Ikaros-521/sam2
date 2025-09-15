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
import ToolbarHeaderWrapper from '@/common/components/toolbar/ToolbarHeaderWrapper';
import {isStreamingAtom, streamingStateAtom} from '@/demo/atoms';
import {useAtomValue} from 'jotai';

export default function ObjectsToolbarHeader() {
  const isStreaming = useAtomValue(isStreamingAtom);
  const streamingState = useAtomValue(streamingStateAtom);

  return (
    <ToolbarHeaderWrapper
      title={
        streamingState === 'full'
          ? '查看跟踪对象'
          : isStreaming
            ? '正在跟踪对象'
            : '选择对象'
      }
      description={
        streamingState === 'full'
          ? '查看视频中选定的对象，如有需要可继续编辑。一切就绪后，点击"下一步"继续。'
          : isStreaming
            ? '仔细观察视频，查看对象跟踪不正确的地方。您也可以停止跟踪以进行额外编辑。'
            : '调整对象的选择，或添加其他对象。点击"跟踪对象"来跟踪整个视频中的对象。'
      }
      className="mb-8"
    />
  );
}
