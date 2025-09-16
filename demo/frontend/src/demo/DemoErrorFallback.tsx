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
import LoadingStateScreen from '@/common/loading/LoadingStateScreen';
import {FallbackProps} from 'react-error-boundary';

export default function DemoErrorFallback(_props: FallbackProps) {
  return (
    <LoadingStateScreen
      title="好吧，这太尴尬了..."
      description="此演示未针对您的设备进行优化。请在屏幕更大的其他设备上重试。后端连接可能异常，请打开F12排查。"
      linkProps={{to: '..', label: '返回首页'}}
    />
  );
}
