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
import ChangeVideoModal from '@/common/components/gallery/ChangeVideoModal';
import type {VideoGalleryTriggerProps} from '@/common/components/gallery/DemoVideoGalleryModal';
import LoadingStateScreen from '@/common/loading/LoadingStateScreen';
import {uploadingStateAtom} from '@/demo/atoms';
import {ImageCopy} from '@carbon/icons-react';
import {useAtomValue} from 'jotai';
import OptionButton from '../components/options/OptionButton';

export default function UploadLoadingScreen() {
  const uploadingState = useAtomValue(uploadingStateAtom);

  if (uploadingState === 'error') {
    return (
      <LoadingStateScreen
        title="哎呀，我们无法处理这个视频"
        description="请上传另一个视频，并确保视频文件大小小于 70MB。">
        <div className="max-w-[250px] w-full mx-auto">
          <ChangeVideoModal
            videoGalleryModalTrigger={UploadLoadingScreenChangeVideoTrigger}
          />
        </div>
      </LoadingStateScreen>
    );
  }

  return (
    <LoadingStateScreen
      title="正在上传视频..."
      description="请稍等，我们正在上传您的视频。"
    />
  );
}

function UploadLoadingScreenChangeVideoTrigger({
  onClick,
}: VideoGalleryTriggerProps) {
  return (
    <OptionButton
      variant="gradient"
      title="更换视频"
      Icon={ImageCopy}
      onClick={onClick}
    />
  );
}
