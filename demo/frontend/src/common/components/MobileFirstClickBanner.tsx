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
import {DEMO_SHORT_NAME} from '@/demo/DemoConfig';
import {spacing} from '@/theme/tokens.stylex';
import {ImageCopy} from '@carbon/icons-react';
import stylex from '@stylexjs/stylex';
import {Button} from 'react-daisyui';

const styles = stylex.create({
  container: {
    position: 'relative',
    backgroundColor: '#000',
    padding: spacing[5],
    paddingVertical: spacing[6],
    display: 'flex',
    flexDirection: 'column',
    gap: spacing[4],
  },
});

export default function MobileFirstClickBanner() {
  return (
    <div {...stylex.props(styles.container)}>
      <div className="flex text-white text-lg">
        点击视频中的对象开始
      </div>
      <div className="text-sm text-[#A7B3BF]">
        <p>
          您可以使用 {DEMO_SHORT_NAME} 对任何视频进行有趣编辑，通过跟踪对象和应用视觉效果。要开始，请点击视频中的任何对象。
          通过跟踪对象和应用视觉效果来播放视频。首先，单击视频中的任何对象。
        </p>
      </div>
      <div className="flex items-center">
        <ChangeVideoModal
          videoGalleryModalTrigger={MobileVideoGalleryModalTrigger}
          showUploadInGallery={true}
        />
      </div>
    </div>
  );
}

type MobileVideoGalleryModalTriggerProps = {
  onClick: () => void;
};

function MobileVideoGalleryModalTrigger({
  onClick,
}: MobileVideoGalleryModalTriggerProps) {
  return (
    <Button
      color="ghost"
      startIcon={<ImageCopy size={20} />}
      onClick={onClick}
      className="text-white p-0">
      更换视频
    </Button>
  );
}
