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
import {Effects} from '@/common/components/video/effects/Effects';
import type {CarbonIconType} from '@carbon/icons-react';
import {
  AppleDash,
  Asterisk,
  Barcode,
  CenterCircle,
  ColorPalette,
  ColorSwitch,
  Development,
  Erase,
  FaceWink,
  Humidity,
  Image,
  Overlay,
  TextFont,
} from '@carbon/icons-react';

export type DemoEffect = {
  title: string;
  Icon: CarbonIconType;
  effectName: keyof Effects;
};

export const backgroundEffects: DemoEffect[] = [
  {title: '原始', Icon: Image, effectName: 'Original'},
  {title: '擦除', Icon: Erase, effectName: 'EraseBackground'},
  {
    title: '渐变',
    Icon: ColorPalette,
    effectName: 'Gradient',
  },
  {
    title: '像素化',
    Icon: Development,
    effectName: 'Pixelate',
  },
  {title: '去饱和', Icon: ColorSwitch, effectName: 'Desaturate'},
  {title: '文字', Icon: TextFont, effectName: 'BackgroundText'},
  {title: '模糊', Icon: Humidity, effectName: 'BackgroundBlur'},
  {title: '轮廓', Icon: AppleDash, effectName: 'Sobel'},
];

export const highlightEffects: DemoEffect[] = [
  {title: '原始', Icon: Image, effectName: 'Cutout'},
  {title: '擦除', Icon: Erase, effectName: 'EraseForeground'},
  {title: '渐变', Icon: ColorPalette, effectName: 'VibrantMask'},
  {title: '像素化', Icon: Development, effectName: 'PixelateMask'},
  {
    title: '叠加',
    Icon: Overlay,
    effectName: 'Overlay',
  },
  {title: '表情', Icon: FaceWink, effectName: 'Replace'},
  {title: '爆发', Icon: Asterisk, effectName: 'Burst'},
  {title: '聚光灯', Icon: CenterCircle, effectName: 'Scope'},
];

export const moreEffects: DemoEffect[] = [
  {title: '噪点', Icon: Barcode, effectName: 'NoisyMask'},
];
