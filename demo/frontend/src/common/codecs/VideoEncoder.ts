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
import {ImageFrame} from '@/common/codecs/VideoDecoder';
import {MP4ArrayBuffer, createFile} from 'mp4box';

// The selection of timescale and seconds/key-frame value are
// explained in the following docs: https://github.com/vjeux/mp4-h264-re-encode
const TIMESCALE = 90000;
const SECONDS_PER_KEY_FRAME = 2;

export function encode(
  width: number,
  height: number,
  numFrames: number,
  framesGenerator: AsyncGenerator<ImageFrame, unknown>,
  progressCallback?: (progress: number) => void,
): Promise<MP4ArrayBuffer> {
  return new Promise(async (resolve, reject) => {
    let encodedFrameIndex = 0;
    let nextKeyFrameTimestamp = 0;
    let trackID: number | null = null;
    const durations: number[] = [];
    let totalDuration = 0; // 添加总时长跟踪
    let trackCreated = false; // 标记轨道是否已创建
    let firstTimestamp = 0;
    let isFirstFrame = true;

    const outputFile = createFile();

    const encoder = new VideoEncoder({
      output(chunk, metaData) {
        const uint8 = new Uint8Array(chunk.byteLength);
        chunk.copyTo(uint8);

        const description = metaData?.decoderConfig?.description;
        if (!trackCreated) {
          // 使用估算的duration创建轨道，后续会通过样本duration自动计算
          console.log(`[VideoEncoder] 创建轨道，使用估算时长`);
          
          trackID = outputFile.addTrack({
            width: width,
            height: height,
            timescale: TIMESCALE,
            avcDecoderConfigRecord: description,
          });
          trackCreated = true;
        }
        const shiftedDuration = durations.shift();
        console.log('shiftedDuration:', shiftedDuration);
        if (shiftedDuration != null && trackID !== null) {
          outputFile.addSample(trackID, uint8, {
            duration: getScaledDuration(shiftedDuration),
            is_sync: chunk.type === 'key',
          });
          encodedFrameIndex++;
          progressCallback?.(encodedFrameIndex / numFrames);
        }

        console.log(outputFile);

        if (encodedFrameIndex === numFrames) {
          console.log(`[VideoEncoder] 编码完成，总时长: ${totalDuration}微秒 (${totalDuration/1_000_000}秒)`);
          
          // 修复：确保 mp4box.js 写入 moov box，包含 duration 信息
          if (typeof outputFile.flush === 'function') outputFile.flush();
          
          // 获取最终的MP4缓冲区
          const buffer = outputFile.getBuffer();
          console.log(`[VideoEncoder] 生成的MP4文件大小: ${buffer.byteLength} 字节`);
          resolve(buffer);
        }
      },
      error(error) {
        reject(error);
        return;
      },
    });

    const setConfigurationAndEncodeFrames = async () => {
      // The codec value was taken from the following implementation and seems
      // reasonable for our use case for now:
      // https://github.com/vjeux/mp4-h264-re-encode/blob/main/mp4box.html#L103

      // Additional details about codecs can be found here:
      //  - https://developer.mozilla.org/en-US/docs/Web/Media/Formats/codecs_parameter
      //  - https://www.w3.org/TR/webcodecs-codec-registry/#video-codec-registry
      //
      // The following setting is a good compromise between output video file
      // size and quality. The latencyMode "realtime" is needed for Safari,
      // which otherwise will produce 20x larger files when in quality
      // latencyMode. Chrome does a really good job with file size even when
      // latencyMode is set to quality.
      const configuration: VideoEncoderConfig = {
        codec: 'avc1.4d0034',
        width: roundToNearestEven(width),
        height: roundToNearestEven(height),
        bitrate: 14_000_000,
        alpha: 'discard',
        bitrateMode: 'variable',
        latencyMode: 'realtime',
      };
      const supportedConfig =
        await VideoEncoder.isConfigSupported(configuration);
      if (supportedConfig.supported === true) {
        encoder.configure(configuration);
      } else {
        throw new Error(
          `Unsupported video encoder config ${JSON.stringify(supportedConfig)}`,
        );
      }

      for await (const frame of framesGenerator) {
        const {bitmap, duration, timestamp} = frame;
        
        // 规范化时间戳，确保从0开始
        let normalizedTimestamp = timestamp;
        if (isFirstFrame) {
          firstTimestamp = timestamp;
          isFirstFrame = false;
        }
        normalizedTimestamp = timestamp - firstTimestamp;
        
        durations.push(duration);
        totalDuration += duration; // 累加总时长
        
        let keyFrame = false;
        if (normalizedTimestamp >= nextKeyFrameTimestamp) {
          await encoder.flush();
          keyFrame = true;
          nextKeyFrameTimestamp = normalizedTimestamp + SECONDS_PER_KEY_FRAME * 1e6;
        }
        encoder.encode(bitmap, {keyFrame});
        bitmap.close();
      }

      await encoder.flush();
      encoder.close();
    };

    setConfigurationAndEncodeFrames();
  });
}

function getScaledDuration(rawDuration: number) {
  return rawDuration / (1_000_000 / TIMESCALE);
}

function roundToNearestEven(dim: number) {
  const rounded = Math.round(dim);

  if (rounded % 2 === 0) {
    return rounded;
  } else {
    return rounded + (rounded > dim ? -1 : 1);
  }
}
