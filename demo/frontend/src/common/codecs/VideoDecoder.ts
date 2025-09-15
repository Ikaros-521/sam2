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
import {cloneFrame} from '@/common/codecs/WebCodecUtils';
import {FileStream} from '@/common/utils/FileUtils';
import Logger from '@/common/logger/Logger';
import {
  createFile,
  DataStream,
  MP4ArrayBuffer,
  MP4File,
  MP4Sample,
  MP4VideoTrack,
} from 'mp4box';
// import {isAndroid, isChrome, isEdge, isWindows} from 'react-device-detect';

export type ImageFrame = {
  bitmap: VideoFrame;
  timestamp: number;
  duration: number;
};

export type DecodedVideo = {
  width: number;
  height: number;
  frames: ImageFrame[];
  numFrames: number;
  fps: number;
};

function decodeInternal(
  identifier: string,
  onReady: (mp4File: MP4File) => Promise<void>,
  onProgress: (decodedVideo: DecodedVideo) => void,
): Promise<DecodedVideo> {
  return new Promise((resolve, reject) => {
    Logger.info(`[VideoDecoder] 开始解码视频: ${identifier}`);
    console.log(`[VideoDecoder] 开始解码视频: ${identifier}`);
    const imageFrames: ImageFrame[] = [];
    const globalSamples: MP4Sample[] = [];

    let decoder: VideoDecoder;

    let track: MP4VideoTrack | null = null;
    const mp4File = createFile();

    mp4File.onError = (error) => {
      Logger.error(`[VideoDecoder] MP4文件错误:`, error);
      console.error(`[VideoDecoder] MP4文件错误:`, error);
      reject(error);
    };
    mp4File.onReady = async info => {
      Logger.info(`[VideoDecoder] MP4文件准备就绪，视频轨道数量: ${info.videoTracks.length}, 其他轨道数量: ${info.otherTracks?.length || 0}`);
      console.log(`[VideoDecoder] MP4文件准备就绪，视频轨道数量: ${info.videoTracks.length}, 其他轨道数量: ${info.otherTracks?.length || 0}`);
      
      if (info.videoTracks.length > 0) {
        track = info.videoTracks[0];
        Logger.info(`[VideoDecoder] 使用视频轨道: ${track.id}, 编码: ${track.codec}, 帧数: ${track.nb_samples}, 尺寸: ${track.track_width}x${track.track_height}`);
        console.log(`[VideoDecoder] 使用视频轨道: ${track.id}, 编码: ${track.codec}, 帧数: ${track.nb_samples}, 尺寸: ${track.track_width}x${track.track_height}`);
      } else {
        // The video does not have a video track, so looking if there is an
        // "otherTracks" available. Note, I couldn't find any documentation
        // about "otherTracks" in WebCodecs [1], but it was available in the
        // info for MP4V-ES, which isn't supported by Chrome [2].
        // However, we'll still try to get the track and then throw an error
        // further down in the VideoDecoder.isConfigSupported if the codec is
        // not supported by the browser.
        //
        // [1] https://www.w3.org/TR/webcodecs/
        // [2] https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Video_codecs#mp4v-es
        track = info.otherTracks[0];
        Logger.info(`[VideoDecoder] 使用其他轨道: ${track?.id}, 编码: ${track?.codec}`);
      }

      if (track == null) {
        Logger.error(`[VideoDecoder] ${identifier} 不包含视频轨道`);
        reject(new Error(`${identifier} does not contain a video track`));
        return;
      }

      const timescale = track.timescale;
      const edits = track.edits;

      let frame_n = 0;
      Logger.info(`[VideoDecoder] 创建VideoDecoder，总帧数: ${track.nb_samples}, 时间尺度: ${timescale}`);
      
      decoder = new VideoDecoder({
        // Be careful with any await in this function. The VideoDecoder will
        // not await output and continue calling it with decoded frames.
        async output(inputFrame) {
          if (track == null) {
            Logger.error(`[VideoDecoder] 轨道为空，无法解码帧 ${frame_n + 1}`);
            console.error(`[VideoDecoder] 轨道为空，无法解码帧 ${frame_n + 1}`);
            reject(new Error(`${identifier} does not contain a video track`));
            return;
          }
          
          Logger.debug(`[VideoDecoder] 解码帧 ${frame_n + 1}/${track.nb_samples}, 时间戳: ${inputFrame.timestamp}`);
          // console.log(`[VideoDecoder] 解码帧 ${frame_n + 1}/${track.nb_samples}, 时间戳: ${inputFrame.timestamp}`);

          const saveTrack = track;

          // If the track has edits, we'll need to check that only frames are
          // returned that are within the edit list. This can happen for
          // trimmed videos that have not been transcoded and therefore the
          // video track contains more frames than those visually rendered when
          // playing back the video.
          if (edits != null && edits.length > 0) {
            const cts = Math.round(
              (inputFrame.timestamp * timescale) / 1_000_000,
            );
            Logger.debug(`[VideoDecoder] 检查编辑列表，CTS: ${cts}, 媒体时间: ${edits[0].media_time}`);
            if (cts < edits[0].media_time) {
              Logger.debug(`[VideoDecoder] 跳过帧 ${frame_n + 1}，不在编辑范围内`);
              inputFrame.close();
              return;
            }
          }

          // Workaround for Chrome where the decoding stops at ~17 frames unless
          // the VideoFrame is closed. So, the workaround here is to create a
          // new VideoFrame and close the decoded VideoFrame.
          // The frame has to be cloned, or otherwise some frames at the end of the
          // video will be black. Note, the default VideoFrame.clone doesn't work
          // and it is using a frame cloning found here:
          // https://webcodecs-blogpost-demo.glitch.me/
          // 更可靠的Chrome检测
          const isChromeBrowser = /Chrome/.test(navigator.userAgent) && !/Edge/.test(navigator.userAgent);
          const isWindowsOS = /Windows/.test(navigator.userAgent);
          
          if (isChromeBrowser && isWindowsOS) {
            Logger.debug(`[VideoDecoder] 应用Chrome兼容性修复，克隆帧 ${frame_n + 1}`);
            // console.log(`[VideoDecoder] 应用Chrome兼容性修复，克隆帧 ${frame_n + 1}`);
            const clonedFrame = await cloneFrame(inputFrame);
            inputFrame.close();
            inputFrame = clonedFrame;
          } else {
            // console.log(`[VideoDecoder] 跳过Chrome兼容性修复，浏览器: ${navigator.userAgent}`);
            // console.log(`[VideoDecoder] Chrome检测: ${isChromeBrowser}, Windows检测: ${isWindowsOS}`);
          }

          const sample = globalSamples[frame_n];
          if (sample != null) {
            const duration = (sample.duration * 1_000_000) / sample.timescale;
            imageFrames.push({
              bitmap: inputFrame,
              timestamp: inputFrame.timestamp,
              duration,
            });
            Logger.debug(`[VideoDecoder] 添加帧 ${frame_n + 1} 到图像帧数组，当前总帧数: ${imageFrames.length}`);
            // 每帧都报告进度（仅控制台），让用户看到解码正在进行
            if (frame_n % 10 === 0 || frame_n < 20) {
              // console.log(`[VideoDecoder] 解码进度: ${frame_n + 1}/${saveTrack.nb_samples} 帧 (${((frame_n + 1) / saveTrack.nb_samples * 100).toFixed(1)}%)`);
            }
            
            // Sort frames in order of timestamp. This is needed because Safari
            // can return decoded frames out of order.
            imageFrames.sort((a, b) => (a.timestamp > b.timestamp ? 1 : -1));
            // Update progress on first frame and then every 5th frame for better responsiveness
            if (onProgress != null && (frame_n === 0 || frame_n % 5 === 0)) {
              // Logger.info(`[VideoDecoder] 报告进度: ${imageFrames.length}/${saveTrack.nb_samples} 帧已解码`);
              // console.log(`[VideoDecoder] 报告进度: ${imageFrames.length}/${saveTrack.nb_samples} 帧已解码`);
              onProgress({
                width: saveTrack.track_width,
                height: saveTrack.track_height,
                frames: imageFrames,
                numFrames: saveTrack.nb_samples,
                fps:
                  (saveTrack.nb_samples / saveTrack.duration) *
                  saveTrack.timescale,
              });
            }
          } else {
            Logger.warn(`[VideoDecoder] 帧 ${frame_n + 1} 没有对应的样本数据`);
            console.warn(`[VideoDecoder] 帧 ${frame_n + 1} 没有对应的样本数据`);
          }
          frame_n++;

          if (saveTrack.nb_samples === frame_n) {
            Logger.info(`[VideoDecoder] 所有帧解码完成！总帧数: ${imageFrames.length}/${saveTrack.nb_samples}`);
            console.log(`[VideoDecoder] 所有帧解码完成！总帧数: ${imageFrames.length}/${saveTrack.nb_samples}`);
            
            // Sort frames in order of timestamp. This is needed because Safari
            // can return decoded frames out of order.
            imageFrames.sort((a, b) => (a.timestamp > b.timestamp ? 1 : -1));
            
            // 解码完成，现在可以flush和close解码器
            Logger.info(`[VideoDecoder] 开始flush和关闭解码器`);
            console.log(`[VideoDecoder] 开始flush和关闭解码器`);
            await decoder.flush();
            decoder.close();
            Logger.info(`[VideoDecoder] 解码器已关闭`);
            console.log(`[VideoDecoder] 解码器已关闭`);
            
            const result = {
              width: saveTrack.track_width,
              height: saveTrack.track_height,
              frames: imageFrames,
              numFrames: saveTrack.nb_samples,
              fps:
                (saveTrack.nb_samples / saveTrack.duration) *
                saveTrack.timescale,
            };
            
            Logger.info(`[VideoDecoder] 解码完成，返回结果: ${result.frames.length} 帧, ${result.width}x${result.height}, ${result.fps.toFixed(2)} FPS`);
            console.log(`[VideoDecoder] 解码完成，返回结果: ${result.frames.length} 帧, ${result.width}x${result.height}, ${result.fps.toFixed(2)} FPS`);
            resolve(result);
          } else {
            // 检查解码器状态
            // console.log(`[VideoDecoder] 解码器状态: ${decoder.state}, 已解码: ${frame_n}/${saveTrack.nb_samples}`);
          }
        },
        error(error) {
          Logger.error(`[VideoDecoder] 解码器错误:`, error);
          console.error(`[VideoDecoder] 解码器错误:`, error);
          console.error(`[VideoDecoder] 解码器状态:`, decoder.state);
          console.error(`[VideoDecoder] 已解码帧数:`, frame_n);
          reject(error);
        },
      });

      let description;
      const trak = mp4File.getTrackById(track.id);
      const entries = trak?.mdia?.minf?.stbl?.stsd?.entries;
      if (entries == null) {
        return;
      }
      for (const entry of entries) {
        if (entry.avcC || entry.hvcC) {
          const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
          if (entry.avcC) {
            entry.avcC.write(stream);
          } else if (entry.hvcC) {
            entry.hvcC.write(stream);
          }
          description = new Uint8Array(stream.buffer, 8); // Remove the box header.
          break;
        }
      }

      const configuration: VideoDecoderConfig = {
        codec: track.codec,
        codedWidth: track.track_width,
        codedHeight: track.track_height,
        description,
      };
      const supportedConfig =
        await VideoDecoder.isConfigSupported(configuration);
      if (supportedConfig.supported == true) {
        Logger.info(`[VideoDecoder] 解码器配置支持，开始配置解码器`);
        console.log(`[VideoDecoder] 解码器配置支持，开始配置解码器`);
        decoder.configure(configuration);
        console.log(`[VideoDecoder] 解码器已配置，状态: ${decoder.state}`);

        Logger.info(`[VideoDecoder] 设置提取选项，开始提取样本`);
        console.log(`[VideoDecoder] 设置提取选项，开始提取样本`);
        mp4File.setExtractionOptions(track.id, null, {
          nbSamples: Infinity,
        });
        mp4File.start();
        console.log(`[VideoDecoder] MP4文件开始处理，等待样本...`);
      } else {
        Logger.error(`[VideoDecoder] 解码器配置不支持: ${JSON.stringify(supportedConfig.config)}`);
        reject(
          new Error(
            `Decoder config faile: config ${JSON.stringify(
              supportedConfig.config,
            )} is not supported`,
          ),
        );
        return;
      }
    };

    mp4File.onSamples = async (
      _id: number,
      _user: unknown,
      samples: MP4Sample[],
    ) => {
      Logger.debug(`[VideoDecoder] 收到样本批次，样本数量: ${samples.length}`);
      // console.log(`[VideoDecoder] 收到样本批次，样本数量: ${samples.length}`);
      for (const sample of samples) {
        globalSamples.push(sample);
        Logger.debug(`[VideoDecoder] 解码样本 ${globalSamples.length}, 时间戳: ${sample.cts}, 持续时间: ${sample.duration}, 同步: ${sample.is_sync}`);
        // console.log(`[VideoDecoder] 解码样本 ${globalSamples.length}, 时间戳: ${sample.cts}, 持续时间: ${sample.duration}, 同步: ${sample.is_sync}`);
        decoder.decode(
          new EncodedVideoChunk({
            type: sample.is_sync ? 'key' : 'delta',
            timestamp: (sample.cts * 1_000_000) / sample.timescale,
            duration: (sample.duration * 1_000_000) / sample.timescale,
            data: sample.data,
          }),
        );
      }
      Logger.debug(`[VideoDecoder] 样本批次处理完成，总样本数: ${globalSamples.length}`);
      console.log(`[VideoDecoder] 样本批次处理完成，总样本数: ${globalSamples.length}`);
      // 不要在这里关闭解码器，让它在所有帧解码完成后自动关闭
    };

    onReady(mp4File);
  });
}

export function decode(
  file: File,
  onProgress: (decodedVideo: DecodedVideo) => void,
): Promise<DecodedVideo> {
  return decodeInternal(
    file.name,
    async (mp4File: MP4File) => {
      const reader = new FileReader();
      reader.onload = function () {
        const result = this.result as MP4ArrayBuffer;
        if (result != null) {
          result.fileStart = 0;
          mp4File.appendBuffer(result);
        }
        mp4File.flush();
      };
      reader.readAsArrayBuffer(file);
    },
    onProgress,
  );
}

export function decodeStream(
  fileStream: FileStream,
  onProgress: (decodedVideo: DecodedVideo) => void,
): Promise<DecodedVideo> {
  return decodeInternal(
    'stream',
    async (mp4File: MP4File) => {
      console.log(`[VideoDecoder] 开始处理流数据`);
      let part = await fileStream.next();
      let partCount = 0;
      while (part.done === false) {
        partCount++;
        // console.log(`[VideoDecoder] 处理流数据部分 ${partCount}, 范围: ${part.value.range.start}-${part.value.range.end}, 数据大小: ${part.value.data.length}`);
        const result = part.value.data.buffer as MP4ArrayBuffer;
        if (result != null) {
          result.fileStart = part.value.range.start;
          mp4File.appendBuffer(result);
        }
        mp4File.flush();
        part = await fileStream.next();
      }
      console.log(`[VideoDecoder] 流数据处理完成，总共处理了 ${partCount} 个部分`);
    },
    onProgress,
  );
}
