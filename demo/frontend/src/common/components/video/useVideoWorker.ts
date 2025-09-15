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
import {RefObject, useEffect, useMemo, useRef} from 'react';
import VideoWorkerBridge from './VideoWorkerBridge';

type Options = {
  createVideoWorker?: () => Worker;
  createWorkerBridge?: CreateWorkerBridgeFunction;
};

const DEFAULT_OPTIONS: Options = {
  createVideoWorker: () =>
    new Worker(new URL('./VideoWorker', import.meta.url), {
      type: 'module',
    }),
};

type WorkerFactory = () => Worker;

type CreateWorkerBridgeFunction = (
  workerFactory: WorkerFactory,
) => VideoWorkerBridge;

export default function useVideoWorker(
  src: string,
  canvasRef: RefObject<HTMLCanvasElement>,
  options: Options = {},
) {
  const isControlTransferredToOffscreenRef = useRef(false);
  const lastSrcRef = useRef<string | null>(null);

  const mergedOptions = useMemo(() => {
    // console.log(`[useVideoWorker] 重新计算mergedOptions`, options);
    const definedProps = (o: Options) =>
      Object.fromEntries(
        Object.entries(o).filter(([_k, v]) => v !== undefined),
      );
    return Object.assign(
      DEFAULT_OPTIONS,
      definedProps(options),
    ) as Required<Options>;
  }, [options.createVideoWorker, options.createWorkerBridge]);

  const worker = useMemo(() => {
    // console.log(`[useVideoWorker] 创建新的worker实例`);
    if (mergedOptions.createWorkerBridge) {
      return mergedOptions.createWorkerBridge(mergedOptions.createVideoWorker);
    }
    return VideoWorkerBridge.create(mergedOptions.createVideoWorker);
  }, [mergedOptions.createWorkerBridge, mergedOptions.createVideoWorker]);

  useEffect(() => {
    const canvas = canvasRef.current;

    if (canvas == null) {
      return;
    }

    if (isControlTransferredToOffscreenRef.current) {
      return;
    }

    isControlTransferredToOffscreenRef.current = true;

    // console.log(`[useVideoWorker] 设置canvas`);
    worker.setCanvas(canvas);

    return () => {
      // Cannot terminate worker in DEV mode
      // workerRef.current?.terminate();
    };
  }, [canvasRef, worker]);

  useEffect(() => {
    // 防止重复调用相同的src
    if (lastSrcRef.current === src) {
      console.log(`[useVideoWorker] 跳过重复的setSource调用: ${src}`);
      return;
    }
    
    console.log(`[useVideoWorker] 调用setSource: ${src}`);
    lastSrcRef.current = src;
    worker.setSource(src);
  }, [src, worker]);

  return worker;
}
