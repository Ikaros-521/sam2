@echo off
REM Copyright (c) Meta Platforms, Inc. and affiliates.
REM All rights reserved.

REM 检查 curl 或 wget 是否存在
where curl >nul 2>nul
if %errorlevel%==0 (
    set "CMD=curl -L -O"
) else (
    where wget >nul 2>nul
    if %errorlevel%==0 (
        set "CMD=wget"
    ) else (
        echo 请先安装 curl 或 wget 以下载权重文件。
        exit /b 1
    )
)

REM 定义 SAM 2.1 权重文件的下载地址
set "SAM2p1_BASE_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824"
set "sam2p1_hiera_t_url=%SAM2p1_BASE_URL%/sam2.1_hiera_tiny.pt"
set "sam2p1_hiera_s_url=%SAM2p1_BASE_URL%/sam2.1_hiera_small.pt"
set "sam2p1_hiera_b_plus_url=%SAM2p1_BASE_URL%/sam2.1_hiera_base_plus.pt"
set "sam2p1_hiera_l_url=%SAM2p1_BASE_URL%/sam2.1_hiera_large.pt"

echo 正在下载 sam2.1_hiera_tiny.pt ...
%CMD% %sam2p1_hiera_t_url%
if %errorlevel% neq 0 (
    echo 下载 %sam2p1_hiera_t_url% 失败
    exit /b 1
)

echo 正在下载 sam2.1_hiera_small.pt ...
%CMD% %sam2p1_hiera_s_url%
if %errorlevel% neq 0 (
    echo 下载 %sam2p1_hiera_s_url% 失败
    exit /b 1
)

echo 正在下载 sam2.1_hiera_base_plus.pt ...
%CMD% %sam2p1_hiera_b_plus_url%
if %errorlevel% neq 0 (
    echo 下载 %sam2p1_hiera_b_plus_url% 失败
    exit /b 1
)

echo 正在下载 sam2.1_hiera_large.pt ...
%CMD% %sam2p1_hiera_l_url%
if %errorlevel% neq 0 (
    echo 下载 %sam2p1_hiera_l_url% 失败
    exit /b 1
)

echo 所有权重文件已成功下载。