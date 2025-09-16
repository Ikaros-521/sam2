# SAM2 后端缓存清理机制

## 概述

为了解决视频处理完成后缓存未释放的问题，我们实现了自动和手动的缓存清理机制。这可以有效释放GPU内存，提高系统性能。

## 功能特性

### 1. 自动缓存清理
- **视频处理完成后自动清理**：在 `propagate_in_video` 方法完成后，自动清理非必要的缓存
- **智能保留**：保留必要的会话状态和条件帧输出，确保后续交互功能正常
- **GPU内存释放**：自动调用 `torch.cuda.empty_cache()` 释放GPU内存

### 2. 手动缓存清理
- **API端点**：提供 `/cleanup_cache` POST 端点手动清理所有会话缓存
- **内存统计**：提供 `/memory_stats` GET 端点查看内存使用情况
- **会话管理**：提供 `close_session` 方法完全清理特定会话

## API 端点

### 1. 清理缓存
```http
POST /cleanup_cache
```

**响应示例：**
```json
{
  "success": true,
  "message": "缓存清理完成",
  "stats_before": {
    "active_sessions": 2,
    "session_ids": ["session-1", "session-2"],
    "gpu_memory_allocated_mb": 1024,
    "gpu_memory_reserved_mb": 1536
  },
  "stats_after": {
    "active_sessions": 2,
    "session_ids": ["session-1", "session-2"],
    "gpu_memory_allocated_mb": 512,
    "gpu_memory_reserved_mb": 768
  }
}
```

### 2. 内存统计
```http
GET /memory_stats
```

**响应示例：**
```json
{
  "active_sessions": 2,
  "session_ids": ["session-1", "session-2"],
  "gpu_memory_allocated_mb": 512,
  "gpu_memory_reserved_mb": 768,
  "gpu_max_memory_allocated_mb": 2048,
  "gpu_max_memory_reserved_mb": 2560
}
```

## 清理策略

### 自动清理（视频处理完成后）
- ✅ 清理缓存的图像特征（保留最近一帧）
- ✅ 清理非条件帧的输出
- ✅ 清理临时输出字典
- ✅ 清理已跟踪帧的记录
- ✅ 释放GPU内存缓存
- ❌ 保留条件帧输出（用于后续交互）
- ❌ 保留会话基本状态
- ❌ 保留对象映射信息

### 完全清理（关闭会话时）
- ✅ 清理所有缓存数据
- ✅ 清理视频帧数据
- ✅ 清理所有输出字典
- ✅ 清理对象映射
- ✅ 释放GPU内存缓存
- ✅ 从会话状态字典中移除会话

## 使用方法

### 1. 自动清理
无需手动操作，视频处理完成后自动执行。

### 2. 手动清理
```python
import requests

# 清理所有会话缓存
response = requests.post("http://localhost:5000/cleanup_cache")
print(response.json())

# 查看内存使用情况
response = requests.get("http://localhost:5000/memory_stats")
print(response.json())
```

### 3. 测试脚本
运行测试脚本验证功能：
```bash
cd demo/backend/server
python test_cache_cleanup.py
```

## 监控和日志

系统会记录详细的清理日志：
- 缓存清理开始和完成
- GPU内存释放状态
- 清理的数据类型和数量
- 错误信息（如果有）

## 注意事项

1. **性能影响**：清理操作会短暂占用CPU时间，但能显著释放GPU内存
2. **交互功能**：自动清理会保留必要的状态，确保后续交互功能正常
3. **错误处理**：所有清理操作都有异常处理，不会影响主要功能
4. **内存监控**：建议定期检查内存使用情况，必要时手动清理

## 技术实现

### 核心方法
- `__cleanup_session_cache()`: 完全清理会话缓存
- `__cleanup_session_cache_after_propagation()`: 视频处理后的智能清理
- `cleanup_all_sessions_cache()`: 清理所有会话缓存
- `get_memory_usage_stats()`: 获取内存使用统计

### 清理的数据类型
- `cached_features`: 缓存的图像特征
- `images`: 视频帧数据
- `output_dict_per_obj`: 对象输出字典
- `temp_output_dict_per_obj`: 临时输出字典
- `frames_tracked_per_obj`: 已跟踪帧记录
- `point_inputs_per_obj`: 点输入数据
- `mask_inputs_per_obj`: 掩码输入数据
