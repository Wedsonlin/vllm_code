# B1 · ZMQ REQ/REP 消息分帧仿真 / Queue-based framing

**课纲**: 模块 B · 原第 2 课 Engine/流式  
**目标**: 用 `queue.Queue` 仿真 REQ/REP（无需真实 pyzmq）。

## 验收

1. `frame_message(payload: bytes) -> bytes` 前缀 4 字节大端长度。
2. `unframe` 还原 payload。
3. `ReqRepBus.request` 客户端发帧、服务端回帧，往返一致。
