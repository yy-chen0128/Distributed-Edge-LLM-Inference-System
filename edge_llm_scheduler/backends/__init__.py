"""edge_llm_scheduler.backends — 具体对接实现。

mock_*：无硬件验证用
vllm_engine / lmcache_storage / tcp_transport / wifi_transport：真实对接（有硬件时）
"""

from .mock_engine import MockEngine
from .mock_storage import MockKVStore
from .mock_transport import MockTransport
from .vllm_engine import VLLMEngine
from .lmcache_storage import LMCacheStore
from .tcp_transport import TCPTransport, TcpReceiver
from .wifi_transport import WiFiTransport

__all__ = [
    "MockEngine", "MockKVStore", "MockTransport",
    "VLLMEngine", "LMCacheStore", "TCPTransport", "TcpReceiver", "WiFiTransport",
]
