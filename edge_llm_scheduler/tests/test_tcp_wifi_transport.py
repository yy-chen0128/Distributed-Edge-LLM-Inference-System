"""集成测试：TCP/WiFi 传输（loopback 真实 socket，无 GPU）。

验证 TCPTransport + TcpReceiver 的真实网络路径，以及 WiFiTransport 的无线参数模拟。
"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.backends.tcp_transport import TCPTransport, TcpReceiver
from edge_llm_scheduler.backends.wifi_transport import WiFiTransport


@pytest.mark.asyncio
async def test_tcp_push_receiver_gets_data():
    """TCP 传输：push 到节点端的 TcpReceiver，数据被接收。"""
    receiver = TcpReceiver(host="127.0.0.1", port=0)
    # TcpReceiver 用 port=0 需拿到实际端口——手动 start_server 方式
    server = await asyncio.start_server(receiver._handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    receiver._server = server

    transport = TCPTransport(nodes={"node_b": ("127.0.0.1", port)})
    await transport.push(b"kv-payload-123", "node_b", "kv_block:42")

    # 给接收端一点时间；关闭传输连接 + 服务端
    await asyncio.sleep(0.1)
    await transport.close()
    server.close()
    await server.wait_closed()

    assert "kv_block:42" in receiver._store
    assert receiver._store["kv_block:42"] == b"kv-payload-123"


@pytest.mark.asyncio
async def test_tcp_transfer_time_estimate():
    transport = TCPTransport(nodes={}, bandwidth_mbps=1000.0)
    # 10MB @ 1000MB/s ≈ 10ms
    cost = await transport.estimate_transfer_time(10 * 1024 * 1024, "any")
    assert 9.0 < cost < 11.0


@pytest.mark.asyncio
async def test_wifi_transfer_time_includes_latency():
    wifi = WiFiTransport(nodes={}, bandwidth_mbps=30.0, latency_ms=20.0)
    # 传输时间 = 数据/带宽 + 时延
    cost = await wifi.estimate_transfer_time(1 * 1024 * 1024, "any")
    assert cost > 20.0  # 至少含时延


@pytest.mark.asyncio
async def test_wifi_bandwidth_measure_jitter():
    wifi = WiFiTransport(nodes={}, bandwidth_mbps=30.0)
    b = await wifi.measure_bandwidth("any")
    assert 20.0 <= b <= 40.0  # ±20% 抖动


@pytest.mark.asyncio
async def test_wifi_loss_drops_packet():
    """WiFi 100% 丢包：数据不会到达接收端。"""
    receiver = TcpReceiver(host="127.0.0.1", port=0)
    server = await asyncio.start_server(receiver._handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    wifi = WiFiTransport(nodes={"node_b": ("127.0.0.1", port)}, loss_rate=1.0)
    await wifi.push(b"x", "node_b", "lost_tag")
    await asyncio.sleep(0.1)
    await wifi.close()
    server.close()
    await server.wait_closed()

    assert "lost_tag" not in receiver._store
