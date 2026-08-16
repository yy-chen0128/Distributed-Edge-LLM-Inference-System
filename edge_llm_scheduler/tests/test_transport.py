"""验证：传输抽象（mock/TCP 的 push/pull/带宽测量/成本估计）。"""

from __future__ import annotations

import asyncio

import pytest

from edge_llm_scheduler.backends import MockTransport


@pytest.mark.asyncio
async def test_mock_push_pull():
    t = MockTransport(bandwidth_mbps=100.0, latency_ms=5.0)
    await t.push(b"hello-kv-block", "node_b", "kv_block:123")
    data = await t.pull("node_b", "kv_block:123")
    assert data == b"hello-kv-block"
    # 取过一次后没有了
    assert await t.pull("node_b", "kv_block:123") is None


@pytest.mark.asyncio
async def test_bandwidth_property():
    t = MockTransport(bandwidth_mbps=200.0)
    assert t.bandwidth_mbps == 200.0


@pytest.mark.asyncio
async def test_transfer_time_estimate():
    t = MockTransport(bandwidth_mbps=100.0, latency_ms=5.0)
    # 10MB @ 100MB/s = 100ms + 5ms 时延 = 105ms
    cost = await t.estimate_transfer_time(10 * 1024 * 1024, "node_b")
    assert 104.0 < cost < 106.0


@pytest.mark.asyncio
async def test_measure_bandwidth():
    t = MockTransport(bandwidth_mbps=150.0, latency_ms=1.0)
    b = await t.measure_bandwidth("node_c")
    assert b == 150.0


@pytest.mark.asyncio
async def test_packet_loss():
    t = MockTransport(loss_rate=1.0)  # 100% 丢包
    await t.push(b"x", "dst", "tag")
    assert await t.pull("dst", "tag") is None
