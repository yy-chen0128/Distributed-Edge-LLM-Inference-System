"""入口网关的纯逻辑测试（不需要 fastapi / 真实 vLLM）。

覆盖档1 的两件核心事：
1. **拼接提示词**：把已生成内容拼回输入，构造重放请求体；
2. **只转发新增部分**：重放输出与已发送内容的对齐（含前缀不一致时的保守处理）。
另测档0 的状态机（停接纳 / 恢复接纳）。
"""

from __future__ import annotations

import pytest

from edge_llm_scheduler.gateway.vllm_gateway import (
    PipelineState,
    build_replay_body,
    extract_delta,
    extract_full,
    skip_already_sent,
)


# ------------------------------------------------------------------ 拼接提示词

def test_replay_body_appends_to_completion_prompt():
    """completions：prompt 是字符串 → 直接拼接。"""
    body = {"prompt": "你好，", "max_tokens": 32, "stream": True}
    replay = build_replay_body("/v1/completions", body, "世界")
    assert replay["prompt"] == "你好，世界"
    assert replay["max_tokens"] == 32
    # 不能改到原请求
    assert body["prompt"] == "你好，"


def test_replay_body_appends_assistant_message_for_chat():
    """chat：把已生成内容作为一条 assistant 消息追加。"""
    body = {"messages": [{"role": "user", "content": "讲个笑话"}], "stream": True}
    replay = build_replay_body("/v1/chat/completions", body, "从前有座山")
    assert replay["messages"][-1] == {"role": "assistant", "content": "从前有座山"}
    assert len(replay["messages"]) == 2
    assert len(body["messages"]) == 1


def test_replay_body_handles_list_prompt_and_stream_options():
    """多 prompt 只取首个；stream_options 不应带进重放。"""
    body = {"prompt": ["abc"], "stream": True, "stream_options": {"include_usage": True}}
    replay = build_replay_body("/v1/completions", body, "def")
    assert replay["prompt"] == "abcdef"
    assert "stream_options" not in replay


# ------------------------------------------------------------ 只转发新增部分

def test_skip_returns_only_new_text_when_prefix_matches():
    new, exact = skip_already_sent("你好世界，再见", "你好世界")
    assert (new, exact) == ("，再见", True)


def test_skip_is_noop_when_nothing_sent_yet():
    new, exact = skip_already_sent("完整输出", "")
    assert (new, exact) == ("完整输出", True)


def test_skip_flags_divergence_and_keeps_longest_common_prefix():
    """重放的前缀与已发送不一致（采样/分词差异）→ 标记不一致，从分歧处之后发。"""
    new, exact = skip_already_sent("你好地球，再见", "你好世界")
    assert exact is False
    assert new == "地球，再见"        # 公共前缀只有 "你好"，之后全部重发


# ------------------------------------------------------------------ 分片解析

def test_extract_delta_handles_both_endpoints():
    chat = {"choices": [{"delta": {"content": "abc"}}]}
    comp = {"choices": [{"text": "xyz"}]}
    assert extract_delta("/v1/chat/completions", chat) == "abc"
    assert extract_delta("/v1/completions", comp) == "xyz"
    assert extract_delta("/v1/completions", {}) == ""


def test_extract_full_handles_both_endpoints():
    chat = {"choices": [{"message": {"content": "hi"}}]}
    comp = {"choices": [{"text": "yo"}]}
    assert extract_full("/v1/chat/completions", chat) == "hi"
    assert extract_full("/v1/completions", comp) == "yo"


# ------------------------------------------------------------------ 档0 状态机

def test_state_drain_and_resume_gate_admission():
    state = PipelineState()
    assert state.available is True
    assert state.resume_event.is_set()

    state.mark_down("node B left")
    assert state.available is False
    assert not state.resume_event.is_set()
    assert state.snapshot()["reason"] == "node B left"

    # 重复 mark_down 不覆盖第一次的原因（保留首个故障）
    state.mark_down("something else")
    assert state.snapshot()["reason"] == "node B left"

    state.mark_up()
    assert state.available is True
    assert state.resume_event.is_set()
