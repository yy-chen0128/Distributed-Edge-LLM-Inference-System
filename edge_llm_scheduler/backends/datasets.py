"""数据集加载器：把论文负载归一化成统一请求格式。

统一格式：
    DatasetRequest = {
        "request_id": str,
        "arrival_time": float,
        "input_ids": List[int],          # 完整 prompt（模拟推理用）
        "prefix_id": str,                # 前缀标识（共享请求同 id → 可命中）
        "suffix_ids": List[int],         # 独特部分
        "max_new_tokens": int,
        "raw": Any,                      # 原始数据
    }

来源：
- MoE-Infinity contextpilot JSON fixtures（共享前缀最清晰）
- Preble 风格（多前缀 + 冷请求混合）
- 合成生成（指定共享比例/长 prompt）

token 化简化：prompt 文本按字符取 hash 作为 token（无真实分词器）。
prefix_id 相同的请求共享同一段前缀 → KVStore.lookup 可命中。
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _text_to_ids(text: str) -> List[int]:
    """文本 → 伪 token id 序列（按字符 hash，无真实分词器）。"""
    return [int(hashlib.md5(ch.encode()).hexdigest()[:8], 16) for ch in text]


class DatasetLoader:
    """统一数据集加载器。"""

    def __init__(self) -> None:
        self.requests: List[Dict] = []
        self._prefix_counter = 0

    # ---------- 加载来源 ----------

    def load_contextpilot_json(self, path: str) -> None:
        """加载 MoE-Infinity contextpilot JSON fixture。

        前缀提取：第一条请求的 messages 里，与前一条重复的 system content 视为共享前缀。
        简化：把整条请求的 messages 文本 hash 作为前缀标识——同 overlap 的请求共享。
        更精确的做法是按 overlap_ratio 切分共享/独特部分。
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        requests = data.get("requests", [])
        for i, r in enumerate(requests):
            messages = r.get("messages", [])
            text = "\n".join(m["content"] for m in messages)
            prefix_id = f"ctx_{data.get('metadata', {}).get('name', 'unknown')}"
            req = {
                "request_id": str(uuid.uuid4()),
                "arrival_time": i * 0.1,
                "input_ids": _text_to_ids(text),
                "prefix_id": prefix_id,
                "suffix_ids": _text_to_ids(text)[-20:],  # 简化：末尾独特部分
                "max_new_tokens": r.get("expected_token_count", 32),
                "raw": r,
            }
            self.requests.append(req)
        logger.info(f"loaded {len(requests)} requests from {path}")

    def load_preble_workload(self, num_prefixes: int = 3, reqs_per_prefix: int = 5,
                             cold_ratio: float = 0.2, context_len: int = 512) -> None:
        """生成 Preble 风格的"多共享前缀 + 冷请求"负载。

        num_prefixes: 共享前缀数
        reqs_per_prefix: 每个前缀的请求数
        cold_ratio: 冷请求（无共享前缀）比例
        context_len: 共享前缀长度
        """
        hot_count = int(reqs_per_prefix * num_prefixes * (1 - cold_ratio))
        cold_count = int(reqs_per_prefix * num_prefixes * cold_ratio)

        # 生成共享前缀
        prefixes = {}
        for p in range(num_prefixes):
            prefix_text = " ".join(f"shared_context_{p}_{i}" for i in range(context_len // 4))
            prefixes[p] = prefix_text

        reqs = []
        rid = 0
        for p in range(num_prefixes):
            for _ in range(hot_count // max(1, num_prefixes)):
                suffix = f" unique_request_{rid} " + "x" * random.randint(10, 50)
                reqs.append({
                    "request_id": f"preble-{rid}",
                    "arrival_time": rid * 0.05,
                    "input_ids": _text_to_ids(prefixes[p] + suffix),
                    "prefix_id": f"hot_{p}",
                    "suffix_ids": _text_to_ids(suffix),
                    "max_new_tokens": 16,
                    "raw": None,
                })
                rid += 1
        for _ in range(cold_count):
            text = "cold " + "y" * random.randint(100, 300)
            reqs.append({
                "request_id": f"preble-{rid}",
                "arrival_time": rid * 0.05,
                "input_ids": _text_to_ids(text),
                "prefix_id": f"cold_{rid}",
                "suffix_ids": _text_to_ids(text),
                "max_new_tokens": 16,
                "raw": None,
            })
            rid += 1

        self.requests.extend(reqs)
        logger.info(f"generated {len(reqs)} preble-style requests")

    def load_node_trace(self, path: str) -> List[Dict]:
        """加载 SpotServe 风格的节点事件 trace。

        格式每行 JSON：[tstamp_ms, "add"/"remove"/"DONE", {"nodes": [...]}]
        返回归一化节点事件。
        """
        events = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    arr = json.loads(line)
                    if len(arr) >= 3 and isinstance(arr[2], dict):
                        events.append({
                            "time_ms": arr[0],
                            "action": arr[1],
                            "nodes": arr[2].get("nodes", []),
                        })
                except json.JSONDecodeError:
                    continue
        logger.info(f"loaded {len(events)} node events from {path}")
        return events

    # ---------- 工具 ----------

    def generate_synthetic(self, num_requests: int = 50, shared_prefix_tokens: int = 2048,
                           suffix_tokens: int = 32, num_prefixes: int = 2) -> None:
        """合成负载：指定共享前缀长度的长 prompt + 小后缀。"""
        prefixes = {}
        for p in range(num_prefixes):
            prefixes[p] = " ".join(f"prefix_{p}_{i}" for i in range(shared_prefix_tokens // 6))
        for i in range(num_requests):
            p = i % num_prefixes
            suffix = "q" * suffix_tokens
            self.requests.append({
                "request_id": f"syn-{i}",
                "arrival_time": i * 0.02,
                "input_ids": _text_to_ids(prefixes[p] + suffix),
                "prefix_id": f"syn_{p}",
                "suffix_ids": _text_to_ids(suffix),
                "max_new_tokens": 16,
                "raw": None,
            })

    def clear(self) -> None:
        self.requests.clear()

    def count(self) -> int:
        return len(self.requests)

    def group_by_prefix(self) -> Dict[str, List[Dict]]:
        """按 prefix_id 分组（统计共享度）。"""
        groups: Dict[str, List[Dict]] = {}
        for r in self.requests:
            groups.setdefault(r["prefix_id"], []).append(r)
        return groups
