"""Prefix-cache / TTFT bench for an OpenAI-compatible vLLM server.

Sends K requests that share a long prefix and reports TTFT + total time for each,
so the effect of the prefix cache (built-in or LMCache) is directly visible:
request 1 is cold, requests 2..K should be warm.

ASCII only. No third-party deps (urllib + json).

Usage:
    python scripts/vllm_bench_prefix.py --base-url http://127.0.0.1:8000 \
        --prefix-tokens 3500 --requests 3 --label plain
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def build_prefix(prefix_tokens: int) -> str:
    """Approximate a target token count.

    IMPORTANT: keep the word roughly 1:1 with tokens. An earlier version used
    'ctx00000'-style words, which tokenise into several tokens each, so a
    "3000-token" prefix was really >8192 tokens and the server correctly answered
    HTTP 400. ' the' is ~1 token per repeat for Qwen-class tokenizers, and the
    real count is reported back from the response usage.
    """
    return "".join(" the" for _ in range(prefix_tokens))


def stream_completion(base_url: str, model: str, prompt: str, max_tokens: int,
                      timeout: float) -> tuple[float, float, int, int]:
    """Return (ttft_ms, total_ms, completion_chars, prompt_tokens)."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    ttft = None
    text_len = 0
    prompt_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = chunk.get("usage") or {}
                if usage.get("prompt_tokens"):
                    prompt_tokens = int(usage["prompt_tokens"])
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("text") or ""
                if text and ttft is None:
                    ttft = (time.perf_counter() - started) * 1000.0
                text_len += len(text)
    except urllib.error.HTTPError as exc:
        # print the server's explanation instead of just "HTTP Error 400"
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise SystemExit(f"HTTP {exc.code} from {base_url}/v1/completions:\n{detail}")
    total = (time.perf_counter() - started) * 1000.0
    return (ttft if ttft is not None else total), total, text_len, prompt_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="")
    parser.add_argument("--prefix-tokens", type=int, default=3500)
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    model = args.model
    if not model:
        with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=30) as resp:
            model = json.loads(resp.read())["data"][0]["id"]

    prefix = build_prefix(args.prefix_tokens)
    print(f"[{args.label}] model={model} shared-prefix words={args.prefix_tokens} "
          f"requests={args.requests}")
    print(f"[{args.label}] {'#':>2} {'TTFT ms':>10} {'total ms':>10} {'chars':>7} "
          f"{'prompt_tok':>10}  note")

    results = []
    for i in range(args.requests):
        # same long prefix, a different short suffix each time
        prompt = f"{prefix}\nQuestion {i}: reply with one short sentence."
        ttft, total, chars, ptok = stream_completion(
            args.base_url, model, prompt, args.max_tokens, args.timeout)
        note = "cold (first)" if i == 0 else ("warm (prefix cached)" if i == 1 else "")
        results.append((ttft, total, chars, ptok))
        print(f"[{args.label}] {i:>2} {ttft:>10.1f} {total:>10.1f} {chars:>7} "
              f"{ptok:>10}  {note}")

    if len(results) >= 2 and results[0][0] > 0:
        speedup = results[0][0] / max(1e-9, min(r[0] for r in results[1:]))
        print(f"[{args.label}] first-request TTFT / best-warm TTFT = {speedup:.1f}x")
    summary = {
        "label": args.label,
        "ttft_ms": [round(r[0], 1) for r in results],
        "total_ms": [round(r[1], 1) for r in results],
        "prompt_tokens": results[0][3],
    }
    print(f"[{args.label}] JSON {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
