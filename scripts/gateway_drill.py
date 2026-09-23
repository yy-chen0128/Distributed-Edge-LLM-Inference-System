"""End-to-end tier-0 + tier-1 drill through the gateway.

Sends one long streaming request to the gateway. Part way through, the vLLM server
is killed and restarted (simulating a node leaving and the pipeline being rebuilt).
The gateway should mark the pipeline down, wait for it to come back, then REPLAY the
request with the generated text appended to the prompt and forward only the NEW
text -- so the client's stream ends with a complete answer and no duplicated prefix.

ASCII only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.request


def reader(base_url: str, model: str, max_tokens: int, kill_after: float,
           restart_script: str) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 500 in words, one per line."}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        # force it to keep generating so the kill really lands mid-stream
        "ignore_eos": True,
    }).encode("utf-8")
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})

    state = {"text": "", "chunks": 0, "error": None, "chunks_before_kill": 0,
             "chunks_after_kill": 0, "kill_done_at": None, "killed": False}
    started = time.perf_counter()

    def killer() -> None:
        time.sleep(kill_after)
        print(f"[drill] t+{kill_after:.1f}s: killing vLLM and restarting it")
        subprocess.run(["bash", restart_script], check=False)
        state["killed"] = True
        state["kill_done_at"] = time.perf_counter() - started
        print(f"[drill] restart script finished at t+{state['kill_done_at']:.1f}s")

    threading.Thread(target=killer, daemon=True).start()

    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
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
                if chunk.get("error"):
                    state["error"] = str(chunk["error"])[:200]
                    break
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content") or ""
                    if not piece:
                        continue
                    state["text"] += piece
                    state["chunks"] += 1
                    if state["killed"]:
                        state["chunks_after_kill"] += 1
                    else:
                        state["chunks_before_kill"] += 1
    except Exception as exc:  # noqa: BLE001
        state["error"] = f"{type(exc).__name__}: {exc}"
    state["elapsed_s"] = time.perf_counter() - started
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--kill-after", type=float, default=3.0)
    parser.add_argument("--restart-script", default=".models/logs/vllm_restart.sh")
    args = parser.parse_args()

    print(f"[drill] sending a {args.max_tokens}-token streaming request through the gateway")
    result = reader(args.gateway_url, args.model, args.max_tokens, args.kill_after,
                    args.restart_script)

    text = result["text"]
    print()
    print("=== RESULT ===")
    print(f"  elapsed            : {result['elapsed_s']:.1f} s")
    print(f"  chunks before kill : {result['chunks_before_kill']}")
    print(f"  chunks after kill  : {result['chunks_after_kill']}")
    print(f"  total chars        : {len(text)}")
    print(f"  error              : {result['error']}")
    print(f"  first 120 chars    : {text[:120]!r}")
    print(f"  last 120 chars     : {text[-120:]!r}")
    # crude duplication check: does any 40-char window repeat in the middle?
    dup = False
    window = 40
    if len(text) > 4 * window:
        seen = {}
        for i in range(0, len(text) - window, 5):
            piece = text[i:i + window]
            if piece in seen and i - seen[piece] > window * 2:
                dup = True
                break
            seen[piece] = i
    print(f"  duplicated 40-char window found: {dup}")
    # NOTE: build the dict first. Putting a multi-line dict literal inside an
    # f-string is what broke this script (and an earlier one) at parse time.
    summary = {
        "chunks_before_kill": result["chunks_before_kill"],
        "chunks_after_kill": result["chunks_after_kill"],
        "chars": len(text),
        "duplicate": dup,
        "error": result["error"],
    }
    print(f"  JSON {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
