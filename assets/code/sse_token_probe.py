#!/usr/bin/env python3
"""Stream chat requests with return_token_ids and count token IDs per SSE event.

Usage: sse_token_probe.py DELAY_S NUM_STREAMS MAX_TOKENS OUT_JSON
More than one token ID in an event means vLLM's RequestOutputCollector merged outputs.
"""
import asyncio, json, sys, time
import aiohttp

URL = "http://127.0.0.1:8000/v1/chat/completions"


async def one(session, max_tokens):
    body = {"model": "qwen3", "messages": [{"role": "user", "content": "probe"}],
            "max_tokens": max_tokens, "stream": True, "return_token_ids": True}
    sizes, gaps, last = [], [], None
    async with session.post(URL, json=body) as r:
        buf = b""
        async for chunk in r.content.iter_any():
            buf += chunk
            while b"\n\n" in buf:
                ev, buf = buf.split(b"\n\n", 1)
                if not ev.startswith(b"data: ") or ev == b"data: [DONE]":
                    continue
                d = json.loads(ev[6:])
                ch = d.get("choices") or []
                if not ch or ch[0].get("token_ids") is None:
                    continue
                now = time.perf_counter()
                if last is not None:
                    gaps.append((now - last) * 1e3)
                last = now
                sizes.append(len(ch[0]["token_ids"]))
    return sizes, gaps


async def main(delay, n, max_tokens, out):
    await asyncio.sleep(delay)
    async with aiohttp.ClientSession() as s:
        res = await asyncio.gather(*[one(s, max_tokens) for _ in range(n)])
    sizes = [x for r in res for x in r[0]]
    gaps = sorted(x for r in res for x in r[1])
    summary = {
        "events": len(sizes), "tokens": sum(sizes),
        "tok_per_event": round(sum(sizes) / len(sizes), 3) if sizes else None,
        "max_tok_per_event": max(sizes) if sizes else None,
        "frac_events_gt1": round(sum(1 for x in sizes if x > 1) / len(sizes), 3) if sizes else None,
        "gap_p50_ms": round(gaps[len(gaps) // 2], 2) if gaps else None,
        "gap_p99_ms": round(gaps[int(len(gaps) * 0.99)], 2) if gaps else None,
    }
    with open(out, "w") as f:
        json.dump(summary, f)


if __name__ == "__main__":
    asyncio.run(main(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))
