#!/usr/bin/env python3
"""Split single-request latency on a running vLLM 0.32 server into the
engine core's share and the two residuals around it.

Usage: python split_latency.py --model <served-model-name>
"""
import argparse
import json
import statistics
import time
import urllib.request

from prometheus_client.parser import text_string_to_metric_families

# 1. Queue, prefill, decode and inference come from engine-core timestamps.
#    e2e comes from the API server's wall clock.
HISTOGRAMS = {
    "queue": "vllm:request_queue_time_seconds",
    "prefill": "vllm:request_prefill_time_seconds",
    "decode": "vllm:request_decode_time_seconds",
    "inference": "vllm:request_inference_time_seconds",
    "e2e": "vllm:e2e_request_latency_seconds",
}

# 2. (prompt tokens, output tokens). One output token leaves little besides
#    prefill and overhead. 256 output tokens show how far decode outweighs it.
SHAPES = [(128, 1), (2048, 1), (128, 256), (2048, 256)]
REPS = 20


def scrape(base):
    # 3. Read every histogram's _sum and _count. Summing across label sets
    #    folds in the model_name and engine labels.
    text = urllib.request.urlopen(f"{base}/metrics").read().decode()
    out = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            for suffix in ("_sum", "_count"):
                if sample.name.endswith(suffix):
                    key = (sample.name[: -len(suffix)], suffix)
                    out[key] = out.get(key, 0.0) + sample.value
    return out


def one_request(base, model, prompt_len, max_tokens):
    # 4. /v1/completions skips the chat template, so L1's input side is
    #    tokenization alone. " the" is close to one token per repeat on
    #    Llama- and Qwen-family tokenizers. The usage field has the exact count.
    body = {
        "model": model,
        "prompt": " the" * prompt_len,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,  # 5. vLLM extension: pin the output length.
    }
    req = urllib.request.Request(
        f"{base}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        usage = json.loads(resp.read())["usage"]
    return time.perf_counter() - t0, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    # 6. One warm-up request, so lazy imports and allocator growth stay out of
    #    the first shape's numbers.
    one_request(args.base, args.model, 128, 8)

    for prompt_len, max_tokens in SHAPES:
        rows = []
        for _ in range(REPS):
            before = scrape(args.base)
            client_s, usage = one_request(args.base, args.model, prompt_len, max_tokens)
            # 7. The response can go out before the stats logger records the
            #    finished request. Wait briefly, then prove one observation landed.
            time.sleep(0.2)
            after = scrape(args.base)
            for m in HISTOGRAMS.values():
                n = after.get((m, "_count"), 0) - before.get((m, "_count"), 0)
                assert n == 1, f"{m}: {n} observations between scrapes"
            d = {k: after.get((m, "_sum"), 0.0) - before.get((m, "_sum"), 0.0)
                 for k, m in HISTOGRAMS.items()}
            # 8. Tokenization, msgpack and ZMQ both ways, engine input-queue wait.
            d["frontend_ipc"] = d["e2e"] - d["queue"] - d["inference"]
            # 9. HTTP parsing, final detokenization, response write, socket.
            d["http_detok"] = client_s - d["e2e"]
            d["client"] = client_s
            rows.append(d)
        med = {k: statistics.median(r[k] for r in rows) * 1e3 for k in rows[0]}
        print(json.dumps({
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            **{f"{k}_ms": round(v, 3) for k, v in med.items()},
        }))


if __name__ == "__main__":
    main()
