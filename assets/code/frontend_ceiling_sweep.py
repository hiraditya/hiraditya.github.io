#!/usr/bin/env python3
"""Frontend-ceiling sweep: a vLLM frontend + the (patched) Rust mock engine + vllm-bench.

For each frontend config, start the frontend and the mock engine once, then run one
closed-loop vllm-bench point per concurrency. Around each point, record CPU seconds and
peak RSS for every frontend process, vLLM's own TTFT histogram, and the client's
throughput over the window where the target concurrency was in flight.

Used for "A Request's Life in vLLM V1" (vLLM 0.32.0, commit 5df63a7). Expected layout
next to this script:
  vllm/                 the vLLM checkout, built for CPU into venv/, with
                        mock-engine-step-interval.patch applied and
                        `cargo build --release -p vllm-mock-engine -p vllm-bench
                        --bin vllm-rs --features native-tls-vendored` run in vllm/rust
  venv/                 the Python environment vLLM was installed into
  models/Qwen3-0.6B/    config and tokenizer files only; no weights are loaded
  sse_token_probe.py    the return_token_ids probe

Example: python frontend_ceiling_sweep.py --kind py --api-servers 1 --conc 128,256,512
"""
import argparse
import json
import os
import pathlib
import signal
import subprocess
import threading
import time
import urllib.request

import psutil

P2 = pathlib.Path(__file__).resolve().parent
RUST = P2 / "vllm/rust/target/release"
VENV = P2 / "venv/bin"
MODEL = P2 / "models/Qwen3-0.6B"
OUT = P2 / os.environ.get("SWEEP_OUT", "results")
PORT = 8000
PROBE = True


def wait_ready(timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("frontend not ready")


def frontend_cmd(kind, api_servers):
    if kind == "py":
        return [str(VENV / "vllm"), "serve", str(MODEL), "--served-model-name", "qwen3",
                "--host", "127.0.0.1", "--port", str(PORT),
                "--data-parallel-address", "127.0.0.1", "--data-parallel-rpc-port", "29550",
                "--data-parallel-size", "1", "--data-parallel-size-local", "0",
                "--api-server-count", str(api_servers), "--max-model-len", "4096"]
    return [str(RUST / "vllm-rs"), "serve", str(MODEL), "--served-model-name", "qwen3",
            "--host", "127.0.0.1", "--port", str(PORT),
            "--data-parallel-size", "1", "--data-parallel-size-local", "0",
            "--handshake-port", "29550", "--max-model-len", "4096"]


def tree(pid):
    try:
        p = psutil.Process(pid)
        return [p] + p.children(recursive=True)
    except psutil.NoSuchProcess:
        return []


def label(p):
    try:
        c = p.cmdline()
        return (c[0] if c else p.name())[:40]
    except psutil.Error:
        return "?"


def scrape_ttft():
    """Sum and count of vLLM's own vllm:time_to_first_token_seconds histogram."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return None
    tot = {"sum": 0.0, "count": 0.0}
    for line in text.splitlines():
        for k in tot:
            if line.startswith(f"vllm:time_to_first_token_seconds_{k}"):
                tot[k] += float(line.rsplit(" ", 1)[1])
    return tot


def cpu_snapshot(procs):
    snap = {}
    for p in procs:
        try:
            t = p.cpu_times()
            snap[p.pid] = (label(p), t.user + t.system)
        except psutil.Error:
            pass
    return snap


class RssSampler(threading.Thread):
    def __init__(self, procs):
        super().__init__(daemon=True)
        self.procs, self.peak, self.stop = procs, {}, threading.Event()

    def run(self):
        while not self.stop.is_set():
            for p in self.procs:
                try:
                    rss = p.memory_info().rss
                    self.peak[p.pid] = max(self.peak.get(p.pid, 0), rss)
                except psutil.Error:
                    pass
            time.sleep(0.2)


def plateau_throughput(res, conc, frac=0.95):
    """Client tokens/s over the window where in-flight requests >= frac * conc."""
    starts, lats = res.get("start_times") or [], res.get("latencies") or []
    edges = sorted([(s, 1) for s in starts] + [(s + l, -1) for s, l in zip(starts, lats)])
    inflight, t_open, t_close = 0, None, None
    for t, d in edges:
        inflight += d
        if inflight >= frac * conc:
            t_open = t if t_open is None else t_open
            t_close = None
        elif t_open is not None and t_close is None:
            t_close = t
    if t_open is None:
        return None
    t_close = t_close or max(s + l for s, l in zip(starts, lats))
    toks = 0.0
    for s, ttft, itl, n in zip(starts, res["ttfts"], res["itls"], res["output_lens"]):
        if not n:
            continue
        per_event = n / (len(itl) + 1)
        t = s + ttft
        toks += per_event if t_open <= t <= t_close else 0
        for g in itl:
            t += g
            toks += per_event if t_open <= t <= t_close else 0
    win = t_close - t_open
    return {"window_s": round(win, 2), "tok_s": round(toks / win) if win > 0 else None}


def run_point(tag, conc, nprompts, out_len, in_len, fe_procs, eng_procs):
    name = f"{tag}_c{conc}"
    cmd = ["/usr/bin/time", "-p", str(RUST / "vllm-bench"), "--backend", "openai-chat",
           "--base-url", f"http://127.0.0.1:{PORT}", "--model", "qwen3",
           "--tokenizer", str(MODEL), "--dataset-name", "random",
           "--random-input-len", str(in_len), "--random-output-len", str(out_len),
           "--ignore-eos", "--num-prompts", str(nprompts), "--max-concurrency", str(conc),
           "--save-result", "--save-detailed", "--result-dir", str(OUT),
           "--result-filename", f"{name}.json", "--disable-tqdm"]
    if conc > 1:
        # macOS caps the listen backlog at kern.ipc.somaxconn = 128, so a closed loop that
        # opens C connections in the same instant gets refused above ~128. Spread arrivals at
        # twice the rate a 10 ms/token, 256-token request needs; the semaphore still holds C.
        cmd += ["--request-rate", str(round(2 * conc / 2.56, 1))]
    sampler = RssSampler(fe_procs)
    sampler.start()
    fe0, en0 = cpu_snapshot(fe_procs), cpu_snapshot(eng_procs)
    m0 = scrape_ttft()
    t0 = time.time()
    probe = None
    if PROBE and conc >= 32:
        # The post's diagnostic: 4 extra streams with return_token_ids, started once the
        # closed loop is at full concurrency.
        probe = subprocess.Popen([str(VENV / "python"), str(P2 / "sse_token_probe.py"), "3", "4",
                                  str(out_len), str(OUT / f"{name}.probe.json")])
    with open(OUT / f"{name}.bench.log", "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    wall = time.time() - t0
    if probe:
        probe.wait(timeout=600)
    fe1, en1 = cpu_snapshot(fe_procs), cpu_snapshot(eng_procs)
    m1 = scrape_ttft()
    vllm_ttft_ms = (round((m1["sum"] - m0["sum"]) / (m1["count"] - m0["count"]) * 1e3, 1)
                    if m0 and m1 and m1["count"] > m0["count"] else None)
    sampler.stop.set()
    sampler.join()
    res = json.loads((OUT / f"{name}.json").read_text())
    tl = dict(l.split() for l in (OUT / f"{name}.bench.log").read_text().splitlines()
              if l.split()[:1] in (["user"], ["sys"]) and len(l.split()) == 2)
    client_cpu = round(float(tl.get("user", 0)) + float(tl.get("sys", 0)), 2)
    itls = res.get("itls") or []
    plateau = plateau_throughput(res, conc)
    out_lens = res.get("output_lens") or []
    events = sum(len(x) + 1 for x, n in zip(itls, out_lens) if n)
    toks = sum(out_lens)
    fe_cpu = {f"{fe1[k][0]}[{k}]": round(fe1[k][1] - fe0.get(k, (None, 0))[1], 2) for k in fe1}
    row = {
        "tag": tag, "conc": conc, "nprompts": nprompts, "wall_s": round(wall, 1),
        "completed": res.get("completed"), "failed": res.get("failed"),
        "plateau": plateau, "out_tok_s": round(res.get("output_throughput", 0)),
        "mean_tpot_ms": round(res.get("mean_tpot_ms", 0), 2),
        "p99_tpot_ms": round(res.get("p99_tpot_ms", 0), 2),
        "mean_itl_ms": round(res.get("mean_itl_ms", 0), 2),
        "p99_itl_ms": round(res.get("p99_itl_ms", 0), 2),
        "mean_ttft_ms": round(res.get("mean_ttft_ms", 0), 1),
        "tokens": toks, "events": events,
        "tok_per_event": round(toks / events, 3) if events else None,
        "fe_cpu_s": fe_cpu,
        "vllm_ttft_mean_ms": vllm_ttft_ms,
        "client_ttft_mean_ms": round(res.get("mean_ttft_ms", 0), 1),
        "fe_cpu_us_per_tok": round(sum(fe_cpu.values()) * 1e6 / toks, 1) if toks else None,
        "client_cpu_s": client_cpu,
        "eng_cpu_s": round(sum(en1[k][1] - en0.get(k, (None, 0))[1] for k in en1), 2),
        "fe_peak_rss_mb": {f"{label(p)}[{p.pid}]": round(sampler.peak.get(p.pid, 0) / 2**20)
                           for p in fe_procs},
        "steady_state": res.get("steady_state"),
        "probe": (json.loads((OUT / f"{name}.probe.json").read_text())
                  if probe and (OUT / f"{name}.probe.json").exists() else None),
    }
    print(json.dumps({k: v for k, v in row.items() if k != "steady_state"}), flush=True)
    with open(OUT / "rows.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["py", "rs"], default="py")
    ap.add_argument("--api-servers", type=int, default=1)
    ap.add_argument("--step-us", type=int, default=10000)
    ap.add_argument("--conc", default="1,32,128,256,512,1024")
    ap.add_argument("--factor", type=int, default=6)
    ap.add_argument("--min-prompts", type=int, default=16)
    ap.add_argument("--out-len", type=int, default=256)
    ap.add_argument("--in-len", type=int, default=256)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    tag = a.tag or f"{a.kind}{a.api_servers}_s{a.step_us}"
    env = dict(os.environ, VLLM_LOGGING_LEVEL="WARNING", RUST_LOG="warn")
    fe_log = open(OUT / f"{tag}.frontend.log", "w")
    en_log = open(OUT / f"{tag}.engine.log", "w")
    fe = subprocess.Popen(frontend_cmd(a.kind, a.api_servers), stdout=fe_log,
                          stderr=subprocess.STDOUT, env=env, start_new_session=True)
    # The frontend binds the handshake socket only after its imports (~20 s for torch), and
    # the mock engine gives up after 5 s, so wait for the listener first.
    deadline = time.time() + 300
    while subprocess.run(["lsof", "-nP", "-iTCP:29550", "-sTCP:LISTEN"],
                         capture_output=True).returncode != 0:
        if time.time() > deadline or fe.poll() is not None:
            raise RuntimeError("frontend never bound the handshake port")
        time.sleep(0.5)
    en = subprocess.Popen([str(RUST / "vllm-mock-engine"),
                           "--handshake-address", "tcp://127.0.0.1:29550",
                           "--engine-count", "1", "--vocab-size", "32000",
                           "--step-interval-us", str(a.step_us)],
                          stdout=en_log, stderr=subprocess.STDOUT, env=env,
                          start_new_session=True)
    try:
        wait_ready()
        time.sleep(2)
        fe_procs, eng_procs = tree(fe.pid), tree(en.pid)
        print("frontend procs:", [(p.pid, label(p)) for p in fe_procs], flush=True)
        # Warm-up: tokenizer caches, first-request paths.
        run_point(tag + "_warm", 8, 16, 32, a.in_len, fe_procs, eng_procs)
        for c in [int(x) for x in a.conc.split(",")]:
            n = max(a.min_prompts, c * a.factor)
            run_point(tag, c, n, a.out_len, a.in_len, fe_procs, eng_procs)
    finally:
        for p in (en, fe):
            try:
                os.killpg(p.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        for p in (en, fe):
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


if __name__ == "__main__":
    main()
