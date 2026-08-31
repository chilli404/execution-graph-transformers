"""Serving benchmark: compare execution modes under vLLM continuous batching.

Launches the vLLM OpenAI-compatible server for each execution mode,
runs the benchmark at multiple concurrency/length combinations, and
collects TTFT, TPOT, throughput into a single JSON.

Usage:
    cd blackwell/vllm_plugin
    uv run python bench_serving.py \
        --hf-dir ../../runs/430m_poly_hf \
        --output ../../blackwell/results/vllm_serving_430m.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def wait_for_server(port, timeout=300):
    """Wait until the vLLM server is ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        for endpoint in ["/health", "/v1/models"]:
            try:
                urllib.request.urlopen(f"http://localhost:{port}{endpoint}")
                print(f"  Server ready ({endpoint} responded after "
                      f"{time.time()-start:.0f}s)")
                return True
            except Exception:
                pass
        time.sleep(3)
    return False


def launch_server(hf_dir, mode, port=8000, dtype="bfloat16"):
    """Launch vLLM server with a specific execution mode."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_dir,
        "--trust-remote-code",
        "--dtype", dtype,
        "--port", str(port),
        "--no-enable-log-requests",
    ]
    if mode != "default":
        cmd += ["--hf-overrides", json.dumps({"execution_mode": mode})]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if not wait_for_server(port):
        proc.kill()
        output = proc.stdout.read().decode() if proc.stdout else ""
        print(f"Server output:\n{output[-2000:]}", file=sys.stderr)
        raise RuntimeError(f"Server failed to start for mode={mode}")
    return proc


def run_benchmark(hf_dir, port, num_prompts, input_len, output_len):
    """Send requests to vLLM server and measure latency/throughput."""
    import urllib.request
    import random
    import statistics

    random.seed(42)
    model_name = hf_dir.rstrip("/").split("/")[-1]

    ttfts = []
    tpots = []
    total_output_tokens = 0
    total_start = time.time()

    for i in range(num_prompts):
        # Random token IDs as prompt
        prompt_tokens = [random.randint(1, 8191) for _ in range(input_len)]

        payload = json.dumps({
            "model": model_name,
            "prompt": prompt_tokens,
            "max_tokens": output_len,
            "temperature": 0,
            "stream": True,
        }).encode()

        req = urllib.request.Request(
            f"http://localhost:{port}/v1/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        first_token_time = None
        last_token_time = None
        n_tokens = 0
        request_start = time.time()

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                for line in resp:
                    line = line.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        if chunk.get("choices", [{}])[0].get("text", ""):
                            now = time.time()
                            if first_token_time is None:
                                first_token_time = now
                            last_token_time = now
                            n_tokens += 1
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"    Request {i} failed: {e}")
            continue

        if first_token_time is not None:
            ttfts.append((first_token_time - request_start) * 1000)
        if n_tokens > 1 and last_token_time and first_token_time:
            tpots.append((last_token_time - first_token_time) / (n_tokens - 1) * 1000)
        total_output_tokens += n_tokens

    total_time = time.time() - total_start

    metrics = {}
    if ttfts:
        ttfts.sort()
        metrics["mean_ttft_ms"] = statistics.mean(ttfts)
        metrics["median_ttft_ms"] = statistics.median(ttfts)
        metrics["p95_ttft_ms"] = ttfts[int(len(ttfts) * 0.95)]
        metrics["p99_ttft_ms"] = ttfts[int(len(ttfts) * 0.99)]
    if tpots:
        tpots.sort()
        metrics["mean_tpot_ms"] = statistics.mean(tpots)
        metrics["median_tpot_ms"] = statistics.median(tpots)
        metrics["p95_tpot_ms"] = tpots[int(len(tpots) * 0.95)]
        metrics["p99_tpot_ms"] = tpots[int(len(tpots) * 0.99)]
    metrics["total_time_s"] = total_time
    metrics["output_tokens"] = total_output_tokens
    metrics["output_token_throughput"] = total_output_tokens / total_time if total_time > 0 else 0
    metrics["request_throughput"] = num_prompts / total_time if total_time > 0 else 0

    return metrics


def kill_server(proc):
    """Gracefully stop the server."""
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--modes", nargs="+",
                        default=["sequential", "parallel", "parallel_fused"])
    parser.add_argument("--input-lens", type=int, nargs="+",
                        default=[128, 512])
    parser.add_argument("--output-lens", type=int, nargs="+",
                        default=[32, 128])
    args = parser.parse_args()

    results = []

    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")

        proc = launch_server(args.hf_dir, mode, args.port, args.dtype)
        print(f"Server started (pid={proc.pid})")

        for input_len in args.input_lens:
            for output_len in args.output_lens:
                print(f"\n  Benchmarking: input={input_len}, output={output_len}, "
                      f"prompts={args.num_prompts}")
                try:
                    metrics = run_benchmark(
                        args.hf_dir, args.port,
                        args.num_prompts, input_len, output_len)
                    # Remove raw output for clean JSON
                    raw = metrics.pop("raw_output", "")
                    entry = {
                        "mode": mode,
                        "input_len": input_len,
                        "output_len": output_len,
                        "num_prompts": args.num_prompts,
                        **metrics,
                    }
                    results.append(entry)

                    throughput = metrics.get("output_token_throughput", 0)
                    ttft = metrics.get("mean_ttft_ms", 0)
                    tpot = metrics.get("mean_tpot_ms", 0)
                    print(f"    Throughput: {throughput:.0f} tok/s, "
                          f"TTFT: {ttft:.1f}ms, TPOT: {tpot:.2f}ms")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "mode": mode,
                        "input_len": input_len,
                        "output_len": output_len,
                        "error": str(e),
                    })

        kill_server(proc)
        print(f"Server stopped")
        time.sleep(5)

    with open(args.output, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
