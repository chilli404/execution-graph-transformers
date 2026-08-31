"""Serving benchmark: compare execution modes under vLLM continuous batching.

Launches the vLLM OpenAI-compatible server for each execution mode,
runs `vllm bench serve` at multiple input/output length combinations,
and collects results into a single JSON.

Usage:
    cd blackwell/vllm_plugin
    uv run python bench_serving.py \
        --hf-dir ../../runs/430m_poly_hf \
        --output ../../blackwell/results/vllm_serving_430m.json
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time


def wait_for_server(port, timeout=300):
    """Wait until the vLLM server is ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health")
            elapsed = time.time() - start
            print(f"  Server ready ({elapsed:.0f}s)")
            return True
        except Exception:
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
    ]
    if mode != "default":
        cmd += ["--hf-overrides", json.dumps({"execution_mode": mode})]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL)
    if not wait_for_server(port):
        proc.kill()
        raise RuntimeError(f"Server failed to start for mode={mode}")
    return proc


def run_benchmark(hf_dir, port, num_prompts, input_len, output_len):
    """Run vllm bench serve and parse the output."""
    cmd = [
        sys.executable, "-m", "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", hf_dir,
        "--port", str(port),
        "--num-prompts", str(num_prompts),
        "--request-rate", "inf",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--temperature", "0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = result.stdout + result.stderr

    metrics = {}
    for line in output.split("\n"):
        line = line.strip()
        # Parse lines like "Mean TTFT (ms):                          122.93"
        match = re.match(r'^(.+?):\s+([\d.]+)\s*$', line)
        if match:
            key = match.group(1).strip()
            try:
                val = float(match.group(2))
            except ValueError:
                continue
            clean_key = re.sub(r'\s*\(.*?\)\s*', ' ', key).strip()
            clean_key = clean_key.lower().replace(' ', '_')
            metrics[clean_key] = val

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

        for input_len in args.input_lens:
            for output_len in args.output_lens:
                print(f"\n  input={input_len}, output={output_len}, "
                      f"prompts={args.num_prompts}")
                try:
                    metrics = run_benchmark(
                        args.hf_dir, args.port,
                        args.num_prompts, input_len, output_len)

                    entry = {
                        "mode": mode,
                        "input_len": input_len,
                        "output_len": output_len,
                        "num_prompts": args.num_prompts,
                        **metrics,
                    }
                    results.append(entry)

                    tput = metrics.get("output_token_throughput", 0)
                    ttft = metrics.get("mean_ttft", 0)
                    tpot = metrics.get("mean_tpot", 0)
                    print(f"    {tput:.0f} tok/s | TTFT {ttft:.1f}ms | TPOT {tpot:.2f}ms")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    results.append({
                        "mode": mode,
                        "input_len": input_len,
                        "output_len": output_len,
                        "error": str(e),
                    })

        kill_server(proc)
        print(f"  Server stopped")
        time.sleep(5)

    with open(args.output, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
