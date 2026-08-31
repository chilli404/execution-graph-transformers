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
        try:
            urllib.request.urlopen(f"http://localhost:{port}/health")
            return True
        except Exception:
            time.sleep(2)
    return False


def launch_server(hf_dir, mode, port=8000, dtype="bfloat16"):
    """Launch vLLM server with a specific execution mode."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", hf_dir,
        "--trust-remote-code",
        "--dtype", dtype,
        "--port", str(port),
        "--disable-log-requests",
    ]
    if mode != "default":
        cmd += ["--hf-overrides", json.dumps({"execution_mode": mode})]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not wait_for_server(port):
        proc.kill()
        raise RuntimeError(f"Server failed to start for mode={mode}")
    return proc


def run_benchmark(hf_dir, port, num_prompts, input_len, output_len):
    """Run vLLM's benchmark_serving and parse output."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.benchmark_serving",
        "--backend", "vllm",
        "--model", hf_dir,
        "--port", str(port),
        "--num-prompts", str(num_prompts),
        "--request-rate", "inf",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    # Parse key metrics from benchmark output
    metrics = {"raw_output": output}
    for line in output.split("\n"):
        line = line.strip()
        if "Request throughput" in line:
            try:
                metrics["request_throughput"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "Output token throughput" in line:
            try:
                metrics["output_token_throughput"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "Mean TTFT" in line:
            try:
                metrics["mean_ttft_ms"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "Median TTFT" in line:
            try:
                metrics["median_ttft_ms"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "P99 TTFT" in line:
            try:
                metrics["p99_ttft_ms"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "Mean TPOT" in line:
            try:
                metrics["mean_tpot_ms"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "Median TPOT" in line:
            try:
                metrics["median_tpot_ms"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
        elif "P99 TPOT" in line:
            try:
                metrics["p99_tpot_ms"] = float(line.split(":")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
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

                    throughput = metrics.get("output_token_throughput", "N/A")
                    ttft = metrics.get("mean_ttft_ms", "N/A")
                    tpot = metrics.get("mean_tpot_ms", "N/A")
                    print(f"    Throughput: {throughput} tok/s, "
                          f"TTFT: {ttft}ms, TPOT: {tpot}ms")
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
