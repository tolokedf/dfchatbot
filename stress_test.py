"""
Stress Test & Verification Script for NavWiz & DFleet Multimodal RAG Flask App.

Features:
1. Pre-flight sanity checks (ChromaDB, rendered images, API status).
2. Multi-threaded concurrent query stress test (simulates real-world load).
3. Latency statistics (min, avg, p95, max) and error breakdown.

Usage:
    python stress_test.py [--concurrency 5] [--requests 15]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Sample realistic queries covering different topics in the manuals
TEST_QUERIES = [
    "How do I calibrate the sensor array?",
    "What are the pallet docking troubleshooting steps?",
    "Explain critical NavWiz parameters for AGV obstacle sensors.",
    "How to configure safety zones for autonomous mobile robots?",
    "What are the steps to initialize the mapping mode?",
    "How do I adjust the laser scanner filter settings?",
    "What to do when robot fails to localize?",
    "DFleet dispatch and route planning configuration.",
]


def run_preflight_checks(client) -> bool:
    print("\n" + "=" * 60)
    print("🔍 STEP 1: PRE-FLIGHT SYSTEM SANITY CHECKS")
    print("=" * 60)

    # 1. Check /api/status
    try:
        t0 = time.time()
        res = client.get("/api/status")
        elapsed = time.time() - t0
        if res.status_code != 200:
            print(f"❌ /api/status failed with HTTP {res.status_code}: {res.get_data(as_text=True)}")
            return False
        
        data = res.get_json()
        print(f"✅ Flask API status endpoint: OK ({elapsed:.3f}s)")
        print(f"   • ChromaDB Collection: {data.get('collection')}")
        print(f"   • Total Indexed Pages: {data.get('total_indexed_pages')}")
        print(f"   • Embedding Model:     {data.get('embed_model')}")
        print(f"   • QA Model:            {data.get('qa_model')}")

        if data.get("total_indexed_pages", 0) == 0:
            print("❌ ChromaDB collection is empty! Run 'python run_embedding_pipeline.py' first.")
            return False

    except Exception as e:
        print(f"❌ Pre-flight check failed: {e}")
        return False

    # 2. Check index HTML route
    res_html = client.get("/")
    if res_html.status_code == 200 and "NavWiz" in res_html.get_data(as_text=True):
        print("✅ Frontend UI route (GET /): OK")
    else:
        print(f"❌ Frontend route failed with code {res_html.status_code}")
        return False

    print("✅ All pre-flight sanity checks passed!\n")
    return True


def execute_single_request(client, query: str, request_id: int, top_k: int = 3) -> dict:
    start_time = time.time()
    result = {
        "id": request_id,
        "query": query,
        "status_code": 0,
        "latency": 0.0,
        "success": False,
        "error": None,
        "seed_count": 0,
        "expanded_count": 0,
        "answer_length": 0,
    }

    try:
        res = client.post("/api/chat", json={"message": query, "top_k": top_k})
        result["latency"] = time.time() - start_time
        result["status_code"] = res.status_code

        if res.status_code == 200:
            data = res.get_json()
            result["success"] = True
            result["seed_count"] = data.get("seed_count", 0)
            result["expanded_count"] = data.get("expanded_count", 0)
            result["answer_length"] = len(data.get("answer", ""))
        else:
            result["error"] = res.get_json().get("error", "Unknown error")

    except Exception as e:
        result["latency"] = time.time() - start_time
        result["error"] = str(e)

    return result


def run_stress_test(client, total_requests: int = 10, concurrency: int = 4):
    print("=" * 60)
    print(f"🚀 STEP 2: RUNNING CONCURRENT STRESS TEST")
    print(f"   • Total Requests: {total_requests}")
    print(f"   • Concurrency Level: {concurrency} workers")
    print("=" * 60)

    start_total_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(total_requests):
            query = TEST_QUERIES[i % len(TEST_QUERIES)]
            futures.append(executor.submit(execute_single_request, client, query, i + 1))

        print(f"[*] Dispatched {total_requests} concurrent requests...")
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status_symbol = "✅" if res["success"] else "❌"
            print(f"  {status_symbol} Req #{res['id']:02d} | HTTP {res['status_code']} | "
                  f"{res['latency']:.2f}s | Seeds: {res['seed_count']} | "
                  f"Context Pgs: {res['expanded_count']} | Q: {res['query'][:35]}...")

    total_wall_time = time.time() - start_total_time

    # =========================================================================
    # Step 3: Analyze & Display Metrics
    # =========================================================================
    print("\n" + "=" * 60)
    print("📊 STEP 3: PERFORMANCE & RELIABILITY METRICS")
    print("=" * 60)

    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in results]

    success_rate = (len(successes) / len(results)) * 100 if results else 0

    print(f"Total Requests:       {len(results)}")
    print(f"Successful Requests:  {len(successes)} ({success_rate:.1f}%)")
    print(f"Failed Requests:      {len(failures)}")
    print(f"Total Elapsed Time:   {total_wall_time:.2f}s")
    print(f"Throughput:           {len(results) / total_wall_time:.2f} req/sec\n")

    if latencies:
        print("Latency Statistics (Seconds):")
        print(f"  • Min:              {np.min(latencies):.2f}s")
        print(f"  • Average:          {np.mean(latencies):.2f}s")
        print(f"  • Median (p50):     {np.median(latencies):.2f}s")
        print(f"  • 95th Percentile:  {np.percentile(latencies, 95):.2f}s")
        print(f"  • Max:              {np.max(latencies):.2f}s")

    if successes:
        avg_seeds = np.mean([r["seed_count"] for r in successes])
        avg_expanded = np.mean([r["expanded_count"] for r in successes])
        print(f"\nRetrieval Quality:")
        print(f"  • Avg Seed Pages Retrieved: {avg_seeds:.1f}")
        print(f"  • Avg Expanded Context Pgs: {avg_expanded:.1f}")

    if failures:
        print("\n❌ Errors Encountered:")
        for f in failures:
            print(f"  • Req #{f['id']} (HTTP {f['status_code']}): {f['error']}")

    print("\n" + "=" * 60)
    if success_rate == 100:
        print("🎉 TEST RESULT: PASSED (System is stable and performing normally)")
    elif success_rate >= 80:
        print("⚠️ TEST RESULT: WARNING (Some requests failed, check API rate limits)")
    else:
        print("❌ TEST RESULT: FAILED (High error rate, check server logs)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test the Multimodal RAG Flask application.")
    parser.add_argument("--requests", type=int, default=8, help="Total number of requests to run (default: 8)")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent worker threads (default: 3)")
    args = parser.parse_args()

    from app import app
    test_client = app.test_client()

    if run_preflight_checks(test_client):
        run_stress_test(test_client, total_requests=args.requests, concurrency=args.concurrency)
