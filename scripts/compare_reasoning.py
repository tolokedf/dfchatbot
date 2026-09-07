#!/usr/bin/env python3
"""
Side-by-side Reasoning Model Comparison: Google Gemini vs. DeepSeek API

Usage:
    /home/tinonn/DF_application/dfchatbot/.venv/bin/python scripts/compare_reasoning.py "Your question or logic problem here"
    or
    python scripts/compare_reasoning.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_FILE, override=True)

from google import genai
from google.genai import types

DEFAULT_PROMPT = (
    "An autonomous mobile robot (AMR) navigates along a path with 3 optical docking stations "
    "spaced 12 meters apart. The robot starts at rest at Station 1, accelerates at 0.5 m/s² "
    "up to a top speed of 1.5 m/s, travels at constant speed, and decelerates at 0.5 m/s² to stop at Station 3. "
    "At Station 2, it does not stop, but reads an RFID tag taking 0.2s. "
    "Calculate the total travel time from Station 1 to Station 3, and state the exact speed as it passes Station 2. "
    "Show complete step-by-step reasoning."
)

def run_deepseek_reasoning(prompt: str, model: str = None) -> dict:
    """Invokes DeepSeek API and extracts reasoning content, latency, and tokens."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY not configured in .env"}

    model_name = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a precise technical reasoning assistant. Think step-by-step and provide thorough reasoning before your conclusion."},
            {"role": "user", "content": prompt}
        ]
    }

    start_time = time.time()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        elapsed = time.time() - start_time
        res_json = resp.json()

        if resp.status_code != 200:
            err_msg = res_json.get("error", {}).get("message", resp.text)
            return {
                "error": f"HTTP {resp.status_code}: {err_msg}",
                "elapsed": elapsed,
                "model": model_name
            }

        choice = res_json.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        usage = res_json.get("usage", {})

        in_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage.get("completion_tokens", 0)
        cost_est = (in_tokens / 1_000_000 * 0.22) + (out_tokens / 1_000_000 * 0.66)

        return {
            "model": model_name,
            "reasoning": reasoning,
            "answer": content,
            "elapsed": elapsed,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "cost_est": cost_est
        }
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start_time, "model": model_name}


def run_gemini_reasoning(prompt: str, model: str = None) -> dict:
    """Invokes Google Gemini with step-by-step technical reasoning."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {"error": "GEMINI_API_KEY not configured in .env"}

    model_name = model or os.environ.get("GEMINI_QA_MODEL", "gemini-3.5-flash-lite")
    client = genai.Client(api_key=api_key)

    full_prompt = (
        "You are an expert robotics systems engineer. Solve the following technical problem with rigorous, "
        "step-by-step reasoning before concluding with the final answer:\n\n"
        f"{prompt}"
    )

    start_time = time.time()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=full_prompt
        )
        elapsed = time.time() - start_time

        in_tokens = 0
        out_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            in_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            out_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        cost_est = (in_tokens / 1_000_000 * 0.30) + (out_tokens / 1_000_000 * 2.50)

        return {
            "model": model_name,
            "answer": response.text.strip() if response.text else "",
            "elapsed": elapsed,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
            "cost_est": cost_est
        }
    except Exception as e:
        return {"error": str(e), "elapsed": time.time() - start_time, "model": model_name}


def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_PROMPT

    print("\n" + "=" * 80)
    print("🤖 REASONING BENCHMARK: GOOGLE GEMINI vs. DEEPSEEK API")
    print("=" * 80)
    print(f"Prompt:\n{prompt}\n")
    print("-" * 80)

    # 1. Test Google Gemini
    print("⏳ Running Gemini Reasoning...")
    gemini_res = run_gemini_reasoning(prompt)

    # 2. Test DeepSeek
    print("⏳ Running DeepSeek Reasoning...")
    deepseek_res = run_deepseek_reasoning(prompt)

    # Display Results
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE & METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Gemini (' + str(gemini_res.get('model')) + ')':<25} | {'DeepSeek (' + str(deepseek_res.get('model')) + ')':<25}")
    print("-" * 80)

    g_status = "Success" if "error" not in gemini_res else f"Error ({gemini_res['error'][:15]}...)"
    d_status = "Success" if "error" not in deepseek_res else f"Error ({deepseek_res['error'][:15]}...)"
    print(f"{'Status':<25} | {g_status:<25} | {d_status:<25}")

    g_time = f"{gemini_res.get('elapsed', 0):.2f}s"
    d_time = f"{deepseek_res.get('elapsed', 0):.2f}s"
    print(f"{'Latency':<25} | {g_time:<25} | {d_time:<25}")

    g_in = str(gemini_res.get("in_tokens", "-"))
    d_in = str(deepseek_res.get("in_tokens", "-"))
    print(f"{'Input Tokens':<25} | {g_in:<25} | {d_in:<25}")

    g_out = str(gemini_res.get("out_tokens", "-"))
    d_out = str(deepseek_res.get("out_tokens", "-"))
    print(f"{'Output Tokens':<25} | {g_out:<25} | {d_out:<25}")

    g_cost = f"${gemini_res.get('cost_est', 0):.6f}" if "cost_est" in gemini_res else "-"
    d_cost = f"${deepseek_res.get('cost_est', 0):.6f}" if "cost_est" in deepseek_res else "-"
    print(f"{'Estimated Query Cost':<25} | {g_cost:<25} | {d_cost:<25}")

    print("=" * 80)

    # Detailed Outputs
    print("\n🔹 [GOOGLE GEMINI OUTPUT]")
    if "error" in gemini_res:
        print(f"⚠️ Error: {gemini_res['error']}")
    else:
        print(gemini_res["answer"])

    print("\n" + "-" * 80)
    print("🔹 [DEEPSEEK OUTPUT]")
    if "error" in deepseek_res:
        print(f"⚠️ Error: {deepseek_res['error']}")
        if "Insufficient Balance" in deepseek_res["error"]:
            print("\n💡 NOTE: Your DeepSeek API key is authenticated and valid, but your account balance is $0.")
            print("   Please add credits at: https://platform.deepseek.com/top_up to activate completions.")
    else:
        if deepseek_res.get("reasoning"):
            print("🧠 [DeepSeek Thinking Trace]:")
            print(deepseek_res["reasoning"])
            print("\n🏁 [Final Answer]:")
        print(deepseek_res["answer"])

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
