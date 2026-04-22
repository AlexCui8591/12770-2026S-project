"""
Phase 0 Smoke Test
验证 Ollama + Qwen2.5-7B 能正常通信并返回自然语言回复。
"""

import ollama

MODEL = "qwen2.5:7b"


def main():
    print(f"[1] 正在连接 Ollama，使用模型: {MODEL}")

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Hello! Please introduce yourself in 2 sentences."}],
        )
    except Exception as e:
        print(f"[ERROR] 连接失败: {e}")
        print("请确认:")
        print("  1. ollama serve 正在运行")
        print("  2. 已执行 ollama pull qwen2.5:7b")
        return

    reply = response["message"]["content"]
    print(f"[2] 模型回复:\n{reply}")

    # 验证 tool-calling 格式是否可用（后续 Phase 会用到）
    print("\n[3] 测试 tool-calling 格式...")
    try:
        tool_response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "What is the weather in Pittsburgh?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                }
            ],
        )

        msg = tool_response["message"]
        if msg.get("tool_calls"):
            tc = msg["tool_calls"][0]
            print(f"  模型发起了 tool call: {tc['function']['name']}({tc['function']['arguments']})")
            print("  Tool-calling 功能正常!")
        else:
            print(f"  模型未发起 tool call，而是直接回复: {msg['content'][:100]}")
            print("  (这是 Qwen2.5-7B 的常见行为，后续用 dual-strategy fallback 处理)")

    except Exception as e:
        print(f"  Tool-calling 测试出错: {e}")
        print("  (不影响 Phase 0，后续 Phase 3 会处理)")

    print("\n[DONE] Phase 0 smoke test 完成。")


if __name__ == "__main__":
    main()
