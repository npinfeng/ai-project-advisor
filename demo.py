"""AI 技术选型与开源项目评估 Agent — Demo 脚本。

运行前确保：
1. pip install -e . 已完成
2. .env 文件已配置 DEEPSEEK_API_KEY 和 TAVILY_API_KEY
"""

import asyncio
import sys
import io

# 修复 Windows GBK 终端下 emoji 编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from project_advisor.graph import graph

# 一个需求明确的提问，避免触发追问
QUESTION = (
    "我要用 Python 开发一个多智能体协作系统，团队有 5 人，半年 Python 经验，"
    "需要部署在自己的服务器上。核心需求：1) 支持 MCP 协议集成外部工具 "
    "2) 支持 RAG 检索增强生成 3) 支持人工审批节点 4) 支持持久化状态管理。"
    "候选项目：LangGraph、CrewAI。请帮我做技术选型评估。"
)


async def main():
    print("=" * 60)
    print("[AI Project Advisor] Demo")
    print("=" * 60)
    print(f"\nQuestion: {QUESTION}\n")
    print("Running (multi-agent workflow, may take 2-5 minutes)...\n")
    print("-" * 60)

    try:
        # 禁用追问，确保完整流程执行
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": QUESTION}]},
            config={"configurable": {"allow_clarification": False}},
        )

        report = result.get("final_report", "")
        if report:
            print("\n" + "=" * 60)
            print("FINAL REPORT")
            print("=" * 60)
            print(report)
        else:
            # 可能触发了追问
            msgs = result.get("messages", [])
            for msg in msgs:
                if hasattr(msg, "type") and msg.type == "ai":
                    print(f"\n[Agent response]: {msg.content}")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
