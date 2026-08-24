"""运行 MVP 的三个最小示例：python -m physioagent.demo

Run the three minimal MVP examples: python -m physioagent.demo
"""

from pathlib import Path

from .agent import RuleBasedAgent


def main() -> None:
    csv_path = Path(__file__).parents[1] / "data" / "sample_ecg.csv"
    agent = RuleBasedAgent()
    for question in ("这段信号的平均心率是多少？", "请检测峰值。", "给出信号的基础统计量。"):
        response = agent.run(question, csv_path, sampling_rate=25)
        print(f"问题：{question}")
        print(f"工具：{response.tool_name}")
        print(f"回答：{response.answer}\n")


if __name__ == "__main__":
    main()
