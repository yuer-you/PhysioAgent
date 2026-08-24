# LoRA v2 真实 ECG 端到端集成评测

## 结论

LoRA v2 在 20 条真实 ECG 集成案例上实现 `20/20` 端到端成功。五个 Agent 工具均为 `4/4`，
工具 JSON、参数、真实执行、参考约束和 grounded answer 全部通过。

这是一套工程集成验收，不是新的未见泛化测试。问题是在模型训练和冻结最终文本评测完成后编写的，
因此结果证明当前组件能够正确连接，不应被解释为模型在未知任务上的准确率。

## 评测范围

| 项目 | 内容 |
| --- | --- |
| 基础模型 | Qwen2.5-3B-Instruct |
| 后训练模型 | LoRA v2 |
| 真实数据 | MIT-BIH 记录 100、101、200、207，各前 30 秒 |
| Signal profile | `ecg` |
| 案例数量 | 20 |
| 工具 | load、statistics、peaks、heart rate、filter，各 4 条 |
| 语言 | 中文和英文 |
| 参数覆盖 | 空参数、列名、截止频率、滤波阶数 |

## 指标

| 指标 | 结果 |
| --- | ---: |
| 合法工具 JSON | 20/20（100%） |
| 工具名和参数严格匹配 | 20/20（100%） |
| 工具执行成功 | 20/20（100%） |
| 真实参考或结构检查通过 | 20/20（100%） |
| 回答忠于工具结果 | 20/20（100%） |
| 端到端成功 | 20/20（100%） |

分类结果：

| 工具 | 成功 |
| --- | ---: |
| `load_signal` | 4/4 |
| `calculate_statistics` | 4/4 |
| `detect_peaks` | 4/4 |
| `calculate_heart_rate` | 4/4 |
| `filter_signal` | 4/4 |

## 各层验证的含义

1. `valid_tool_call`：模型输出能被严格 JSON 解析器接受。
2. `tool_call_exact`：工具名和参数与人工标签完全相同，不允许多余默认参数。
3. `execution_success`：调用能够在真实 10800 点 ECG 文件上成功运行。
4. `reference_check_passed`：R 峰和心率对照专家标注；加载、统计和滤波满足对应结构约束。
5. `answer_grounded`：最终回答中出现的数值来自工具返回值，而不是模型自由生成。
6. `end_to_end_success`：以上各项同时成立。

为了避免结果文件膨胀，加载和滤波得到的 10800 点数组只保存长度、均值、标准差、范围和前五个值。

## 可复现性

```text
results.jsonl SHA-256:
4c3075fc4f3b13fc29401d5d519033eabee73b2ddaf4eb67ff83b50cf3ac538e

summary.json SHA-256:
52750089c1455e2f0c50bd57ebfdd065077a772aab16de067582e11f1e983e40
```

运行命令：

```bash
python -m physioagent.evaluate_real_agent \
  --model-path "$PHYSIOAGENT_MODEL_PATH" \
  --adapter-path outputs/sft/qwen2.5-3b-lora-v2/final_adapter \
  --output-dir outputs/end_to_end/real_agent_integration_v1
```

## 限制与下一步

- 案例只有 20 条，主要验证接口连接和确定性执行。
- 四段 ECG 均来自 MIT-BIH，不能代表跨数据集、跨设备或强噪声条件。
- 当前每个问题只允许一次工具调用，还不能完成“先滤波，再计算心率”之类的组合任务。
- 本项目不用于临床诊断或治疗决策。

下一阶段将保持 LoRA v2 和 `ecg_detector_v1` 冻结，只升级 Agent 控制流：允许一个任务依次执行
多个已有工具，并保存每一步的调用、结果摘要和停止原因。
