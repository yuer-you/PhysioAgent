# PhysioAgent SFT 数据 v1

## 文件

- `train.jsonl`：500 条，用于更新 LoRA 参数；
- `validation.jsonl`：100 条，用于观察验证损失和选择训练轮次；
- `test.jsonl`：100 条，训练与调参期间禁止查看结果；
- `manifest.json`：生成版本、随机种子和数量记录。

每个集合均在五个工具间严格平衡。生成器固定随机种子，并为三个集合使用不同的问题
模板；所有问题文本全局唯一。

## 单条格式

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "检测突出度至少为 0.3 的峰"}
  ],
  "completion": [
    {"role": "assistant", "content": "{\"name\":\"detect_peaks\",\"arguments\":{\"prominence\":0.3}}"}
  ]
}
```

使用 prompt-completion 而不是普通 messages，是为了训练时只对 assistant 的工具调用答案
计算损失，不要求模型背诵 system/user 提示。

## 限制

这些样本由人工规则与模板合成，适合验证第一版 SFT/LoRA 管线，但不能单独证明模型对
真实用户表达的泛化能力。项目最终报告还需要一份人工独立编写、未参与提示词设计、模板
设计或训练调参的封存测试集。

重新生成：

```bash
python scripts/generate_sft_data.py
```
