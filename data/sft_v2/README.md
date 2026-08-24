# SFT v2 数据

这一版在 v1 基础能力上增加参数 schema 稳定性训练：

- 所有样本使用列出合法参数名的 v2 system prompt；
- 增加中文“二阶/三阶/四阶”等数词；
- 增加英文 `second-order` / `third-order` 等序数词；
- 强化 `signal_column`、`lowcut`、`highcut`、`order` 等规范键名；
- 增加无参数、单边截止频率和上下文干扰数字。

数据规模：

```text
train.jsonl       700 条，每个工具 140 条
validation.jsonl  150 条，每个工具 30 条
test.jsonl        100 条，每个工具 20 条
```

`test.jsonl` 继承自已经查看过结果的 v1 合成测试，因此只能作为开发测试，不能再称为最终
测试。真正冻结的最终测试是 `evaluation/final_cases_v1.jsonl`，生成器不会读取它。

重新生成：

```bash
python scripts/generate_sft_data_v2.py
```
