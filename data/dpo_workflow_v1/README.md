# Workflow DPO v1 偏好数据

## 文件

- `train.jsonl`：1000 对偏好训练样例；
- `validation.jsonl`：200 对偏好验证样例；
- `manifest.json`：生成种子、偏好分布、长度方向和文件哈希。

每条记录包含 conversational `prompt`、`chosen` 和 `rejected`，可直接交给 TRL DPOTrainer。两种回答都能通过严格 workflow schema；chosen 表示符合用户要求的计划，rejected 表示一种受控语义错误。

## 偏好类型

| 类型 | chosen | rejected |
|---|---|---|
| `omit_load` | 完整加载链路 | 省略开头加载 |
| `default_invent_column` | 默认加载参数 `{}` | 虚构普通词作为列名 |
| `explicit_drop_column` | 保留明确列名 | 丢失明确列名 |
| `extra_unrequested_load` | 直接处理现有信号 | 擅自添加加载 |
| `omit_final_analysis` | 包含最终分析工具 | 只做前置处理 |
| `duplicate_final_analysis` | 最终工具执行一次 | 重复执行最终工具 |

训练与验证使用不同的问题模板和显式列名。生成器不会读取任何冻结测试问题或模型输出；final v3 只提供错误类型统计，不提供训练文本。

预期 SHA-256：

```text
train.jsonl      b6c7cd4244943cdfeb92b5dcb7ee805f4ffab8e212754b76ba4b1f3cae59ddd2
validation.jsonl b2a12a322522ccee388e7cbc82c9a7f939ec7212fc9ef90ad81fa738f84f053e
```

该数据仅用于学习和软件评测，不用于临床诊断。
