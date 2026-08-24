# PhysioAgent 真实 ECG 工具评测报告

## 结论

当前通用峰检测器不适合直接计算真实 ECG 心率；它经常把 T 波当成额外心搏。在算法修改前
固定数据划分后，项目实现并冻结了 `ecg_detector_v1`。该检测器在开发/验证集与一次性冻结
测试上均实现了完全心搏匹配。

这些结果说明当前轻量算法能处理选定的四个 MIT-BIH 片段，不代表它已经能适用于任意患者、
导联、设备、噪声条件或临床场景。

## 实验设计

数据划分在修改 ECG 算法前写入 `evaluation/real_signal_split_v1.json`：

| 划分 | MIT-BIH 记录 | 用途 |
| --- | --- | --- |
| Development | 100 | 分析通用检测器错误并开发 ECG 检测器 |
| Validation | 101、200 | 选择一个固定配置 |
| Frozen test | 207 | 配置冻结后只评测一次 |

每条记录使用第 0 通道的前 30 秒，采样率为 360 Hz。划分文件 SHA-256：
`39f94aa91bec163cbaa19b22a589c6ff8b4ea3d526620013a04bb7675b556332`。

心搏检测允许误差为标注位置前后 150 ms，并采用一对一匹配。指标包括 sensitivity、positive
predictive value、F1，以及检测心率相对专家标注心率的绝对误差。

## 通用峰检测基线

在 development + validation 的 115 个专家标注心搏上：

| 指标 | 结果 |
| --- | ---: |
| True positives | 106 |
| False positives | 120 |
| False negatives | 9 |
| Micro sensitivity | 0.922 |
| Micro PPV | 0.469 |
| Micro F1 | 0.622 |
| 平均心率绝对误差 | 73.84 BPM |

主要错误是重复检测 T 波；记录 200 还存在异常心搏漏检。因此不能只依靠延长最小峰间隔。

## ECG detector v1

固定配置：

```text
5–15 Hz 二阶 Butterworth 带通
差分平方
120 ms 移动积分
threshold = median + 5 × MAD
最小候选间隔 250 ms
在 ±120 ms 内按滤波信号绝对幅度精确定位
```

该流程借鉴 Pan–Tompkins 的核心处理思想，但不是原论文算法的逐行复现。使用绝对幅度定位是
为了兼容正向和倒置 QRS。

### Development + validation

| 指标 | 结果 |
| --- | ---: |
| 参考心搏 | 115 |
| True positives | 115 |
| False positives | 0 |
| False negatives | 0 |
| Micro sensitivity / PPV / F1 | 1.000 / 1.000 / 1.000 |
| 平均心率绝对误差 | 0.006 BPM |

### 一次性冻结测试

记录 207 在配置冻结后下载并评测，配置未修改：

| 指标 | 结果 |
| --- | ---: |
| 参考心搏 / 检测峰 | 29 / 29 |
| True positives / false positives / false negatives | 29 / 0 / 0 |
| Sensitivity / PPV / F1 | 1.000 / 1.000 / 1.000 |
| 参考心率 / 检测心率 | 56.847 / 56.847 BPM |
| 心率绝对误差 | 0.000 BPM |

## 限制与下一步

- 数据量只有四条 30 秒片段，不能据此声称跨患者或跨设备泛化。
- 当前只使用每条记录的第 0 通道，没有评估其他导联。
- 尚未系统加入运动伪迹、基线漂移、工频干扰和缺失数据。
- 工具仅用于学习与软件评测，不用于诊断、治疗或临床决策。
- `ecg_detector_v1` 不再根据记录 207 修改；下一版需要增加新的开发数据并预留新的未见测试记录。

下一工程步骤是把已经冻结的 ECG 检测器作为执行层 profile 接入原有 `detect_peaks` 与
`calculate_heart_rate`，然后运行 LoRA v2 Agent → 工具 JSON → 真实 ECG 计算 → grounded answer
的完整闭环。

## LoRA v2 真实 ECG 端到端轨迹

执行层 profile 接入后，LoRA v2 在 OpenPAI 上对记录 207 运行以下问题：

```text
请计算这段 ECG 的平均心率
```

模型严格输出：

```json
{"name":"calculate_heart_rate","arguments":{}}
```

执行器使用 `signal_profile=ecg` 调用冻结的 `ecg_detector_v1`，返回 29 个 R 峰、平均 RR 间隔
1.05546 秒和平均心率 56.84745 BPM。最终确定性回答为：

```text
估计平均心率为 56.8 BPM（检测到 29 个峰）。
```

29 个检测峰全部在专家标注的 ±150 ms 容差内；22 个与标注采样点完全相同，平均绝对定位误差
为 2.11 ms，最大误差为 44.44 ms。完整机器可读轨迹保存在
`outputs/end_to_end/real_ecg_lora_v2_record207.json`。
