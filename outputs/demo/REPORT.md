# Workflow DPO v1 最终演示验收

## 结论

最终一键演示通过。Workflow DPO v1 在 MIT-BIH 记录 207 上严格生成并执行三步计划：

```text
load_signal(signal)
  -> filter_signal(lowcut=0.5, highcut=40.0)
  -> calculate_heart_rate()
```

没有启用保守恢复，原始计划通过严格 JSON 与 schema 校验。执行器确认第二步输入来自第一步数组，第三步输入来自第二步滤波数组。

## 结果

- 输入采样点：10800
- 采样率：360 Hz
- ECG detector：`ecg_detector_v1`
- 检测峰数：29
- 平均 RR 间隔：1.055456 秒
- 平均心率：56.847448 BPM
- 停止原因：`plan_completed`
- 最终回答：`已按顺序执行 3 个工具。估计平均心率为 56.8 BPM（检测到 29 个峰）。`

轨迹文件：`outputs/demo/workflow_dpo_v1_record207.json`

SHA-256：

```text
2e6fff3d26de0049e3f6a39a98f8400b315f2536285572215d03d2f860dcc376
```

该演示是项目交付验收，不是新的未见泛化测试。项目仅用于学习和软件评测，不用于临床诊断。
