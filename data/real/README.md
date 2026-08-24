# Real data workspace

这里存放从公开数据源转换出的真实生理信号，不用玩具波形冒充真实 ECG。

当前入口是 MIT-BIH Arrhythmia Database。执行 `scripts/prepare_mitdb_record.py` 后，默认生成：

```text
data/real/mitdb/100_30s/
├── signal.csv      # 项目五个工具可直接读取的单通道信号
└── reference.json  # 数据来源、许可、采样率和专家心搏标注
```

MIT-BIH 数据来源与许可：

- https://physionet.org/content/mitdb/1.0.0/
- https://physionet.org/content/mitdb/view-license/1.0.0/

仓库中的 `signal.csv` 和 `reference.json` 是从记录 100、101、200、207 的第0通道前30秒转换得到的派生片段，不是完整数据库。文件遵循 Open Data Commons Attribution License v1.0，并保留以下署名：

```text
Moody GB, Mark RG. MIT-BIH Arrhythmia Database. PhysioNet.
Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database.
IEEE Eng Med Biol Mag. 2001;20(3):45-50.
DOI: 10.13026/C2F305
```

每条 `reference.json` 记录原始数据集、记录号、通道、采样率、来源 URL、许可 URL 和专家标注。

这些数据仅用于学习和软件评测，不用于临床诊断。
