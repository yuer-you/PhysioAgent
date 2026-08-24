"""标准 JSON Schema 格式的工具说明。

模型只负责选择工具和填写“用户真正指定的参数”。信号路径与采样率属于应用上下文，
由 Agent 在执行时提供，避免模型编造路径或传入一大串信号数值。
"""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "load_signal",
            "description": "读取当前 CSV 生理时序信号。仅当用户明确要求加载或读取信号时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_column": {
                        "type": "string",
                        "minLength": 1,
                        "description": "CSV 中的信号列名，默认是 signal。",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_statistics",
            "description": "计算当前信号的样本数、时长、均值、标准差、最小值和最大值。",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_peaks",
            "description": "检测当前信号中的局部峰值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_distance_seconds": {
                        "type": "number",
                        "description": "相邻峰的最小时间间隔（秒），默认 0.3。",
                    },
                    "prominence": {
                        "type": "number",
                        "description": "峰的最小突出度；用户未指定时不要填写。",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_heart_rate",
            "description": "根据当前信号的峰间隔估计平均心率（BPM）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_distance_seconds": {
                        "type": "number",
                        "description": "相邻心搏峰的最小时间间隔（秒），默认 0.3。",
                    },
                    "prominence": {
                        "type": "number",
                        "description": "峰的最小突出度；用户未指定时不要填写。",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_signal",
            "description": "对当前信号进行 Butterworth 带通滤波。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lowcut": {"type": "number", "description": "低截止频率（Hz），默认 0.5。"},
                    "highcut": {"type": "number", "description": "高截止频率（Hz），默认 8.0。"},
                    "order": {"type": "integer", "description": "滤波器阶数，默认 4。"},
                },
                "additionalProperties": False,
            },
        },
    },
]

TOOL_NAMES = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
