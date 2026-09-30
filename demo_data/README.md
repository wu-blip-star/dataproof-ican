# 已有 Demo 数据

这些文件是已验收合成演示数据的逐字节副本，没有重新生成数据或实验结果，不是真实调查。

| 文件 | 原文件 | 建议主张与使用方式 |
| --- | --- | --- |
| stable_demo.csv | data/demo_students.csv | 学习时间越长，考试成绩越高。选择内置“体验模拟数据”，可直接使用离线预设解析。 |
| fragile_demo.csv | data/stability_demo.csv | synthetic_x 与 synthetic_y 呈正相关。选择“合成脆弱性演示”或上传文件，使用结构化输入，X=synthetic_x、Y=synthetic_y、positive，执行 Pearson 稳定性。 |
| quality_impact_demo.csv | data/quality_flip_demo.csv | exposure 与 outcome 呈正相关。上传此文件，进入 V4 结构化输入，X=exposure、Y=outcome、positive，质量敏感性选“IQR 潜在极端值”，启用 Pearson 稳定性后核验、查看图并导出报告。 |

后两组案例的变量名不在固定自然语言预设中；不要把手动结构化输入称为 AI 解析。侧栏 Demo D 可直接体验旧版质量翻转工作区；如需 V4 图与 HTML，请按上表上传该 CSV 并进入 V4 工作区。
