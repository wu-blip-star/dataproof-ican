# DataProof 数证链

**从原始数据到研究结论的全链路可验证AI审查系统**

Windows 快速体验：

1. 解压 `DataProof_Runtime.zip`，进入含 `start.bat` 的文件夹。
2. 双击 `start.bat`（需先安装 Python，首次启动需联网安装依赖）。
3. 浏览器打开 <http://127.0.0.1:8504>。
4. 直接加载内置 Demo；无需 API Key。进入“V4 AI主张与证据图”可体验解析、确认、证据图与报告。

## 1. 项目简介

大模型负责理解研究语言，统计引擎负责验证研究证据。DataProof 把原始数据、变量映射、质量提示、方法审计、真实统计计算与研究主张连接成 D2C Evidence Graph。

本运行包来自已冻结的 V5 项目。统计核心、Prompt 和原有 Demo 数据均保持原样，仅调整交付副本的离线默认选项、启动提示与部署配置。不包含比赛材料或验证开发文件；本包不是完整证据归档。

## 2. 系统运行要求

Windows 10/11、64 位 Python 3.13（冻结环境为 3.13.2）及浏览器。安装 Python 时勾选 Add Python to PATH。首次安装依赖需要网络；安装完成后，内置离线流程不需要调用 LLM。ZIP 不含 Python、`.venv` 或任何密钥。

## 3. Windows 一键启动

双击 `start.bat`。脚本先检查 Python（没有现成虚拟环境时），创建 `.venv`，检查依赖并按需安装 `requirements.txt`，再启动 Streamlit，明确提示本地网址。保持命令窗口打开；按 Ctrl+C 停止。若启动失败，窗口会保留错误提示。

## 4. 手动启动方式

在项目文件夹打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

无需激活虚拟环境。端口占用时在最后一条命令末尾添加 `--server.port 8505`，浏览器改访问相应端口。

## 5. Demo 数据说明

`demo_data/` 提供三组已验收数据的原样副本。`data/` 保留应用原有文件名供内置选项读取；不要删除该目录。CSV 均为明确标记的合成演示数据，不代表真实采集。

| Demo | 用途 | 建议主张 |
| --- | --- | --- |
| stable_demo.csv | 观察学习时间与成绩的稳定关联 | 学习时间越长，考试成绩越高。 |
| fragile_demo.csv | 检查表面显著但对分析路径敏感的关系 | synthetic_x 与 synthetic_y 呈正相关。 |
| quality_impact_demo.csv | 对照风险样本敏感性处理前后的判断 | exposure 与 outcome 呈正相关。 |

后两组使用结构化输入。详细映射及入口见 [Demo 操作说明](demo_data/README.md)。所有结果由当前统计引擎实时计算，不预先写入固定 p 值或判断。

## 6. 基本操作流程

完整离线示范（不需要密钥）：

1. 左侧选择“体验模拟数据”，分析工作区选择“V4 AI主张与证据图”。
2. 未配置 API 时自动选择“离线契约演示（固定响应，非真实LLM）”。自然语言 Demo 选择“学习时间越长，考试成绩越高。”，点击“解析研究主张”。
3. 核对 X=study_hours、Y=exam_score、方向 positive，勾选映射确认，再点击“确认映射并进入证据审计”。
4. 查看方法诊断，保留 Pearson；勾选“执行 V2 证据稳定性分析（Pearson）”，在质量敏感性排除规则中选择“IQR 潜在极端值”。
5. 点击“V4 执行真实统计核验”，查看数据质量、统计结果、质量敏感性、稳定性与 D2C Evidence Graph。
6. 点击“一键导出 HTML 可信性审计报告”。下载的 HTML 可在浏览器独立打开，也可下载完整证据 JSON。

上传自己的 CSV / Excel 后，也可用“结构化输入（无需 API）”确认变量并运行。因果主张需要明确确认降级为关联表述；系统不验证原始因果断言。

## 7. Offline Parser 说明

当前已有 Offline Parser 是**固定契约演示**，仅支持以下 5 句预设，不是通用中文理解模型：

- 女生满意度显著高于男生。
- 学习时间越长，考试成绩越高。
- 短视频使用导致考试成绩下降。
- 控制睡眠时间和短视频使用时间后，学习时间与考试成绩仍呈显著正相关。
- 不同年级的满意度存在显著差异。

预设使用内置大学生模拟数据。任意其他文字请使用结构化输入或自行配置真实 API。离线模式仅固定语言解析响应；用户确认、统计检验、质量审计、稳定性和报告都由原有程序真实执行。Bootstrap / Jackknife 当前主要适用于 Pearson，其他方法会标记未执行，不应声称都经过同样的稳定性检验。

## 8. 真实 LLM 配置方式

**真实 LLM 解析是可选增强能力。** 在 Windows“编辑账户的环境变量”中自行配置以下变量，然后重启程序：

| 变量 | 设置 |
| --- | --- |
| DATAPROOF_LLM_API_KEY | 自己的兼容服务密钥，仅在本机配置，不提交文件。 |
| DATAPROOF_LLM_BASE_URL | DeepSeek 使用 `https://api.deepseek.com`。 |
| DATAPROOF_LLM_MODEL | DeepSeek 使用 `deepseek-flash`。 |

之后选择“API解析”。API 只接收主张和字段描述，不负责生成统计量、p 值、显著性结果或 Evidence Judgment。不要共享密钥，调用可能产生费用。`DEEPSEEK_API_KEY` 是原 V5 验证脚本使用的变量，不是本应用的直接配置变量。

在线版本只能由部署者在 Streamlit Cloud Secrets / 环境变量中配置；无需密钥也能部署和使用离线流程。具体步骤见 [部署说明](DEPLOYMENT.md)。

## 9. 常见问题

- **找不到 Python**：安装 Python 并加入 PATH，关闭终端后重新启动。
- **依赖下载失败**：检查网络；按手动方式重试 pip 安装。不要复制其他电脑的 `.venv`。
- **网页没自动打开**：手动访问本地网址；不要双击 `app.py`。
- **没有 API Key**：直接使用自动选中的离线预设或结构化输入。
- **自定义文字不能离线解析**：已有离线模式只支持 5 句预设，这不是程序故障。
- **切换变量后结果消失**：需重新确认映射并核验，避免展示旧数据结果。
- **运行包没有 validation / reports / competition_materials**：本包只用于运行；完整冻结实验材料另行提交，不在此处重新计算或改写。
- **在线休眠或资源不足**：Community Cloud 是托管服务，有资源与休眠限制；赛前应实测公网链接并保留本源码包作为备用，不保证平台 SLA。

## 10. 系统边界

相关不能证明因果；显著不等于实际重要；不显著不证明无关系或等效。IQR / Isolation Forest 仅提示风险，不自动证明样本无效。受控扰动不等于科研造假检测。所有解析必须人工确认，系统不替代研究者与领域专家判断。图的箭头是证据依赖而非因果；文件指纹用于版本核对而非真实性认证。
