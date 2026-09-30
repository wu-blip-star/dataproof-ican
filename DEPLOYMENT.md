# GitHub + Streamlit Community Cloud 部署

此目录是准备上传仓库的文件，不包含 Git 历史、密钥或开发环境。尚未部署成功，不提供虚构网址。

1. 打开 https://github.com/login ，由你登录；打开 https://github.com/new ，创建你自己的仓库（例如 dataproof-runtime）。需要接受协议或授权时由你操作。
2. 将 DataProof_Deploy 文件夹内的内容上传到仓库根目录，而非上传 ZIP；必须包含隐藏的 `.streamlit/config.toml` 与 `.gitignore`。可使用 GitHub Desktop 的 Add local repository / Publish repository，或 GitHub 网页 Add file → Upload files。入口是根目录 `app.py`。不要上传整个冻结项目。
3. 打开 https://share.streamlit.io/ ，由你登录并授权连接该 GitHub 仓库。点击 Create app → Yup, I have an app，选择实际仓库和分支，Main file path 填 `app.py`。
4. Advanced settings 中 Python 选择 3.13；离线部署不需要填 Secrets。由你点击部署。服务成功后复制平台实际分配的 streamlit.app URL，不凭空填写网址。
5. 用未登录浏览器或另一台设备访问真实 URL。按 README 的完整离线示范检查数据、解析、确认、计算、稳定性、质量敏感性、图和 HTML 下载；确认无报错、无服务器路径或密钥泄露后再提交链接。

可选真实 LLM：在 Cloud 的 Secrets 界面设置顶层 TOML 键 `DATAPROOF_LLM_API_KEY`、`DATAPROOF_LLM_BASE_URL`、`DATAPROOF_LLM_MODEL`；Key 只由你在该受保护界面填写。base URL 为 https://api.deepseek.com，model 为 deepseek-flash。顶层 secrets 可作为环境变量读取。不要在 `[section]` 下填写这些键，不要把 secrets.toml 上传仓库。重启服务后再选择 API解析。

部署目录的 config.toml 已去掉开发机地址/端口绑定，Cloud 使用平台自己的服务参数；本地运行仍可指定 `--server.address 127.0.0.1 --server.port 8504`。服务器详细异常展示已关闭，实际错误由部署者查看私有日志。

依赖全部来自 requirements.txt；路径基于 app.py 所在目录，无开发机绝对路径。app.py 的 generate_* 导入是既有后备依赖，虽随包数据已齐全，仍需保留这三个模块。无系统级额外安装脚本。

官方参考：
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/develop/concepts/connections/secrets-management
- https://docs.streamlit.io/deploy/streamlit-community-cloud/status
