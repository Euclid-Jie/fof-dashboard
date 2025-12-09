# fof-dashboard

FOF（基金中的基金）静态展示页生成器

简介
- 本项目生成一个静态的单文件 HTML（out/index.html），用于展示 FOF 组合的持仓信息、组合净值曲线、子基金市值配置走势图、当前配置饼图及常见绩效指标。
- 数据源为 CSV：data/funds.csv、data/navs.csv、data/transactions.csv。你可以把它们替换为真实数据后运行生成脚本。

主要文件
- generate_index.py：主脚本，读取 CSV，计算持仓与组合净值，并生成内联的 index.html（包含 echarts JS）。
- data/*.csv：示例数据（funds/navs/transactions）。
- templates/index_template.html：Jinja2 模板（用于渲染最终 HTML）。
- requirements.txt：Python 依赖。
- LICENSE：MIT 授权。

快速开始（本机）
1. 创建虚拟环境并安装依赖（Python 3.12）：
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows
   pip install -r requirements.txt

2. 查看/替换数据（可使用示例）：
   - data/funds.csv
   - data/navs.csv
   - data/transactions.csv

3. 生成单文件 HTML（示例）：
   python generate_index.py --data-dir data --out-dir out --base-value 1.0 --fund-url-template "https://example.com/fund/{code}"

4. 打开 out/index.html 查看结果（为单文件，已内联 echarts.js，若网络可用将成功内联；若网络不可用会回退到 CDN 引用）。

脚本说明与可选参数
- --data-dir: CSV 所在目录（默认 data)
- --out-dir: 输出目录（默认 out)
- --base-value: 组合净值归一起点（默认 1.0)
- --fund-url-template: 基金链接模板，使用 {code} 作为占位符（默认 "https://example.com/fund/{code}")
- --echarts-version: 下载的 echarts 版本（默认 5.4.2)
- 如果生成过程中无法联网下载 echarts，会自动回退到使用 CDN 外链（脚本会打印提示）。

数据格式（简述)
- funds.csv: fund_code,name (可选 link_template)
- navs.csv: fund_code,date(YYYY-MM-DD),nav
- transactions.csv: fund_code,date, type(buy/sell), shares, price, note
