# 番茄育种智能决策系统 (Tomato Breeding Intelligence System)

基于 RAG（检索增强生成）的育种知识库问答系统，面向育种科研人员，解决大量种质实验数据人工整理耗时长、知识检索困难、决策缺乏依据的痛点。

## 核心能力

- **抗病基因知识问答** — 覆盖 Ty1/Ty3/Tm-2a/SW/ToBRFV 等 11 种抗病基因，支持遗传方式、连锁关系、育种应用等深度查询
- **育种数据智能查询** — 自然语言查询 Excel 育种数据，支持糖度/硬度/果重筛选、TOP 排名、均值统计，多季节对比
- **季节品质报告生成** — 一键生成 252/261 等多季品质分析报告，含均值/极值/TOP5/分布统计
- **杂交组合推荐** — 基于抗病基因聚合 + 性状互补原则，结合分子标记数据，智能推荐最优杂交组合
- **API 容错降级** — LLM API 不可用时自动降级为本地检索模式，保证系统始终可用

## 技术架构

```
用户提问 → 意图分类(4类) → 多源检索(TF-IDF + 结构化知识库 + Pandas数据) → LLM生成/本地降级 → 回答
```

| 模块 | 技术 | 说明 |
|------|------|------|
| RAG 引擎 | Python | 意图分类 + 混合检索 + LLM 生成 + 报告生成 |
| 检索引擎 | NumPy (TF-IDF) | 纯 NumPy 实现，零外部依赖 |
| LLM | Qwen-Turbo (DashScope) | API 不可用时自动降级为本地检索 |
| Web 界面 | HTML + JS | 暗色主题聊天界面 |
| 数据处理 | Pandas | Excel 育种数据读取与查询 |

## 项目结构

```
breeding-qa-agent/
├── src/
│   ├── rag_engine.py          # RAG 引擎核心（意图分类/混合检索/LLM 生成/报告生成）
│   ├── embedding_manager.py   # TF-IDF 检索引擎（纯 NumPy 实现）
│   ├── chat_server.py         # Web 服务器
│   └── config.py              # 配置管理
├── knowledge_base/
│   ├── docs/                  # 非结构化育种文档（4 篇 Markdown）
│   │   ├── 01_抗病基因详解.md
│   │   ├── 02_性状遗传规律.md
│   │   ├── 03_杂交育种方法.md
│   │   └── 04_育种目标与评价标准.md
│   └── breeding_knowledge.json # 结构化知识库（基因/遗传/标准）
├── templates/
│   └── chat.html             # 前端聊天界面
├── portfolio.html             # 作品集页面
└── 启动.bat                   # 启动脚本
```

## 快速开始

```bash
# 1. 安装依赖
pip install pandas openpyxl requests numpy

# 2. 配置 API Key
cp config.example.json config.json
# 编辑 config.json 填入你的 DashScope API Key

# 3. 启动系统
python src/chat_server.py
# 或双击 启动.bat
```

浏览器自动打开聊天界面，输入问题即可获得育种知识解答。

## 知识库覆盖

| 类别 | 数量 | 示例 |
|------|------|------|
| 抗病基因 | 11 种 | Ty1, Ty3, Tm-2a, SW, ToBRFV, Frl, Sm, Mi-1 |
| 性状遗传规律 | 6 类 | 果实颜色、大小、糖度、硬度、抗病性、生长习性 |
| 评价标准 | 5 种 | 糖度、硬度、喜好度、抗病性、分析模板 |
| 杂交原则 | 5 条 | 互补、抗病聚合、避免近缘、目标导向、杂合注意 |

## 意图分类

系统自动识别用户意图，路由到不同检索通道：

| 意图 | 触发关键词 | 处理逻辑 |
|------|-----------|----------|
| report_query | "生成报告""分析报告""品质分析" | 汇总指定季节的品质/性状数据 |
| data_query | "糖度大于X""TOP""排名""筛选" | Pandas 条件查询 Excel 数据 |
| hybrid_query | "杂交组合""推荐""配组""选配" | 基于基因聚合+性状互补推荐 |
| knowledge_query | 其他 | TF-IDF + 结构化知识库检索 |

## 技术亮点

- **TF-IDF 纯 NumPy 实现** — 检索引擎零外部依赖，适合离线场景
- **双模态容错** — API 模式（LLM 综合分析）+ 本地模式（知识库检索直出）
- **三路检索融合** — 结构化知识库 + 非结构化文档 + Pandas 数据查询
- **RAG 知识库约束** — system prompt 强制"只基于知识库内容回答"，杜绝 LLM 幻觉

## License

MIT
