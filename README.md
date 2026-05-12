# graphiti-iran-minimal

## 项目目标

这是一个基于 `graphiti-core` 官方库的最小可复现项目，用 Neo4j 作为后端，跑通美伊冲突事件的写入、检索、证据包生成和离线规则型风险预测流程。

预测层只读取 `outputs/evidence_bundle.json`，不直接访问 Neo4j，不调用 LLM API，不输出精确概率，不包含爬虫或前端。

## 环境要求

- Python 3.11+
- Neo4j 5.x
- `graphiti-core`
- DeepSeek 或其他 OpenAI-compatible chat API，用于 Graphiti 实体/关系抽取
- 智谱或其他 OpenAI-compatible embeddings API，用于 Graphiti embedding

## Neo4j 启动方式

如果本机已安装 Neo4j Desktop，直接启动一个 Neo4j 5.x database，并确认 Bolt 地址可访问。

也可以使用 Docker：

```bash
docker run --name graphiti-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password_here \
  neo4j:5
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## .env 配置

复制环境变量模板：

```powershell
copy .env.example .env
```

编辑 `.env`：

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

LLM_API_KEY=sk-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash

EMBEDDING_API_KEY=your-zhipu-api-key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=1024
EMBEDDING_DIMENSIONS=1024

RERANKER_API_KEY=
RERANKER_BASE_URL=
RERANKER_MODEL=

SUPPRESS_NEO4J_NOTIFICATIONS=true
```

`LLM_*` 用于 Graphiti 抽取实体和关系。`EMBEDDING_*` 用于 Graphiti 向量化。`RERANKER_*` 可留空，默认复用 `LLM_*`。

---

## 两个主要命令

### 开发者：更新图谱

```bash
python -m src.update_graph --source data/events.jsonl
```

将本地 `data/events.jsonl` 中的事件写入 Graphiti 图谱。自动按 `source_url` / `event_id` 去重，打印成功、跳过、失败数量。

也支持 GDELT 实时抓取：

```bash
python -m src.update_graph --source gdelt --days 14 --limit 50
```

### 用户：提问

```bash
python -m src.ask "未来30天美伊冲突会升级吗？"
```

基于已有图谱检索相关证据，调用离线规则模型生成局势推演。自动识别问题主题（军事、制裁、核、霍尔木兹、外交谈判等），输出风险等级、最可能情景、关键驱动因素和反向信号。

支持参数：`--window 30`（检索天数）、`--limit 10`（证据条数）、`--json`（输出纯 JSON）。

---

## 其他命令（参考）

### 检索证据

```bash
python -m src.evidence_search --query "Iran sanctions" --limit 5
python -m src.evidence_search --actor "IAEA" --event-type "diplomacy"
```

### 生成 Evidence Bundle

```bash
python -m src.evidence_bundle --limit 5 --output outputs/evidence_bundle.json
```

### 离线预测

```bash
python -m src.predict --input outputs/evidence_bundle.json --output outputs/forecast.json
```

### 回测评估

```bash
python -m src.backtest --cases data/backtest_cases.jsonl --output outputs/backtest_report.json
```

### 测试

```bash
pytest
```

这些测试只校验 schema、JSONL 加载和 episode 文本转换，不连接 Neo4j，也不调用模型 API。

---

## 项目目录结构

```text
.
├── data/
│   ├── backtest/
│   │   └── bt*.json
│   ├── backtest_cases.jsonl
│   ├── events.jsonl
│   └── sample_event.json
├── outputs/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── ask.py              # 用户入口：提问 + 局势推演（只读）
│   ├── backtest.py
│   ├── batch_ingest.py
│   ├── config.py
│   ├── evidence_bundle.py
│   ├── evidence_search.py
│   ├── fetch_gdelt.py
│   ├── graphiti_client.py
│   ├── predict.py
│   ├── risk_model.py
│   ├── schema.py
│   ├── smoke_ingest.py
│   ├── smoke_search.py
│   └── update_graph.py     # 开发者入口：更新图谱（写入）
├── tests/
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```
