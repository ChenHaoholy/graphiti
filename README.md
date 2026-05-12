# graphiti-iran-minimal

基于 `graphiti-core` 的美伊冲突知识图谱项目，用 Neo4j 作为后端。

## 环境要求

- Python 3.11+
- Neo4j 5.x
- `graphiti-core`
- DeepSeek 或其他 OpenAI-compatible chat API
- 智谱或其他 OpenAI-compatible embeddings API

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

复制环境变量模板并编辑：

```powershell
copy .env.example .env
```

配置 `.env`：

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
```

## 使用

### 更新图谱

```bash
python -m src.update_graph --source data/events.jsonl
```

### 提问

```bash
python -m src.ask "未来30天美伊冲突会升级吗？"
```

### 抓取 GDELT 新闻

```bash
python -m src.update_graph --source gdelt --days 14 --limit 50
```

## 项目结构

```
.
├── data/
│   ├── events.jsonl
│   └── test_events.jsonl
├── src/
│   ├── __init__.py
│   ├── ask.py              # 提问入口
│   ├── config.py           # 配置
│   ├── evidence_search.py # 证据检索
│   ├── fetch_gdelt.py      # GDELT 抓取
│   ├── graphiti_client.py  # Graphiti 客户端
│   ├── schema.py           # 数据模型
│   └── update_graph.py     # 更新图谱
├── outputs/
├── README.md
├── requirements.txt
└── .env.example
```
