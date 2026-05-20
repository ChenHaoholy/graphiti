# 🌍 Graphiti-Iran

> 基于 Graphiti + Neo4j 的美伊冲突知识图谱系统，支持新闻抓取、实体关系抽取与局势分析

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green.svg)](https://neo4j.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 常用命令

```bash
# 1. 增量构建/更新图谱，并自动生成社区摘要
python -m src.build

# 2. 问答：系统会自动判断是普通回答还是局势预测
python -m src.ask "你的问题"

# 3. 只抓取 GDELT 数据到 data/events.jsonl，不写入图谱
python -m src.fetch_gdelt --days 7 --limit 20

# 4. 兼容旧入口，等价于 python -m src.build
python -m src.update_graph
```

---

## 📋 目录

- [项目简介](#-项目简介)
- [核心功能](#-核心功能)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [API 配置](#-api-配置)
- [项目结构](#-项目结构)
- [技术栈](#-技术栈)

---

## 📖 项目简介

Graphiti-Iran 是一个专注于**美伊冲突及中东地区局势**的知识图谱系统。通过整合 GDELT 全球新闻数据，利用大语言模型进行实体关系抽取，构建可检索、可分析的知识网络。

### 应用场景

- 📰 **舆情监测** - 追踪美伊关系动态
- 🔍 **事件溯源** - 检索历史事件与关联证据
- 📊 **风险评估** - 基于图谱进行局势推演
- 🧠 **关系挖掘** - 发现多国博弈中的隐藏关系

---

## ✨ 核心功能

| 功能 | 描述 |
|------|------|
| 🔄 **图谱更新** | 支持本地 JSONL 文件导入和 GDELT 实时抓取 |
| 🔎 **智能检索** | 基于向量相似度的证据检索，支持多维度查询 |
| 💬 **智能问答** | 自然语言提问，自动生成局势推演报告 |
| 🕸️ **关系抽取** | LLM 自动识别实体、事件与关系类型 |
| 📦 **证据管理** | 结构化存储证据来源，支持去重与溯源 |

---

## 🏗️ 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   GDELT     │     │   Local     │     │   User     │
│   News API  │     │   JSONL     │     │   Query    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   ▼
          ┌────────────────┐
          │  Graphiti Core │
          │   (LLM + NLP)  │
          └────────┬───────┘
                   ▼
          ┌────────────────┐
          │     Neo4j      │
          │   Knowledge    │
          │     Graph      │
          └────────────────┘
```

### 数据流

1. **数据采集** → GDELT API / 本地文件
2. **实体抽取** → LLM 识别实体与关系
3. **向量化** → Embedding API 生成向量
4. **存储** → Neo4j 图数据库
5. **检索** → 向量相似度搜索 + 图遍历

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- Neo4j 5.x (本地或 Docker)
- DeepSeek API (LLM)
- 智谱 AI API (Embeddings)

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API 密钥。

### 4. 启动 Neo4j

**Docker 方式：**
```bash
docker run --name graphiti-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password_here \
  neo4j:5
```

**或使用 Neo4j Desktop** 启动本地数据库。

### 5. 更新图谱

```bash
# 从本地文件更新
python -m src.build

# 从 GDELT 抓取最新新闻
python -m src.fetch_gdelt --days 14 --limit 50
```

### 6. 开始提问

```bash
python -m src.ask "未来30天美伊冲突会升级吗？"
```

---

## 📖 使用指南

### 更新图谱命令

```bash
# 基本用法
python -m src.build

# GDELT 实时抓取
python -m src.fetch_gdelt --days 14 --limit 50

# 常用参数
# src.build: --events 指定 JSONL；--no-communities 跳过社区摘要
# src.fetch_gdelt: --days 抓取天数；--limit 每个 query 最大文章数
```

### 提问命令

```bash
# 基本提问
python -m src.ask "美国制裁对伊朗经济影响有多大？"

# 指定时间窗口和证据数量
python -m src.ask "伊朗核协议前景如何？" --window 60 --limit 10

# 输出 JSON 格式
python -m src.ask "中俄对美伊冲突的态度？" --json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window` | 30 | 检索时间窗口（天） |
| `--limit` | 10 | 每个 query 返回的最大证据数 |

---

## 🔑 API 配置

### LLM 配置（用于实体/关系抽取）

```env
LLM_API_KEY=sk-your-deepseek-api-key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-v4-flash
```

### Embedding 配置（用于向量化）

```env
EMBEDDING_API_KEY=your-zhipu-api-key
EMBEDDING_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EMBEDDING_MODEL=embedding-3
EMBEDDING_DIM=1024
```

### Neo4j 配置

```env
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
```

---

## 📁 项目结构

```
graphiti-iran/
├── data/
│   ├── events.jsonl          # 美伊冲突事件数据
│   └── test_events.jsonl     # 测试用事件数据
├── src/
│   ├── __init__.py
│   ├── ask.py                # 💬 智能问答模块
│   ├── config.py             # ⚙️ 配置管理
│   ├── evidence_search.py    # 🔍 证据检索
│   ├── fetch_gdelt.py        # 🌐 GDELT 数据抓取
│   ├── graphiti_client.py    # 🔗 Graphiti 核心客户端
│   ├── schema.py             # 📐 数据模型定义
│   └── update_graph.py       # 🔄 图谱更新入口
├── outputs/                  # 📦 输出目录
├── .env.example              # 📝 环境变量模板
├── requirements.txt          # 📦 Python 依赖
└── README.md                 # 📚 项目文档
```

### 核心模块说明

| 模块 | 功能 |
|------|------|
| `ask.py` | 接收用户问题，生成检索计划，调用图谱检索，返回局势推演 |
| `update_graph.py` | 读取数据源，调用 Graphiti 写入 Neo4j |
| `evidence_search.py` | 基于向量检索，从图谱获取相关证据 |
| `fetch_gdelt.py` | GDELT API 客户端，抓取全球新闻 |
| `graphiti_client.py` | Graphiti 核心封装，处理实体抽取与存储 |

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **图数据库** | Neo4j 5.x |
| **图谱框架** | graphiti-core |
| **大语言模型** | DeepSeek V4 Flash |
| **Embedding** | 智谱 Embedding-3 |
| **数据源** | GDELT Global News |
| **语言** | Python 3.11+ |

---

## 📝 数据格式

### 事件数据 (JSONL)

每行一个 JSON 对象：

```json
{
  "event_id": "unique_event_id",
  "source_url": "https://example.com/news",
  "source_name": "News Source",
  "event_time": "2024-01-15T10:30:00Z",
  "content": "事件描述文本...",
  "actors": ["美国", "伊朗"],
  "event_type": "military"
}
```

### 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `event_id` | ✅ | 事件唯一标识 |
| `source_url` | ✅ | 新闻来源链接 |
| `source_name` | ❌ | 来源媒体名称 |
| `event_time` | ✅ | 事件时间 (ISO 8601) |
| `content` | ✅ | 事件描述文本 |
| `actors` | ❌ | 涉及的行为者 |
| `event_type` | ❌ | 事件类型 |

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License
