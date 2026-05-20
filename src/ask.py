"""
ask.py — 用户入口：基于已有 Graphiti 图谱 + LLM 生成局势推演。

只读操作（不调用 add_episode、batch_ingest、update_graph）。
用户问题 → LLM 生成检索计划 → Graphiti.search → 证据 → LLM 回答。

CLI：
  python -m src.ask "用户问题"
  python -m src.ask "用户问题" --window 365 --limit 10 --json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.evidence_search import search_evidence
    from src.config import get_llm_api_key, get_llm_base_url, get_llm_model
except ImportError:
    from evidence_search import search_evidence
    from config import get_llm_api_key, get_llm_base_url, get_llm_model

# ─────────────────────────────────────────────────────────────────────────────
# LLM 调用
# ─────────────────────────────────────────────────────────────────────────────

async def _llm_complete(
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    api_key = get_llm_api_key()
    base_url = get_llm_base_url() or "https://api.openai.com/v1"
    model = get_llm_model() or "gpt-4o-mini"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_err = None
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < 3:
                await asyncio.sleep(2 ** attempt)
    raise last_err


ROUTER_SYSTEM = """Classify the user question for this evidence-grounded graph QA system.

Return only JSON:
{
  "answer_mode": "forecast | factual",
  "reason": "short reason"
}

Use "forecast" only when the user asks about future scenarios, probability, escalation,
risk level, triggers, or what may happen next.
Use "factual" for normal non-prediction questions: what happened, who is involved,
why it matters, historical/current explanations, evidence lookup, summaries, and comparisons.
"""


def fallback_answer_mode(question: str) -> str:
    lowered = question.lower()
    forecast_markers = [
        "future",
        "forecast",
        "predict",
        "prediction",
        "probability",
        "scenario",
        "risk",
        "escalate",
        "next",
        "will",
        "未来",
        "预测",
        "推演",
        "概率",
        "可能会",
        "会不会",
        "升级",
        "接下来",
        "风险",
    ]
    return "forecast" if any(marker in lowered for marker in forecast_markers) else "factual"


async def route_question(question: str) -> dict[str, str]:
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        raw = await _llm_complete(messages, temperature=0.0, max_tokens=160)
        route = json.loads(raw)
    except Exception:
        route = {
            "answer_mode": fallback_answer_mode(question),
            "reason": "fallback keyword routing",
        }

    answer_mode = str(route.get("answer_mode", "")).strip().lower()
    if answer_mode not in {"forecast", "factual"}:
        answer_mode = fallback_answer_mode(question)
    route["answer_mode"] = answer_mode
    route.setdefault("reason", "")
    return route


# ─────────────────────────────────────────────────────────────────────────────
# 第一步：LLM 生成检索计划
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """你是情报检索规划助手。根据用户问题生成图谱检索计划。

主题：美伊冲突及外溢影响（美国、伊朗、中国、俄罗斯、以色列、中东各方）。

输出 JSON：
{
  "question_type": "类型",
  "target_actors": ["相关方"],
  "event_focus": "1-2句描述",
  "time_window_days": 数字,
  "search_queries": ["query1", "query2", ...]  // 3-6个英文检索词
}"""


async def generate_search_plan(question: str, default_window: int) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": f"用户问题：{question}\n\n请生成检索计划。"},
    ]
    raw = await _llm_complete(messages, temperature=0.2, max_tokens=512)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    plan = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            # 兜底：生成通用查询
            plan = {
                "question_type": "general",
                "target_actors": [],
                "event_focus": question,
                "time_window_days": default_window,
                "search_queries": [
                    "Iran military strike attack conflict",
                    "Middle East regional tension escalation",
                    "diplomatic response Iran conflict",
                ],
                "required_evidence": [],
            }

    # 确保字段完整
    plan.setdefault("search_queries", [])
    plan.setdefault("time_window_days", default_window)
    return plan


def _resolve_evidence_window_days(answer_mode: str, window_arg: int | None) -> int | None:
    if window_arg is not None:
        return window_arg if window_arg > 0 else None
    return 365 if answer_mode == "forecast" else 30


def _format_start_time(now: datetime, window_days: int | None) -> str | None:
    if window_days is None:
        return None
    return (now - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_entity(entity: Any) -> str:
    return str(entity or "").strip()


def _extract_entities_from_content(content: str) -> set[str]:
    import re

    entities = set()
    for field in ("Actors", "Targets", "Locations"):
        match = re.search(rf"^{field}:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        for value in match.group(1).split(","):
            entity = _normalize_entity(value)
            if entity and entity.lower() != "none":
                entities.add(entity)

    entities.update(_extract_actors_from_content(content))
    return entities


def _load_event_entity_sets() -> list[set[str]]:
    events_path = Path(__file__).resolve().parents[1] / "data" / "events.jsonl"
    if not events_path.exists():
        return []

    entity_sets: list[set[str]] = []
    try:
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entities = set()
                for field in ("actors", "targets", "locations"):
                    values = event.get(field) or []
                    if isinstance(values, str):
                        values = [values]
                    for value in values:
                        entity = _normalize_entity(value)
                        if entity and entity.lower() != "none":
                            entities.add(entity)

                if len(entities) >= 2:
                    entity_sets.append(entities)
    except OSError:
        return []

    return entity_sets


def _entity_matches(seed: str, entity: str) -> bool:
    seed_lower = seed.lower()
    entity_lower = entity.lower()
    return seed_lower == entity_lower or seed_lower in entity_lower or entity_lower in seed_lower


def _expand_related_entities(plan: dict[str, Any], items: list[dict[str, Any]], max_entities: int = 6) -> list[str]:
    seeds = {_normalize_entity(actor) for actor in plan.get("target_actors", []) if _normalize_entity(actor)}
    evidence_entity_sets = []
    for item in items:
        entities = _extract_entities_from_content(item.get("content", "") or "")
        if entities:
            seeds.update(entities)
            evidence_entity_sets.append(entities)

    if not seeds:
        return []

    scores: dict[str, int] = {}
    for entity_set in [*evidence_entity_sets, *_load_event_entity_sets()]:
        if not any(_entity_matches(seed, entity) for seed in seeds for entity in entity_set):
            continue

        for entity in entity_set:
            if any(_entity_matches(seed, entity) for seed in seeds):
                continue
            scores[entity] = scores.get(entity, 0) + 1

    return sorted(scores, key=lambda entity: (-scores[entity], entity))[:max_entities]


def build_supplementary_searches(
    question: str,
    plan: dict[str, Any],
    items: list[dict[str, Any]],
    quality: dict[str, Any],
    now: datetime,
    original_window_days: int | None,
    base_limit: int,
) -> list[dict[str, Any]]:
    """按固定策略生成补充检索：扩展实体范围。"""
    original_start_time = _format_start_time(now, original_window_days)
    base_query = next((str(q) for q in plan.get("search_queries", []) if q), "")
    focus_terms = base_query or str(plan.get("event_focus") or question)

    searches: list[dict[str, Any]] = []
    for entity in _expand_related_entities(plan, items):
        searches.append(
            {
                "query": f"{entity} {focus_terms}",
                "start_time": original_start_time,
                "limit": base_limit,
                "strategy": f"扩展实体范围：{entity}",
            }
        )

    seen = set()
    deduped = []
    for search in searches:
        key = (search["query"], search["start_time"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(search)

    return deduped[:8]


# ─────────────────────────────────────────────────────────────────────────────
# 第三步：评估证据是否充分
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_CHECK_SYSTEM = """评估证据是否足以回答用户问题。只基于提供的证据，不要编造。

输出 JSON：
{
  "sufficiency": "sufficient | insufficient | marginal",
  "gap_note": "如不足，说明关键缺口"
}"""


async def check_evidence_sufficiency(question: str, evidence_summary: str) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": EVIDENCE_CHECK_SYSTEM},
        {"role": "user", "content": f"用户问题：{question}\n\n证据摘要：\n{evidence_summary}\n\n请评估证据充分性。"},
    ]
    raw = await _llm_complete(messages, temperature=0.1, max_tokens=256)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"sufficiency": "insufficient", "gap_note": "无法解析评估结果"}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 第四步：精细化证据充分性评估
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_SUFFICIENCY_THRESHOLD = 0.68
GENERIC_RELEVANCE_TERMS = {
    "future",
    "forecast",
    "prediction",
    "predict",
    "next",
    "days",
    "day",
    "outlook",
    "scenario",
    "scenarios",
    "未来",
    "局势",
    "演变",
    "趋势",
    "发展",
    "变化",
    "可能",
    "接下来",
    "未来天",
    "天局",
    "势会",
    "会如",
    "何演",
}


def _extract_actors_from_content(content: str) -> set[str]:
    """从文本内容中提取涉及的行为者（国家/组织）。"""
    import re

    actors = set()

    actor_line = re.search(r"^Actors:\s*(.+)$", content, re.IGNORECASE | re.MULTILINE)
    if actor_line:
        for actor in actor_line.group(1).split(","):
            actor = actor.strip()
            if actor and actor.lower() != "none":
                actors.add(actor)

    actor_patterns = {
        "United States": r"\b(?:United States|USA|U\.S\.|US|America|American)\b",
        "Iran": r"\b(?:Iran|Iranian|Persia)\b",
        "Israel": r"\b(?:Israel|Israeli)\b",
        "China": r"\b(?:China|Chinese)\b",
        "Russia": r"\b(?:Russia|Russian)\b",
        "United Kingdom": r"\b(?:United Kingdom|UK|Britain|British)\b",
        "France": r"\b(?:France|French)\b",
        "Germany": r"\b(?:Germany|German)\b",
        "Saudi Arabia": r"\b(?:Saudi Arabia|Saudi)\b",
        "United Nations": r"\b(?:United Nations|UN)\b",
        "European Union": r"\b(?:European Union|EU)\b",
        "NATO": r"\bNATO\b",
        "Hezbollah": r"\b(?:Hezbollah|Hizballah)\b",
        "Hamas": r"\bHamas\b",
        "IRGC": r"\b(?:IRGC|Islamic Revolutionary Guard)\b",
        "GCC": r"\b(?:Gulf Cooperation|GCC)\b",
        "OPEC": r"\bOPEC\b",
    }
    for actor, pattern in actor_patterns.items():
        if re.search(pattern, content, re.IGNORECASE):
            actors.add(actor)
    return actors


def _extract_terms(text: str) -> set[str]:
    """抽取用于粗略相关性评估的中英文关键词。"""
    import re

    lowered = text.lower()
    terms = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", text)}

    cjk_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    for chunk in cjk_chunks:
        if len(chunk) <= 4:
            terms.add(chunk)
            continue
        for i in range(len(chunk) - 1):
            terms.add(chunk[i : i + 2])

    alias_groups = [
        ("美国", "美方", "united", "states", "usa", "us", "america", "american"),
        ("伊朗", "iran", "iranian"),
        ("以色列", "israel", "israeli"),
        ("中国", "china", "chinese"),
        ("俄罗斯", "russia", "russian"),
        ("制裁", "sanction", "sanctions"),
        ("袭击", "打击", "攻击", "attack", "strike", "strikes"),
        ("核", "核计划", "nuclear"),
        ("霍尔木兹", "hormuz"),
        ("谈判", "外交", "negotiation", "talks", "diplomacy"),
        ("风险", "升级", "risk", "escalation", "escalate"),
    ]
    for aliases in alias_groups:
        if any(alias in lowered or alias in terms for alias in aliases):
            terms.update(aliases)

    stop_terms = {
        "the",
        "and",
        "for",
        "with",
        "what",
        "why",
        "how",
        "of",
        "to",
        "in",
        "on",
        "is",
        "are",
        "哪些",
        "什么",
        "是否",
        "如何",
        "可以",
        "证据",
        "显示",
    }
    return {term for term in terms if term not in stop_terms}


def _domain_terms(text: str) -> set[str]:
    return _extract_terms(text) - GENERIC_RELEVANCE_TERMS


def _average_relevance(items: list[dict[str, Any]], question: str) -> float:
    question_terms = _domain_terms(question)
    query_terms_by_item = [_domain_terms(item.get("matched_query", "") or "") for item in items]
    if not question_terms and not any(query_terms_by_item):
        return 0.5

    scores = []
    for item, query_terms in zip(items, query_terms_by_item):
        content_text = " ".join(
            [
                item.get("content", "") or "",
                item.get("source_name", "") or "",
            ]
        )
        item_terms = _domain_terms(content_text)
        if not item_terms:
            scores.append(0.0)
            continue

        score_parts = []
        if question_terms:
            score_parts.append(len(question_terms & item_terms) / len(question_terms))
        if query_terms:
            score_parts.append(0.85 * len(query_terms & item_terms) / len(query_terms))

        scores.append(min(max(score_parts) if score_parts else 0.0, 1.0))

    return sum(scores) / len(scores) if scores else 0.0


def _unique_evidence_count(items: list[dict[str, Any]]) -> int:
    seen = set()
    for item in items:
        key = item.get("source_url") or item.get("content") or repr(item)
        seen.add(str(key).strip())
    return len(seen)


def _source_key(item: dict[str, Any]) -> str | None:
    source_name = item.get("source_name")
    if source_name:
        return str(source_name).strip()

    source_url = item.get("source_url")
    if source_url:
        from urllib.parse import urlparse

        hostname = urlparse(str(source_url)).hostname
        return hostname.strip() if hostname else str(source_url).strip()

    return None


def _detect_contradictions(items: list[dict[str, Any]]) -> float:
    """
    检测矛盾信息。
    简单策略：同一时间、同一事件类型但不同结果的条目视为潜在矛盾。
    返回矛盾得分（0=完全一致，1=高度矛盾）。
    """
    if len(items) < 2:
        return 0.0

    contradiction_pairs = 0
    total_pairs = 0

    import re

    for i in range(len(items)):
        content_i = items[i].get("content", "")
        time_i = items[i].get("event_time", "")
        # 提取事件类型关键词
        strike_kw = re.findall(r"(?:strike|attack|raid|hit|bomb)", content_i, re.IGNORECASE)
        if not strike_kw:
            continue

        for j in range(i + 1, len(items)):
            content_j = items[j].get("content", "")
            time_j = items[j].get("event_time", "")

            # 时间相同但结果矛盾
            if time_i and time_j and time_i != time_j:
                continue

            strike_j = re.findall(r"(?:strike|attack|raid|hit|bomb)", content_j, re.IGNORECASE)
            if strike_j:
                # 一个说"成功"一个说"失败"算矛盾
                success_i = re.search(r"(?:success|successful|hit|destro)", content_i, re.IGNORECASE)
                fail_i = re.search(r"(?:fail|intercepted|shot down|counter)", content_i, re.IGNORECASE)
                success_j = re.search(r"(?:success|successful|hit|destro)", content_j, re.IGNORECASE)
                fail_j = re.search(r"(?:fail|intercepted|shot down|counter)", content_j, re.IGNORECASE)

                if (success_i and fail_j) or (fail_i and success_j):
                    contradiction_pairs += 1
            total_pairs += 1

    if total_pairs == 0:
        return 0.0
    return min(contradiction_pairs / total_pairs, 1.0)


def evaluate_evidence_quality(items: list[dict[str, Any]], question: str = "") -> dict[str, Any]:
    """
    从多个维度评估证据是否足以回答问题。

    核心原则：
    1. 单条证据不能仅因有来源和时间就被判定充分；
    2. 至少需要基本相关性、可交叉验证的证据量和独立来源；
    3. 时间戳、行为者覆盖和矛盾检测作为质量修正项。
    """
    if not items:
        return {
            "score": 0.0,
            "sufficient": False,
            "dimensions": {
                "source_diversity": 0.0,
                "time_coverage": 0.0,
                "entity_coverage": 0.0,
                "consistency": 0.0,
                "evidence_volume": 0.0,
                "relevance": 0.0,
            },
            "gap_note": "图谱中无相关证据",
        }

    unique_evidence = _unique_evidence_count(items)
    evidence_volume = min(unique_evidence / 4, 1.0)

    # 1. 与问题的相关性
    relevance = _average_relevance(items, question)

    # 2. 来源多样性：要求独立来源数量，同时惩罚单一来源占比过高
    sources = [source for item in items if (source := _source_key(item))]
    unique_sources = set(sources)
    source_count_score = min(len(unique_sources) / 3, 1.0)
    if sources:
        max_source_share = max(sources.count(source) for source in unique_sources) / len(sources)
        source_balance = 1.0 - max(0.0, (max_source_share - 0.5) / 0.5)
    else:
        source_balance = 0.0
    source_diversity = source_count_score * 0.75 + source_balance * 0.25

    # 3. 时间覆盖度：这里评估的是时间元数据完整度，不等同于时间跨度
    times = [item.get("event_time") for item in items if item.get("event_time")]
    time_coverage = len(times) / len(items) if items else 0.0

    # 4. 实体覆盖度（从所有内容中提取行为者）
    all_actors: set[str] = set()
    for item in items:
        content = item.get("content", "") or ""
        all_actors.update(_extract_actors_from_content(content))
    ENTITY_BASELINE = 4
    entity_coverage = min(len(all_actors) / ENTITY_BASELINE, 1.0)

    # 5. 矛盾信息处理。单条证据没有交叉验证基础，不能直接给满分。
    consistency = 0.5 if unique_evidence < 2 else 1.0 - _detect_contradictions(items)

    score = (
        evidence_volume * 0.20
        + relevance * 0.25
        + source_diversity * 0.20
        + time_coverage * 0.15
        + entity_coverage * 0.10
        + consistency * 0.10
    )
    score = round(score, 3)
    gate_failures = []
    if unique_evidence < 2:
        gate_failures.append(f"证据量不足（{unique_evidence}条可区分证据，至少需要2条）")
    if relevance < 0.25:
        gate_failures.append("证据与问题关键词重合较低，可能未覆盖核心问题")
    if source_diversity < 0.4:
        gate_failures.append(f"独立来源不足（{len(unique_sources)}个来源，至少建议2个）")
    if time_coverage < 0.5:
        gate_failures.append(f"时间元数据不足（{len(times)}/{len(items)}条有时间戳）")
    if consistency < 0.6:
        gate_failures.append("证据间缺少最低交叉验证或存在明显矛盾")

    hard_gates_pass = not gate_failures
    sufficient = score >= EVIDENCE_SUFFICIENCY_THRESHOLD and hard_gates_pass

    gap_parts = list(gate_failures)
    if entity_coverage < 0.35:
        gap_parts.append(f"实体覆盖不足（{len(all_actors)}个行为者）")
    if 0.6 <= consistency < 0.7:
        gap_parts.append("证据间缺少交叉验证或存在矛盾信息")

    gap_note = "; ".join(gap_parts) if gap_parts else "各维度均达标"

    return {
        "score": score,
        "sufficient": sufficient,
        "dimensions": {
            "source_diversity": round(source_diversity, 3),
            "time_coverage": round(time_coverage, 3),
            "entity_coverage": round(entity_coverage, 3),
            "consistency": round(consistency, 3),
            "evidence_volume": round(evidence_volume, 3),
            "relevance": round(relevance, 3),
        },
        "gate_failures": gate_failures,
        "unique_evidence": unique_evidence,
        "unique_sources": list(unique_sources),
        "detected_actors": list(all_actors),
        "gap_note": gap_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 第四步（原）：生成最终推演
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """你是一个专业的美伊冲突局势分析助手，基于知识图谱中的真实证据为用户提供局势推演。

【核心约束】
1. 只能基于 evidence_context 中提供的证据，禁止编造图谱中不存在的事件、数据或引用。
2. 如果证据不足，明确说明哪些方面证据缺失。
3. 输出的是局势推演（可能情景），而非确定性预言。
4. 语言简洁专业，适合决策参考。
5. 严格区分"有证据支持的事实陈述"和"基于证据的推演判断"。

【输出格式】（严格按以下字段输出，不要添加额外章节）
- 用户问题：（原样引用用户问题）
- 简短结论：（2~3 句话总结当前局势，引用直接嵌入句子中间，格式为"内容（来源, 日期）"，不要用箭头或列表）
- 风险等级：（low / medium / high / uncertain）
- 主要依据：（2~4 条，每条将证据内容与推断结论融合在一个段落中，引用嵌入句子内，不要用列表或箭头）
- 历史类比：（从 evidence_context 中找 1~3 个历史事件或历史模式做类比，说明相似点、差异点以及对当前推演的启发；如果证据中没有可类比历史事件，明确写“当前证据中缺少可用历史类比”）
- 可能情景：（列出 2~3 种可能的发展情景，每条单独一行并用 1. 2. 3. 编号，用"较高概率""中等概率""小概率"标注，不要用数字百分比，也不要用**加粗）
- 触发条件：（什么情况下当前趋势会改变，列出 1~3 条，每条单独一行并用 1. 2. 3. 编号）
- 缓和信号：（什么迹象表明局势可能缓和，列出 1~3 条，每条单独一行并用 1. 2. 3. 编号）

【引用要求】
- 引用必须直接嵌入句子中间，格式："...结论（来源, YYYY-MM-DD）"
- 不要在句末用列表或箭头，引用是句子的一部分而非补充说明
- 主要依据字段中，每条证据要写成一个完整段落：先引用原文内容，再接推断链，最后出结论
- 历史类比必须来自 evidence_context 中已有事件，不要引入图谱外历史知识

【完整性要求】
- 即使证据不足，也必须输出“可能情景”“触发条件”“缓和信号”三个字段。
- 如果无法从证据中判断某个字段，写明“当前证据不足以判断”，不要省略字段。

请严格按上述字段输出，不要省略任何一个字段。"""


GENERAL_ANSWER_SYSTEM = """You are an evidence-grounded analyst for a US-Iran conflict knowledge graph.

Rules:
1. Answer only from the provided evidence_context. Do not invent events, numbers, sources, or quotes.
2. This is not a forecast template. Do not force risk levels, scenarios, triggers, or future predictions unless the user explicitly asks.
3. Give a complete factual answer when evidence supports it: define the entity/event, explain the time/background, and connect the evidence into a coherent explanation.
4. If the evidence is thin or does not contain the exact fact requested, say exactly what is missing and avoid filling gaps from memory.
5. Answer in the same language as the user's question.

Output format:
- 用户问题: quote the user's question
- 简短回答: direct answer in 3-6 sentences. If the exact answer is in evidence, state it clearly first.
- 背景说明: explain relevant context from the evidence in 1-3 short paragraphs
"""


async def generate_answer(
    question: str,
    evidence_text: str,
    search_plan: dict[str, Any],
    answer_mode: str = "forecast",
) -> str:
    focus_note = search_plan.get("event_focus", "")
    actors = ", ".join(search_plan.get("target_actors", []))
    focus_block = f"\n\nSearch focus: {focus_note}. Actors: {actors}." if focus_note else ""
    system_prompt = ANSWER_SYSTEM if answer_mode == "forecast" else GENERAL_ANSWER_SYSTEM

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"User question: {question}{focus_block}\n\nEvidence context:\n{evidence_text}",
        },
    ]
    return await _llm_complete(messages, temperature=0.3, max_tokens=2048)



def format_evidence_context(items: list[dict[str, Any]]) -> str:
    if not items:
        return "（图谱中无相关证据）"

    lines = []
    for i, item in enumerate(items, 1):
        content = (item.get("content") or "").strip()
        source = item.get("source_name") or "未知来源"
        url = item.get("source_url") or ""
        event_time = item.get("event_time") or "时间未知"
        matched = item.get("matched_query", "")

        block = f"[证据{i}] 时间：{event_time} | 来源：{source}"
        if url:
            block += f" | {url}"
        if matched:
            block += f" | 检索词：{matched}"
        block += f"\n{content}"
        lines.append(block)

    return "\n\n".join(lines)


def summarize_evidence(items: list[dict[str, Any]], max_chars: int = 800) -> str:
    """生成证据摘要，用于传给补充检索 LLM。"""
    if not items:
        return "图谱中无相关证据。"
    parts = []
    for item in items[:5]:
        time = item.get("event_time", "?")
        content = (item.get("content") or "")[:120]
        parts.append(f"[{time}] {content}")
    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."
    return summary


def parse_answer_text(text: str, question: str) -> dict[str, Any]:
    """从 LLM 输出文本中解析结构化字段。"""
    structured: dict[str, str] = {"user_question": question, "answer_raw": text}
    current_key: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # 匹配 "- 字段名：" 或 "字段名：" 开头的行
        for prefix in ("- ", "• ", "* "):
            if line.startswith(prefix):
                line = line[len(prefix):]
                break

        if "：" in line or ":" in line:
            sep = "：" if "：" in line else ":"
            parts = line.split(sep, 1)
            key = parts[0].strip().lower()
            value = parts[1].strip() if len(parts) > 1 else ""

            if current_key and current_lines:
                structured[current_key] = "\n".join(current_lines).strip()

            current_key = key
            current_lines = [value] if value else []
        elif current_key:
            current_lines.append(line)

    if current_key and current_lines:
        structured[current_key] = "\n".join(current_lines).strip()

    # 清理所有字段中的 markdown 加粗
    for key in structured:
        if isinstance(structured[key], str):
            structured[key] = structured[key].replace("**", "")

    return structured


def _center(text: str, width: int = 64) -> str:
    return f"{' ' * ((width - len(text)) // 2)}{text}"


def _display_width(text: str) -> int:
    import unicodedata

    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def _wrap_display(text: str, width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    current_width = 0

    for char in text:
        char_width = _display_width(char)
        if current and current_width + char_width > width:
            lines.append(current.rstrip())
            current = char
            current_width = char_width
        else:
            current += char
            current_width += char_width

    if current.strip():
        lines.append(current.rstrip())
    return lines or [""]


def _section(title: str, body: str, width: int = 64, indent: str = "  ") -> str:
    """渲染一个独立字段：标题行 + 内容缩进行。"""
    lines = []
    lines.append(f"{indent}{title}")
    for ln in body.splitlines():
        ln = ln.strip()
        if ln:
            lines.append(f"{indent}  {ln}")
    return "\n".join(lines)


def _render_scenarios(text: str, width: int = 64) -> str:
    """从原始文本中提取"可能情景/触发条件/缓和信号"三个字段并美化输出。"""
    lines = []
    segments = [
        ("可能情景", "可能情景"),
        ("触发条件", "触发条件"),
        ("缓和信号", "缓和信号"),
    ]
    stop_fields = {"主要依据", "用户问题"}

    raw_lines = text.splitlines()

    for field_key, label in segments:
        start = -1
        end = -1
        for i, ln in enumerate(raw_lines):
            stripped = ln.strip().lstrip("-•*").strip()
            key_part = stripped.split("：")[0].split(":")[0].strip().lower()
            if key_part == field_key.lower():
                start = i
            elif start >= 0 and end < 0:
                next_stripped = stripped.lstrip("-•*").strip()
                if next_stripped and ("：" in next_stripped or ":" in next_stripped):
                    next_key = next_stripped.split("：")[0].split(":")[0].strip().lower()
                    if next_key in [f.lower() for f, _ in segments] or next_key in stop_fields:
                        end = i
                        break
                # 额外防护：即使没有冒号，行首出现截断关键词也结束
                for stop in stop_fields:
                    if next_stripped.startswith(stop):
                        end = i
                        break
                if end >= 0:
                    break
        if start >= 0:
            if end < 0:
                end = len(raw_lines)
            body_lines = raw_lines[start:end]
            # 过滤掉截断字段的行
            filtered = []
            for l in body_lines[1:]:
                s = l.strip().lstrip("-•*").strip()
                if any(s.startswith(stop) for stop in stop_fields):
                    continue
                if l.strip():
                    filtered.append(l.strip())
            content = "\n".join(filtered)
            # 清理 markdown 加粗
            content = content.replace("**", "")
            if content:
                lines.append(_section(label, content, width))
                lines.append("")

    return "\n".join(lines).strip()


def _render_evidence_items(items: list[dict[str, Any]], max_items: int = 6) -> str:
    """渲染最多 max_items 条 event_id。"""
    import re

    lines = []
    for i, item in enumerate(items[:max_items], 1):
        content = item.get("content", "")
        m = re.search(r"Event ID[:\s]+([A-Za-z0-9\-]+)", content)
        if m:
            lines.append(f"  {i}. {m.group(1)}")
    return "\n".join(lines)


def _print_block(title: str, body: str, indent: str = "  ") -> None:
    body = body.strip()
    if not body:
        return

    content_width = 88
    print(f"{indent}{title}")
    print(f"{indent}{'─' * max(_display_width(title), 8)}")
    for ln in body.splitlines():
        ln = ln.strip()
        if ln:
            for wrapped in _wrap_display(ln, content_width):
                print(f"{indent}  {wrapped}")
    print()


def _split_display_items(title: str, body: str) -> list[str]:
    import re

    raw_lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(raw_lines) > 1:
        return raw_lines

    text = raw_lines[0] if raw_lines else body.strip()
    if not text:
        return []

    if title == "可能情景":
        pattern = r"(?=(?:较高概率|中等概率|小概率)[:：])"
        parts = [part.strip(" ；;") for part in re.split(pattern, text) if part.strip(" ；;")]
        if len(parts) > 1:
            return parts

    parts = [part.strip() for part in re.split(r"[；;]\s*|(?<=。)\s*", text) if part.strip()]
    return parts if len(parts) > 1 else [text]


def _print_list_block(title: str, body: str, indent: str = "  ") -> None:
    import re

    items = _split_display_items(title, body)
    if not items:
        return

    content_width = 84
    print(f"{indent}{title}")
    print(f"{indent}{'─' * max(_display_width(title), 8)}")
    for index, item in enumerate(items, 1):
        item = item.strip()
        if re.match(r"^\d+[\.、)]\s*", item):
            prefix = ""
            text = item
        else:
            prefix = f"{index}. "
            text = item

        wrapped = _wrap_display(text, content_width)
        if wrapped:
            print(f"{indent}  {prefix}{wrapped[0]}")
            continuation_indent = " " * _display_width(prefix)
            for extra in wrapped[1:]:
                print(f"{indent}  {continuation_indent}{extra}")
    print()


def print_report(
    text: str,
    question: str,
    evidence_count: int,
    plan: dict[str, Any],
    queries: list,
    all_items: list[dict[str, Any]],
    answer_mode: str = "",
    quality: dict[str, Any] | None = None,
) -> None:
    w = 72
    bar = "─" * w
    is_forecast = answer_mode == "forecast"

    print()
    print(f"  {bar}")
    print(f"  Q  {question}")
    print(f"  {bar}")
    print()

    # 检索维度（用户关心的，一行一个）
    if is_forecast and queries:
        print(f"  检索维度 ({len(queries)})")
        print(f"  {'─' * (len('检索维度') + len(str(len(queries))) + 3)}")
        for i, q in enumerate(queries, 1):
            print(f"    {i}. {q}")
        print()

    # 从文本中解析结构化字段
    parsed = parse_answer_text(text, question)

    # 结论/回答（核心内容，最显眼位置）
    answer_body = parsed.get("简短结论") or parsed.get("简短回答") or parsed.get("回答") or ""
    if answer_body:
        _print_block("结论" if is_forecast else "回答", answer_body)

    if not is_forecast:
        for key in ("背景说明",):
            body = parsed.get(key, "").strip()
            if body:
                _print_block(key, body)
        return

    # 风险等级
    risk = parsed.get("风险等级", "").strip()
    if risk:
        print(f"  风险等级")
        print(f"  {'─' * len('风险等级')}")
        print(f"    {risk}")
        print()

    historical_analogy = parsed.get("历史类比", "").strip()
    if historical_analogy:
        _print_block("历史类比", historical_analogy)

    for key in ("可能情景", "触发条件", "缓和信号"):
        value = parsed.get(key, "").strip()
        if value:
            _print_list_block(key, value)

    # 底栏
    print(f"  {bar}")
    mode_tag = f"[推演]" if answer_mode == "forecast" else "[问答]"
    print(f"  {mode_tag} {evidence_count} 条证据 | {answer_mode}")
    print(f"  {bar}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="[用户] 基于 Graphiti 图谱 + LLM 生成证据问答或局势推演。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例：
  python -m src.ask "未来30天美伊冲突会升级吗？"
  python -m src.ask "美国制裁影响多大？" --window 60 --limit 10 --json
  python -m src.ask "未来30天局势会如何演变" --window 365
  python -m src.ask "最近有哪些证据显示霍尔木兹海峡紧张？"
  python -m src.ask "伊朗打击某军事基地，中国反应如何？" --json
""",
    )
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="证据回看窗口天数；不传时事实问答默认30天、推演默认365天；传0表示不限制时间",
    )
    parser.add_argument("--limit", type=int, default=10, help="每个 query 最大证据数（默认 10）")
    parser.add_argument("--json", action="store_true", help="输出纯 JSON 格式")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    if not args.question:
        print("用法: python -m src.ask \"你的问题\" [--window 365] [--limit 10] [--json]")
        return

    question = args.question.strip()
    route = await route_question(question)
    answer_mode = route["answer_mode"]
    mode_label = "局势推演" if answer_mode == "forecast" else "事实问答"
    print(f"  [路由] 问题类型：{mode_label}", file=sys.stderr)
    evidence_window_days = _resolve_evidence_window_days(answer_mode, args.window)

    plan = await generate_search_plan(question, evidence_window_days or 0)
    plan["answer_mode"] = answer_mode
    plan["evidence_window_days"] = evidence_window_days
    queries = plan.get("search_queries", [])
    if not queries:
        print("错误：LLM 未生成有效检索词。", file=sys.stderr)
        sys.exit(1)
    queries = queries[:6]

    now = datetime.now(timezone.utc)
    start_time = _format_start_time(now, evidence_window_days)

    all_items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    supplementary_searches: list[dict[str, Any]] = []

    for q in queries:
        try:
            items = await search_evidence(
                query=q,
                start_time=start_time,
                end_time=None,
                limit=args.limit,
            )
            for item in items:
                item_dict = item.model_dump()
                url = item_dict.get("source_url") or ""
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                item_dict["matched_query"] = q
                all_items.append(item_dict)
        except Exception:
            pass

    evidence_text = format_evidence_context(all_items)
    evidence_summary = summarize_evidence(all_items)

    if not all_items:
        print("当前图谱中没有足够证据支持推演，请先由维护者导入相关事件数据。", file=sys.stderr)
        sys.exit(0)

    # 向用户简要汇报检索结果（stderr，不混入报告正文）
    print(f"  [检索] 首轮命中 {len(all_items)} 条证据", file=sys.stderr)

    # 精细化证据质量评估
    quality = evaluate_evidence_quality(all_items, question)
    sufficiency = {
        "sufficiency": "sufficient" if quality["sufficient"] else "insufficient",
        "gap_note": quality["gap_note"],
    }

    if answer_mode == "forecast" and not quality["sufficient"]:
        supplementary_searches = build_supplementary_searches(
            question=question,
            plan=plan,
            items=all_items,
            quality=quality,
            now=now,
            original_window_days=evidence_window_days,
            base_limit=args.limit,
        )

        added = 0
        for search in supplementary_searches:
            q = search["query"]
            strategy_added = 0
            try:
                items = await search_evidence(
                    query=q,
                    start_time=search["start_time"],
                    end_time=None,
                    limit=search["limit"],
                )
                for item in items:
                    item_dict = item.model_dump()
                    url = item_dict.get("source_url") or ""
                    if url and url in seen_urls:
                        continue
                    if url:
                        seen_urls.add(url)
                    item_dict["matched_query"] = q
                    all_items.append(item_dict)
                    added += 1
                    strategy_added += 1
            except Exception:
                pass
            print(f"  [补充] {search['strategy']} | 命中新增 {strategy_added} 条", file=sys.stderr)

        if added:
            print(f"  [检索] 补充命中 {added} 条证据，共 {len(all_items)} 条", file=sys.stderr)
            evidence_text = format_evidence_context(all_items)
            evidence_summary = summarize_evidence(all_items)
            # 重新评估补充后的质量
            quality = evaluate_evidence_quality(all_items, question)
            sufficiency = {
                "sufficiency": "sufficient" if quality["sufficient"] else "insufficient",
                "gap_note": quality["gap_note"],
            }
            if not quality["sufficient"]:
                failures = quality.get("gate_failures") or [quality.get("gap_note", "未达到充分性要求")]
                print(
                    f"  [评估] 补充后综合得分 {quality['score']}，仍不充分：{'；'.join(failures)}",
                    file=sys.stderr,
                )
    elif answer_mode != "forecast" and not quality["sufficient"]:
        print(f"  [评估] 事实问答不做实体补充；当前缺口：{quality['gap_note']}", file=sys.stderr)

    answer_text = await generate_answer(question, evidence_text, plan, answer_mode=answer_mode)

    if args.json:
        structured = parse_answer_text(answer_text, question)
        structured["evidence_count"] = len(all_items)
        structured["search_queries"] = queries
        structured["question_type"] = plan.get("question_type", "")
        structured["answer_mode"] = answer_mode
        structured["route_reason"] = route.get("reason", "")
        structured["evidence_window_days"] = evidence_window_days
        structured["target_actors"] = plan.get("target_actors", [])
        if answer_mode == "forecast":
            structured["supplementary_searches"] = supplementary_searches
            structured["evidence_quality"] = quality
        print(json.dumps(structured, ensure_ascii=False, indent=2))
    else:
        print_report(answer_text, question, len(all_items), plan, queries, all_items, answer_mode=answer_mode, quality=quality)


if __name__ == "__main__":
    asyncio.run(main())
