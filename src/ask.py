"""
ask.py — 用户入口：基于已有 Graphiti 图谱 + LLM 生成局势推演。

只读操作（不调用 add_episode、batch_ingest、update_graph）。
用户问题 → LLM 生成检索计划 → Graphiti.search → 证据 → LLM 回答。

CLI：
  python -m src.ask "用户问题"
  python -m src.ask "用户问题" --window 30 --limit 10 --json
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

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# 第一步：LLM 生成检索计划
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = """你是一个专业的情报检索规划助手。你的任务是根据用户问题，生成最优的图谱检索计划。

【核心原则】
- 项目主题：美伊冲突及其外溢影响，包括美国、伊朗、中国、俄罗斯、以色列、中东各方。
- 用户问题可能涉及任意相关方，不要预设只问美国/伊朗。
- 检索 query 应该覆盖问题的多个维度：直接相关方、背景上下文、可能的影响因素。
- 返回结构化的 JSON 计划，不要输出其他内容。

【输出格式】
请返回如下 JSON 格式（不要有任何额外文字）：
{
  "question_type": "问题类型，如：military_action | diplomatic_response | economic_impact | regional_spread | general_situation",
  "target_actors": ["主要相关方列表"],
  "event_focus": "事件关注点，1-2句话描述",
  "time_window_days": 数字，建议30-90天,
  "search_queries": ["query1", "query2", ...],  // 3到6个英文检索词，越靠前越核心
  "required_evidence": ["证据类型列表，如：官方声明、军事行动记录、外交声明等"]
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
        # JSON 解析失败，尝试提取 JSON 块
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


# ─────────────────────────────────────────────────────────────────────────────
# 第二步：补充检索（如果第一轮证据不足）
# ─────────────────────────────────────────────────────────────────────────────

SUPPLEMENT_SYSTEM = """你是一个情报检索规划助手。用户问题在图谱中检索到的证据不足以回答，你需要生成补充检索词来填补信息缺口。

【输入】
- 用户原始问题
- 已有证据摘要
- 信息缺口描述（由分析 LLM 提供）

【任务】
生成 2-3 个补充检索 query，帮助填补关键信息缺口。
检索词应该针对尚未覆盖的维度。

【输出格式】
只返回 JSON：
{
  "gap_summary": "简要描述当前证据缺口",
  "supplementary_queries": ["补充query1", "补充query2", "补充query3"]
}"""


async def generate_supplementary_queries(
    question: str,
    evidence_summary: str,
    gap_note: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SUPPLEMENT_SYSTEM},
        {"role": "user", "content": f"用户问题：{question}\n\n已有证据摘要：{evidence_summary}\n\n信息缺口：{gap_note}\n\n请生成补充检索 query。"},
    ]
    raw = await _llm_complete(messages, temperature=0.2, max_tokens=384)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"gap_summary": gap_note, "supplementary_queries": []}

    result.setdefault("supplementary_queries", [])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 第三步：评估证据是否充分
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE_CHECK_SYSTEM = """你是一个专业的情报分析助手。你的任务是评估图谱检索到的证据是否足以回答用户问题。

【约束】
- 只能基于提供的证据回答，不要编造不存在的信息。
- 如果证据不足以支持推演，明确指出缺口。
- 你的判断将决定是否需要补充检索。

【输入】
- 用户问题
- 图谱检索到的证据列表

【任务】
判断：
1. 证据是否足以回答用户问题（sufficient / insufficient / marginal）
2. 如果 insufficient 或 marginal，指出关键信息缺口

【输出格式】
只返回 JSON：
{
  "sufficiency": "sufficient | insufficient | marginal",
  "gap_note": "如果不足/边缘，说明缺少哪些关键证据类型或维度"
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
# 第四步：生成最终推演
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_SYSTEM = """你是一个专业的美伊冲突局势分析助手，基于知识图谱中的真实证据为用户提供局势推演。

【核心约束】
1. 只能基于 evidence_context 中提供的证据，禁止编造图谱中不存在的事件、数据或引用。
2. 如果证据不足，明确说明哪些方面证据缺失。
3. 输出的是局势推演（可能情景），而非确定性预言。
4. 语言简洁专业，适合决策参考。
5. 严格区分"有证据支持的事实陈述"和"基于证据的推演判断"。

【输出格式】（严格按以下 8 个字段输出，不要添加额外章节）
- 用户问题：（原样引用用户问题）
- 简短结论：（2~3 句话总结当前局势）
- 风险等级：（low / medium / high / uncertain）
- 主要依据：（列出 2~4 条核心证据，每条注明来源和时间）
- 可能情景：（列出 2~3 种可能的发展情景，注明各自概率）
- 触发条件：（什么情况下当前趋势会改变，列出 1~3 条）
- 缓和信号：（什么迹象表明局势可能缓和，列出 1~3 条）
- 证据不足之处：（指出图谱中缺少哪些方面的信息，对推演有何影响）

请严格按上述 8 个字段输出，不要省略任何一个字段。"""


async def generate_answer(question: str, evidence_text: str, search_plan: dict[str, Any]) -> str:
    focus_note = search_plan.get("event_focus", "")
    focus_block = f"\n\n参考：本次检索关注点为「{focus_note}」，涉及方为「{', '.join(search_plan.get('target_actors', []))}」。" if focus_note else ""

    messages = [
        {"role": "system", "content": ANSWER_SYSTEM},
        {"role": "user", "content": f"用户问题：{question}{focus_block}\n\n证据上下文：\n{evidence_text}"},
    ]
    return await _llm_complete(messages, temperature=0.3, max_tokens=2048)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

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

    return structured


def print_report(text: str, question: str, evidence_count: int, plan: dict[str, Any]) -> None:
    print("=" * 64)
    print(f"  问题：{question}")
    print("=" * 64)
    print(text)
    print("=" * 64)
    queries = plan.get("search_queries", [])
    print(f"[{evidence_count} 条证据 | {len(queries)} 个检索词 | "
          f"关注：{plan.get('event_focus', 'general')}]", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="[用户] 基于 Graphiti 图谱 + LLM 生成局势推演。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例：
  python -m src.ask "未来30天美伊冲突会升级吗？"
  python -m src.ask "美国制裁影响多大？" --window 60 --limit 10 --json
  python -m src.ask "伊朗打击某军事基地，中国反应如何？" --json
""",
    )
    parser.add_argument("question", nargs="?", help="用户问题")
    parser.add_argument("--window", type=int, default=30, help="检索时间窗口天数（默认 30）")
    parser.add_argument("--limit", type=int, default=10, help="每个 query 最大证据数（默认 10）")
    parser.add_argument("--json", action="store_true", help="输出纯 JSON 格式")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    args = parse_args()

    if not args.question:
        print("用法: python -m src.ask \"你的问题\" [--window 30] [--limit 10] [--json]")
        return

    question = args.question.strip()

    # ── Step 1: LLM 生成检索计划 ────────────────────────────────────────────
    print(f"[Step 1] 生成检索计划...", file=sys.stderr)
    plan = await generate_search_plan(question, args.window)
    queries = plan.get("search_queries", [])
    if not queries:
        print("错误：LLM 未生成有效检索词。", file=sys.stderr)
        sys.exit(1)
    queries = queries[:6]  # 最多 6 个
    print(f"[Step 1] 检索计划 OK — {len(queries)} 个 query：{queries}", file=sys.stderr)

    # ── Step 2: 图谱检索 ────────────────────────────────────────────────────
    now = datetime.now(timezone.utc)
    start_time = (now - timedelta(days=args.window)).strftime("%Y-%m-%dT%H:%M:%SZ")

    all_items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

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
        except Exception as exc:
            print(f"[警告] query \"{q}\" 检索失败：{exc}", file=sys.stderr)

    # ── Step 3: 证据不足则补充一轮 ──────────────────────────────────────────
    evidence_text = format_evidence_context(all_items)
    evidence_summary = summarize_evidence(all_items)

    if not all_items:
        print("当前图谱中没有足够证据支持推演，请先由维护者导入相关事件数据。", file=sys.stderr)
        sys.exit(0)

    print(f"[Step 2] 获取 {len(all_items)} 条证据，评估充分性...", file=sys.stderr)
    sufficiency = await check_evidence_sufficiency(question, evidence_summary)

    # 最多补充一轮
    if sufficiency.get("sufficiency") in ("insufficient", "marginal") and len(all_items) < 5:
        gap = sufficiency.get("gap_note", "")
        print(f"[Step 3] 证据不足，生成补充检索词。缺口：{gap}", file=sys.stderr)
        supplement = await generate_supplementary_queries(question, evidence_summary, gap)
        extra_queries = supplement.get("supplementary_queries", [])[:3]

        for q in extra_queries:
            if q in queries:
                continue
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
            except Exception as exc:
                print(f"[警告] 补充 query \"{q}\" 失败：{exc}", file=sys.stderr)

        if all_items:
            evidence_text = format_evidence_context(all_items)
            evidence_summary = summarize_evidence(all_items)
        print(f"[Step 3] 补充后共 {len(all_items)} 条证据。", file=sys.stderr)

    # ── Step 4: LLM 生成最终推演 ────────────────────────────────────────────
    print(f"[Step 4] 生成最终推演...", file=sys.stderr)
    answer_text = await generate_answer(question, evidence_text, plan)

    # ── 输出 ──────────────────────────────────────────────────────────────────
    if args.json:
        structured = parse_answer_text(answer_text, question)
        structured["evidence_count"] = len(all_items)
        structured["search_queries"] = queries
        structured["question_type"] = plan.get("question_type", "")
        structured["target_actors"] = plan.get("target_actors", [])
        print(json.dumps(structured, ensure_ascii=False, indent=2))
    else:
        print_report(answer_text, question, len(all_items), plan)


if __name__ == "__main__":
    asyncio.run(main())
