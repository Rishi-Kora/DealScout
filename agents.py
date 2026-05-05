"""DealScout agents.

M1: Listing Analyst — extracts a structured FactSheet from a used-car listing
URL or raw text.

Later milestones add: Coordinator (M2), Visual Inspector + Price Auditor (M3),
History Checker (M3+), Critic (M6). They will all live in this file.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from observability import EventType, Scratchpad, TraceLogger
from prompts import (
    COORDINATOR_PROMPT,
    CRITIC_PROMPT,
    HISTORY_CHECKER_PROMPT,
    LISTING_ANALYST_PROMPT,
    PRICE_AUDITOR_PROMPT,
    VISUAL_INSPECTOR_PROMPT,
)
from schemas import (
    CriticReview,
    FactSheet,
    HistoryReport,
    PriceAuditReport,
    Verdict,
    VisualInspectionReport,
)
from tools import (
    HISTORY_CHECKER_TOOL_FUNCTIONS,
    HISTORY_CHECKER_TOOL_SCHEMAS,
    analyze_photos,
    fetch_listing,
    market_price_lookup,
    parse_listing,
)

# Keywords in `claimed_condition` that surface as scratchpad red flags
# from the Listing Analyst. Cheap heuristic — the FactSheet itself preserves
# the full text; this just promotes notable phrases into shared state.
_RED_FLAG_KEYWORDS = (
    "lapsed", "accident", "salvage", "rebuilt", "flood", "stolen", "totaled"
)


def _extract_claimed_red_flags(claimed_condition: str | None) -> list[str]:
    """Cheap keyword scan on claimed_condition. Returns matched keywords."""
    if not claimed_condition:
        return []
    text_lower = claimed_condition.lower()
    return [kw for kw in _RED_FLAG_KEYWORDS if kw in text_lower]

OUTPUT_DIR = Path(__file__).parent / "output"

load_dotenv()

client = OpenAI()

MODEL = "gpt-4o-mini"
COORDINATOR_MAX_STEPS = 10
HISTORY_MAX_STEPS = 7
MAX_REVISIONS = 2  # M6: hard cap on Critic↔Coordinator rounds.


URL_PATTERN = re.compile(r"(mock://\S+|https?://\S+)")


def analyze_listing(
    user_input: str,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
    verbose: bool = True,
) -> FactSheet:
    """Run the Listing Analyst on a listing URL or raw text.

    Design note: the fetch is done deterministically in Python before the
    LLM is called. The LLM does only what it is uniquely good at — turning
    text into a structured FactSheet via structured output. There are no
    tool calls, so no loop, no MAX_STEPS, no opportunity for the model to
    chunk-and-loop. Tool-calling ReAct loops show up properly at M2 in the
    Coordinator, where the agent actually has decisions to make.

    Args:
        user_input: A string containing either a listing URL (e.g.
                    "mock://swift_clean") or raw listing text.
        verbose:    Print live updates while the agent runs.

    Returns:
        A validated FactSheet.

    Raises:
        RuntimeError: If the model returns neither a parsed FactSheet nor
                      a refusal we can surface.
    """
    if tracer is not None:
        tracer.log("listing_analyst", EventType.AGENT_STARTED, {"input": user_input})

    url_match = URL_PATTERN.search(user_input)

    if url_match:
        url = url_match.group(1)
        if verbose:
            print(f"  fetching {url}...")
        fetched = fetch_listing(url)
        if fetched["status"] == "ok":
            text = fetched["text"]
            hints = fetched["structured_hints"]
            error_note = ""
        else:
            text = ""
            hints = {}
            error_note = f"\n\nNote: fetch_listing failed -> {fetched['reason']}"
    else:
        if verbose:
            print("  parsing raw input as listing text...")
        parsed = parse_listing(user_input)
        text = parsed["text"]
        hints = parsed["structured_hints"]
        error_note = ""

    user_msg = (
        f"Extract the FactSheet from this used-car listing.{error_note}\n\n"
        f"=== Listing text ===\n{text}\n\n"
        f"=== Structured hints already extracted ===\n"
        f"{json.dumps(hints, indent=2) if hints else '(none)'}"
    )

    if verbose:
        print(f"  sending to {MODEL}...")

    llm_start = time.time()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": LISTING_ANALYST_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=FactSheet,
    )
    if tracer is not None:
        tracer.log(
            "listing_analyst",
            EventType.LLM_CALL,
            {"duration_ms": int((time.time() - llm_start) * 1000)},
        )

    message = response.choices[0].message

    if message.parsed is None:
        if message.refusal:
            raise RuntimeError(f"Model refused: {message.refusal}")
        raise RuntimeError("Model did not produce a FactSheet.")

    fact_sheet = message.parsed

    # Promote claimed_condition red-flag keywords to the scratchpad so other
    # specialists / the Critic can see them at a glance.
    if scratchpad is not None:
        for flag in _extract_claimed_red_flags(fact_sheet.claimed_condition):
            scratchpad.add("listing_analyst", "claimed_red_flag_keyword", flag)
        if fact_sheet.missing_fields:
            scratchpad.add(
                "listing_analyst",
                "missing_fields",
                fact_sheet.missing_fields,
            )

    if tracer is not None:
        tracer.log(
            "listing_analyst",
            EventType.AGENT_FINISHED,
            {
                "missing_fields_count": len(fact_sheet.missing_fields),
                "make": fact_sheet.make,
                "model": fact_sheet.model,
            },
        )

    if verbose:
        missing = len(fact_sheet.missing_fields)
        print(f"  -> FactSheet produced ({missing} missing fields)")

    return fact_sheet


def inspect_photos(
    photo_urls: list[str],
    fact_sheet_context: dict | None = None,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
    verbose: bool = True,
) -> VisualInspectionReport:
    """Run the Visual Inspector on a listing's photos.

    Same pattern as analyze_listing — call the mock photo-analysis tool first,
    then make a single LLM call to synthesise a VisualInspectionReport.

    Args:
        photo_urls: list of photo URLs from the FactSheet.
        fact_sheet_context: small dict with year, km_driven, claimed_condition
            for cross-checking. None is allowed but reduces inspection quality.

    Returns: a validated VisualInspectionReport.
    """
    if tracer is not None:
        tracer.log(
            "visual_inspector",
            EventType.AGENT_STARTED,
            {"photo_count": len(photo_urls)},
        )

    if verbose:
        print(f"  inspecting {len(photo_urls)} photo(s)...")

    findings = analyze_photos(photo_urls)
    context = fact_sheet_context or {}

    user_msg = (
        "You are inspecting a used car for the DealScout team.\n\n"
        "=== Photo findings (from analyze_photos) ===\n"
        f"{json.dumps(findings, indent=2)}\n\n"
        "=== FactSheet context (for cross-checking) ===\n"
        f"- year: {context.get('year')}\n"
        f"- km_driven: {context.get('km_driven')}\n"
        f"- claimed_condition: {context.get('claimed_condition', '')}\n"
    )

    if verbose:
        print(f"  sending to {MODEL}...")

    llm_start = time.time()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": VISUAL_INSPECTOR_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=VisualInspectionReport,
    )
    if tracer is not None:
        tracer.log(
            "visual_inspector",
            EventType.LLM_CALL,
            {"duration_ms": int((time.time() - llm_start) * 1000)},
        )

    message = response.choices[0].message
    if message.parsed is None:
        if message.refusal:
            raise RuntimeError(f"Visual Inspector refused: {message.refusal}")
        raise RuntimeError("Visual Inspector did not produce a report.")

    report = message.parsed

    if scratchpad is not None:
        for concern in report.concerns:
            scratchpad.add(
                "visual_inspector",
                concern.type.value,
                concern.description,
            )
        scratchpad.add("visual_inspector", "photo_count", report.photo_count)
        scratchpad.add(
            "visual_inspector",
            "overall_condition",
            report.overall_condition.value,
        )

    if tracer is not None:
        tracer.log(
            "visual_inspector",
            EventType.AGENT_FINISHED,
            {
                "concern_count": len(report.concerns),
                "overall_condition": report.overall_condition.value,
            },
        )

    if verbose:
        print(
            f"  -> VisualInspectionReport produced "
            f"({len(report.concerns)} concern(s), "
            f"condition={report.overall_condition.value})"
        )
    return report


def audit_price(
    fact_sheet: dict,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
    verbose: bool = True,
) -> PriceAuditReport:
    """Run the Price Auditor on a FactSheet dict.

    Same pattern: look up the market band deterministically, compute deltas
    in Python (LLMs are bad at arithmetic), then make a single LLM call to
    judge the verdict.
    """
    make = fact_sheet.get("make") or ""
    model = fact_sheet.get("model") or ""
    year = fact_sheet.get("year") or 0
    km_driven = fact_sheet.get("km_driven") or 0
    location = fact_sheet.get("location") or {}
    city = location.get("city") or ""
    asking_price = fact_sheet.get("asking_price") or 0

    if tracer is not None:
        tracer.log(
            "price_auditor",
            EventType.AGENT_STARTED,
            {"year": year, "make": make, "model": model, "asking_price": asking_price},
        )

    if verbose:
        print(f"  auditing price for {year} {make} {model}...")

    market_data = market_price_lookup(make, model, year, km_driven, city)

    p50 = market_data.get("p50") or 0
    if p50 > 0 and asking_price > 0:
        delta_from_median = asking_price - p50
        delta_pct = (asking_price - p50) / p50 * 100.0
    else:
        delta_from_median = 0
        delta_pct = 0.0

    user_msg = (
        "You are auditing the asking price for the DealScout team.\n\n"
        "=== Vehicle details (from FactSheet) ===\n"
        f"- make: {make}\n"
        f"- model: {model}\n"
        f"- year: {year}\n"
        f"- km_driven: {km_driven}\n"
        f"- city: {city}\n"
        f"- asking_price: {asking_price}\n\n"
        "=== Market band (from market_price_lookup) ===\n"
        f"{json.dumps(market_data, indent=2)}\n\n"
        "=== Pre-computed deltas ===\n"
        f"- delta_from_median: {delta_from_median} (asking_price - p50)\n"
        f"- delta_pct: {delta_pct:+.1f}% (relative difference from p50)\n"
    )

    if verbose:
        print(f"  sending to {MODEL}...")

    llm_start = time.time()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": PRICE_AUDITOR_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=PriceAuditReport,
    )
    if tracer is not None:
        tracer.log(
            "price_auditor",
            EventType.LLM_CALL,
            {"duration_ms": int((time.time() - llm_start) * 1000)},
        )

    message = response.choices[0].message
    if message.parsed is None:
        if message.refusal:
            raise RuntimeError(f"Price Auditor refused: {message.refusal}")
        raise RuntimeError("Price Auditor did not produce a report.")

    report = message.parsed

    if scratchpad is not None:
        scratchpad.add("price_auditor", "verdict", report.verdict.value)
        scratchpad.add(
            "price_auditor",
            "delta_pct",
            round(report.delta_pct, 1),
        )

    if tracer is not None:
        tracer.log(
            "price_auditor",
            EventType.AGENT_FINISHED,
            {
                "verdict": report.verdict.value,
                "delta_pct": round(report.delta_pct, 1),
            },
        )

    if verbose:
        print(
            f"  -> PriceAuditReport produced "
            f"(verdict={report.verdict.value}, "
            f"delta_pct={report.delta_pct:+.1f}%)"
        )
    return report


def investigate_history(
    reg_no: str | None,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
    verbose: bool = True,
) -> HistoryReport:
    """Run the History Checker on a vehicle's registration number.

    This is the M4 fractal — same ReAct loop pattern as the Coordinator,
    one level smaller. The LLM decides which tools to call in what order
    based on RC results.

    Defensive patterns from M2's Coordinator are ported in:
    - parallel_tool_calls=False (sequential — RC informs the rest)
    - duplicate-call cache (synthetic 'you already called this' result)
    - HISTORY_MAX_STEPS budget

    Returns a validated HistoryReport. The `tools_called` field is overwritten
    by this loop's actual call history rather than trusting the LLM to track it.
    """
    if reg_no is None:
        reg_no = ""

    if tracer is not None:
        tracer.log("history_checker", EventType.AGENT_STARTED, {"reg_no": reg_no})

    messages: list = [
        {"role": "system", "content": HISTORY_CHECKER_PROMPT},
        {
            "role": "user",
            "content": f"Investigate the registration history for reg_no='{reg_no}'.",
        },
    ]

    tool_call_cache: dict[str, dict] = {}
    tools_called_in_order: list[str] = []
    expected_tools = {
        "rc_lookup", "insurance_status", "challan_check", "accident_lookup_hint",
    }

    for step in range(1, HISTORY_MAX_STEPS + 1):
        if verbose:
            print(f"  [history step {step}] -> {MODEL}...")

        # Once all expected tools have been called (or we're past step 4),
        # strip tools from the next call so the model must emit the report.
        all_tools_called = expected_tools.issubset(set(tools_called_in_order))
        force_emit = all_tools_called or step >= 5

        llm_start = time.time()
        if force_emit:
            if verbose:
                print(f"    (forcing report emission — all tools called or budget reached)")
            response = client.beta.chat.completions.parse(
                model=MODEL,
                messages=messages,
                response_format=HistoryReport,
            )
        else:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                messages=messages,
                tools=HISTORY_CHECKER_TOOL_SCHEMAS,
                response_format=HistoryReport,
                parallel_tool_calls=False,
            )
        if tracer is not None:
            tracer.log(
                "history_checker",
                EventType.LLM_CALL,
                {"step": step, "duration_ms": int((time.time() - llm_start) * 1000)},
            )

        message = response.choices[0].message

        # Branch 1: model wants to call one or more tools.
        if message.tool_calls:
            assistant_msg: dict = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            if message.content:
                assistant_msg["content"] = message.content
            messages.append(assistant_msg)

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                cache_key = f"{name}::{json.dumps(args, sort_keys=True)}"

                if cache_key in tool_call_cache:
                    if verbose:
                        print(f"    -> duplicate call to {name} (cached)")
                    result = {
                        "duplicate_call_detected": True,
                        "instruction": (
                            "You have already called this tool. Move on or "
                            "finalise the HistoryReport now."
                        ),
                        "previous_result": tool_call_cache[cache_key],
                    }
                else:
                    func = HISTORY_CHECKER_TOOL_FUNCTIONS[name]
                    if verbose:
                        print(f"    -> {name}({args})")
                    if tracer is not None:
                        tracer.log(
                            "history_checker",
                            EventType.TOOL_CALLED,
                            {"tool": name, "args": args},
                        )
                    result = func(**args)
                    if tracer is not None:
                        tracer.log(
                            "history_checker",
                            EventType.TOOL_RETURNED,
                            {"tool": name, "status": result.get("status")},
                        )
                    tool_call_cache[cache_key] = result
                    tools_called_in_order.append(name)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result),
                    }
                )
            continue

        # Branch 2: model emitted the final HistoryReport.
        if message.parsed is not None:
            # Override LLM's tools_called with the loop's authoritative list.
            report = message.parsed.model_copy(
                update={"tools_called": tools_called_in_order}
            )

            if scratchpad is not None:
                for rf in report.red_flags:
                    scratchpad.add("history_checker", rf.type.value, rf.description)
                scratchpad.add(
                    "history_checker",
                    "overall_assessment",
                    report.overall_assessment.value,
                )

            if tracer is not None:
                tracer.log(
                    "history_checker",
                    EventType.AGENT_FINISHED,
                    {
                        "red_flag_count": len(report.red_flags),
                        "overall_assessment": report.overall_assessment.value,
                        "tools_called": tools_called_in_order,
                    },
                )

            if verbose:
                print(
                    f"    -> HistoryReport produced "
                    f"({len(report.red_flags)} red flag(s), "
                    f"assessment={report.overall_assessment.value})"
                )
            return report

        if message.refusal:
            raise RuntimeError(f"History Checker refused: {message.refusal}")

        raise RuntimeError(
            "History Checker produced neither tool calls nor a HistoryReport."
        )

    raise RuntimeError(
        f"History Checker exceeded HISTORY_MAX_STEPS ({HISTORY_MAX_STEPS})."
    )


def critique_verdict(
    verdict: Verdict,
    reports: dict[str, dict | None],
    scratchpad: Scratchpad | None = None,
    revision_round: int = 0,
    *,
    tracer: TraceLogger | None = None,
    verbose: bool = True,
) -> CriticReview:
    """Run the Critic on a Coordinator verdict (M6+).

    Single LLM call, no loop. Reads the verdict + all four specialist reports
    + the shared scratchpad and produces a CriticReview that either approves
    the verdict or lists specific issues for revision.

    Args:
        verdict: the Coordinator's current verdict to review.
        reports: dict of specialist reports keyed by name (from coordinate()).
        scratchpad: optional shared state to cross-check against.
        revision_round: which round this is (0 = first review).

    Returns:
        A validated CriticReview.
    """
    if tracer is not None:
        tracer.log(
            "critic",
            EventType.AGENT_STARTED,
            {"revision_round": revision_round},
        )

    if verbose:
        print(f"  reviewing verdict (round {revision_round})...")

    scratchpad_text = (
        scratchpad.to_user_message_section() if scratchpad is not None else "(none)"
    )

    user_msg = (
        "You are reviewing the Coordinator's verdict for quality.\n\n"
        f"=== Revision round ===\n{revision_round}\n\n"
        "=== Verdict (current attempt) ===\n"
        f"{json.dumps(verdict.model_dump(mode='json'), indent=2)}\n\n"
        "=== FactSheet ===\n"
        f"{json.dumps(reports.get('fact_sheet') or {}, indent=2)}\n\n"
        "=== VisualInspectionReport ===\n"
        f"{json.dumps(reports.get('visual_inspection') or {}, indent=2)}\n\n"
        "=== PriceAuditReport ===\n"
        f"{json.dumps(reports.get('price_audit') or {}, indent=2)}\n\n"
        "=== HistoryReport ===\n"
        f"{json.dumps(reports.get('history_report') or {}, indent=2)}\n\n"
        "=== Shared scratchpad ===\n"
        f"{scratchpad_text}\n"
    )

    if verbose:
        print(f"  sending to {MODEL}...")

    llm_start = time.time()
    response = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=CriticReview,
    )
    if tracer is not None:
        tracer.log(
            "critic",
            EventType.LLM_CALL,
            {"duration_ms": int((time.time() - llm_start) * 1000)},
        )

    message = response.choices[0].message
    if message.parsed is None:
        if message.refusal:
            raise RuntimeError(f"Critic refused: {message.refusal}")
        raise RuntimeError("Critic did not produce a CriticReview.")

    review = message.parsed.model_copy(update={"revision_round": revision_round})

    if tracer is not None:
        tracer.log(
            "critic",
            EventType.AGENT_FINISHED,
            {
                "approved": review.approved,
                "issue_count": len(review.issues),
                "revision_round": revision_round,
            },
        )

    if verbose:
        status = "APPROVED" if review.approved else f"REVISE ({len(review.issues)} issues)"
        print(f"  -> CriticReview: {status}")

    return review


def consult_listing_analyst(
    user_input: str,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
) -> dict:
    """Agent-as-tool wrapper for the Listing Analyst.

    From the Coordinator's perspective this is just a function that takes a
    string and returns a dict. It does not know (and should not know) that
    internally another LLM with its own loop is doing the work.

    The `scratchpad` and `tracer` keyword args are passed through transparently
    so the wrapped agent can record events/findings under its own identity.
    """
    fact_sheet = analyze_listing(
        user_input, scratchpad=scratchpad, tracer=tracer, verbose=False
    )
    return fact_sheet.model_dump(mode="json")


def consult_visual_inspector(
    photo_urls: list[str],
    fact_sheet_context: dict,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
) -> dict:
    """Agent-as-tool wrapper for the Visual Inspector (M3+)."""
    report = inspect_photos(
        photo_urls,
        fact_sheet_context=fact_sheet_context,
        scratchpad=scratchpad,
        tracer=tracer,
        verbose=False,
    )
    return report.model_dump(mode="json")


def consult_price_auditor(
    fact_sheet: dict,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
) -> dict:
    """Agent-as-tool wrapper for the Price Auditor (M3+)."""
    report = audit_price(
        fact_sheet, scratchpad=scratchpad, tracer=tracer, verbose=False
    )
    return report.model_dump(mode="json")


def consult_history_checker(
    reg_no: str | None,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
) -> dict:
    """Agent-as-tool wrapper for the History Checker (M4+).

    Wraps an internal ReAct loop with its own four tools — the fractal.
    Returns the HistoryReport as a dict for the Coordinator to read.
    """
    report = investigate_history(
        reg_no, scratchpad=scratchpad, tracer=tracer, verbose=False
    )
    return report.model_dump(mode="json")


# OpenAI-format tool schemas that the Coordinator's LLM sees.
# Each entry wraps one specialist agent. M4 will add the History Checker.
COORDINATOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "consult_listing_analyst",
            "description": (
                "Send a used-car listing URL or raw listing text to the Listing "
                "Analyst specialist. Returns a structured FactSheet with extracted "
                "fields (make, model, year, km_driven, owner_count, asking_price, "
                "location, reg_no, photos, claimed_condition, missing_fields, "
                "extraction_confidence). Call this FIRST and ALONE — the other "
                "specialists need its output."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": (
                            "The listing URL (e.g. 'mock://swift_clean') or raw "
                            "listing text the user provided. Pass through verbatim."
                        ),
                    }
                },
                "required": ["user_input"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_visual_inspector",
            "description": (
                "Send the listing's photos to the Visual Inspector specialist. "
                "Returns a VisualInspectionReport (concerns, missing angles, "
                "overall_condition, photo_count, notes). Call this AFTER the "
                "Listing Analyst returns. Can run in parallel with the Price Auditor."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "photo_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of photo URLs from the FactSheet's `photos` field. "
                            "Pass an empty list if none are available."
                        ),
                    },
                    "fact_sheet_context": {
                        "type": "object",
                        "description": (
                            "Cross-check context from the FactSheet: year, km_driven, "
                            "claimed_condition. Used by the Visual Inspector to detect "
                            "contradictions like odometer suspicion."
                        ),
                        "properties": {
                            "year": {"type": ["integer", "null"]},
                            "km_driven": {"type": ["integer", "null"]},
                            "claimed_condition": {"type": ["string", "null"]},
                        },
                        "required": ["year", "km_driven", "claimed_condition"],
                        "additionalProperties": False,
                    },
                },
                "required": ["photo_urls", "fact_sheet_context"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_price_auditor",
            "description": (
                "Send the FactSheet to the Price Auditor specialist. Returns a "
                "PriceAuditReport with market_band, delta_from_median, delta_pct, "
                "verdict (fair / overpriced / underpriced / suspicious_low), and "
                "notes. Call this AFTER the Listing Analyst returns. Can run in "
                "parallel with the Visual Inspector and History Checker."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "fact_sheet": {
                        "type": "object",
                        "description": (
                            "The FactSheet dict returned by the Listing Analyst. "
                            "Pass through verbatim — the Price Auditor will pull "
                            "make, model, year, km_driven, location, asking_price."
                        ),
                        "properties": {
                            "make": {"type": ["string", "null"]},
                            "model": {"type": ["string", "null"]},
                            "year": {"type": ["integer", "null"]},
                            "km_driven": {"type": ["integer", "null"]},
                            "location": {
                                "type": ["object", "null"],
                                "properties": {
                                    "city": {"type": "string"},
                                    "state": {"type": "string"},
                                },
                                "required": ["city", "state"],
                                "additionalProperties": False,
                            },
                            "asking_price": {"type": ["integer", "null"]},
                        },
                        "required": [
                            "make", "model", "year", "km_driven",
                            "location", "asking_price",
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["fact_sheet"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_history_checker",
            "description": (
                "Send the registration number to the History Checker specialist. "
                "Returns a HistoryReport with rc_data (owner_count, year, "
                "hypothecation, blacklist), insurance_record (insurer, lapsed), "
                "challans, iib_recommendation, red_flags, overall_assessment "
                "(clean / minor_concerns / major_concerns / insufficient_data). "
                "Call this AFTER the Listing Analyst returns. Can run in parallel "
                "with the Visual Inspector and Price Auditor."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reg_no": {
                        "type": ["string", "null"],
                        "description": (
                            "The vehicle registration number from the FactSheet's "
                            "`reg_no` field. Pass null if the FactSheet did not "
                            "include one — the History Checker will return an "
                            "insufficient_data report."
                        ),
                    },
                },
                "required": ["reg_no"],
                "additionalProperties": False,
            },
        },
    },
]

# Dispatch table — Coordinator's loop looks up specialist calls by name.
COORDINATOR_TOOL_FUNCTIONS = {
    "consult_listing_analyst": consult_listing_analyst,
    "consult_visual_inspector": consult_visual_inspector,
    "consult_price_auditor": consult_price_auditor,
    "consult_history_checker": consult_history_checker,
}


def coordinate(
    user_input: str,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
    verbose: bool = True,
) -> tuple[Verdict, dict[str, dict | None]]:
    """Run the Coordinator on a listing URL or raw text.

    The Coordinator is a real ReAct loop: it decides which specialists to
    consult, reads the results, then produces a structured Verdict. At M2 it
    had one specialist; at M3 it has three (Listing Analyst, Visual Inspector,
    Price Auditor). M4 adds the History Checker.

    Returns:
        (Verdict, reports_dict) where reports_dict has keys
        "fact_sheet", "visual_inspection", "price_audit", each holding the
        specialist's output as a dict (or None if the specialist was not
        consulted on this run).

    Raises:
        RuntimeError: if the Coordinator exceeds COORDINATOR_MAX_STEPS or
                      produces neither a tool call nor a Verdict.
    """
    if tracer is not None:
        tracer.log("coordinator", EventType.AGENT_STARTED, {"input": user_input})

    messages: list = [
        {"role": "system", "content": COORDINATOR_PROMPT},
        {"role": "user", "content": user_input},
    ]

    # Captured outputs from each specialist so the caller can save them.
    reports: dict[str, dict | None] = {
        "fact_sheet": None,
        "visual_inspection": None,
        "price_audit": None,
        "history_report": None,
    }
    # Map tool name -> reports key.
    tool_to_report = {
        "consult_listing_analyst": "fact_sheet",
        "consult_visual_inspector": "visual_inspection",
        "consult_price_auditor": "price_audit",
        "consult_history_checker": "history_report",
    }

    # Defensive: if the model calls the same tool with the same args twice,
    # return cached result + an instruction to produce the verdict now.
    tool_call_cache: dict[str, dict] = {}

    required_specialists = [
        "fact_sheet", "visual_inspection", "price_audit", "history_report"
    ]

    for step in range(1, COORDINATOR_MAX_STEPS + 1):
        if verbose:
            print(f"[coordinator step {step}] -> {MODEL}...")

        # Once all specialists are consulted, strip tools from the request.
        # The model can no longer call anything — it must emit the Verdict.
        all_specialists_done = all(
            reports.get(k) is not None for k in required_specialists
        )

        llm_start = time.time()
        if all_specialists_done:
            if verbose:
                print("  (all specialists consulted — forcing verdict emission)")
            response = client.beta.chat.completions.parse(
                model=MODEL,
                messages=messages,
                response_format=Verdict,
            )
        else:
            response = client.beta.chat.completions.parse(
                model=MODEL,
                messages=messages,
                tools=COORDINATOR_TOOL_SCHEMAS,
                response_format=Verdict,
                parallel_tool_calls=True,
            )
        if tracer is not None:
            tracer.log(
                "coordinator",
                EventType.LLM_CALL,
                {"step": step, "duration_ms": int((time.time() - llm_start) * 1000)},
            )

        message = response.choices[0].message

        # Branch 1: Coordinator wants to consult one or more specialists.
        if message.tool_calls:
            assistant_msg: dict = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            if message.content:
                assistant_msg["content"] = message.content
            messages.append(assistant_msg)

            # Plan each tool call: cached -> synthetic result; otherwise queue for run.
            plans: list[dict] = []
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                cache_key = f"{name}::{json.dumps(args, sort_keys=True)}"

                if cache_key in tool_call_cache:
                    plans.append({
                        "id": tool_call.id, "name": name, "args": args,
                        "cache_key": cache_key, "needs_run": False,
                        "result": {
                            "duplicate_call_detected": True,
                            "instruction": (
                                "You have already received this information. "
                                "Produce the final Verdict now without further tool calls."
                            ),
                            "previous_result": tool_call_cache[cache_key],
                        },
                    })
                else:
                    plans.append({
                        "id": tool_call.id, "name": name, "args": args,
                        "cache_key": cache_key, "needs_run": True, "result": None,
                    })

            # Run all uncached plans in parallel via a thread pool.
            to_run = [p for p in plans if p["needs_run"]]
            if to_run:
                if verbose and len(to_run) > 1:
                    print(f"  -> running {len(to_run)} specialists in parallel: "
                          f"{[p['name'] for p in to_run]}")
                elif verbose:
                    print(f"  -> consulting: {to_run[0]['name']}")

                start = time.time()
                # Log the dispatch decision (one TOOL_CALLED per specialist)
                if tracer is not None:
                    for p in to_run:
                        tracer.log(
                            "coordinator",
                            EventType.TOOL_CALLED,
                            {"tool": p["name"], "args_keys": list(p["args"].keys())},
                        )

                with ThreadPoolExecutor(max_workers=len(to_run)) as executor:
                    futures = [
                        (p, executor.submit(
                            COORDINATOR_TOOL_FUNCTIONS[p["name"]],
                            **p["args"],
                            scratchpad=scratchpad,
                            tracer=tracer,
                        ))
                        for p in to_run
                    ]
                    for plan, future in futures:
                        plan["result"] = future.result()
                        tool_call_cache[plan["cache_key"]] = plan["result"]
                        report_key = tool_to_report.get(plan["name"])
                        if report_key:
                            reports[report_key] = plan["result"]
                        if tracer is not None:
                            tracer.log(
                                "coordinator",
                                EventType.TOOL_RETURNED,
                                {"tool": plan["name"]},
                            )

                if verbose:
                    print(f"  -> {len(to_run)} specialist(s) finished in "
                          f"{time.time() - start:.2f}s")

            # Append all tool messages in the original tool_calls order.
            for plan in plans:
                if plan["needs_run"] is False and verbose:
                    print(f"  -> duplicate call to {plan['name']} (cached)")
                messages.append({
                    "role": "tool",
                    "tool_call_id": plan["id"],
                    "content": json.dumps(plan["result"]),
                })
            continue

        # Branch 2: Coordinator emitted the final Verdict.
        if message.parsed is not None:
            verdict = message.parsed
            if tracer is not None:
                tracer.log(
                    "coordinator",
                    EventType.AGENT_FINISHED,
                    {
                        "recommendation": verdict.recommendation.value,
                        "confidence": verdict.confidence.value,
                        "disagreement_count": len(verdict.disagreements),
                    },
                )
            if verbose:
                print(
                    f"  -> Verdict: {verdict.recommendation.value} "
                    f"({verdict.confidence.value} confidence)"
                )
            return verdict, reports

        if message.refusal:
            raise RuntimeError(f"Coordinator refused: {message.refusal}")

        raise RuntimeError("Coordinator produced neither tool calls nor a Verdict.")

    raise RuntimeError(
        f"Coordinator exceeded COORDINATOR_MAX_STEPS ({COORDINATOR_MAX_STEPS}). "
        f"Tool calls cached: {list(tool_call_cache.keys())}"
    )


def _format_revision_feedback(review: CriticReview) -> str:
    """Render a CriticReview's issues as a prompt-friendly block to feed into
    the Coordinator's next attempt."""
    if review.approved or not review.issues:
        return ""
    lines = [
        "",
        "PREVIOUS ATTEMPT WAS REVISED. Address the Critic's feedback:",
    ]
    for issue in review.issues:
        lines.append(
            f"- [{issue.type.value} / {issue.severity.value}] "
            f"{issue.description}"
        )
        lines.append(f"  Suggested fix: {issue.suggested_fix}")
    return "\n".join(lines)


def orchestrate(
    user_input: str,
    *,
    scratchpad: Scratchpad | None = None,
    tracer: TraceLogger | None = None,
    verbose: bool = True,
) -> tuple[Verdict, dict[str, dict | None], list[CriticReview]]:
    """Run the full DealScout pipeline with Critic revision loop (M6+).

    1. Coordinator produces a verdict + reports.
    2. Critic reviews.
    3. If approved, ship.
    4. Otherwise, Coordinator re-runs with the Critic's feedback prepended.
    5. Hard cap at MAX_REVISIONS rounds — ship whatever's current after that
       even if the Critic still has issues (transparent failure).

    Returns:
        (final_verdict, final_reports, list_of_all_reviews_in_order)
    """
    reviews: list[CriticReview] = []
    augmented_input = user_input
    verdict: Verdict | None = None
    reports: dict[str, dict | None] = {}

    for revision_round in range(MAX_REVISIONS + 1):
        if verbose:
            print(f"\n  --- attempt {revision_round} ---")

        verdict, reports = coordinate(
            augmented_input, scratchpad=scratchpad, tracer=tracer, verbose=verbose
        )

        review = critique_verdict(
            verdict,
            reports,
            scratchpad=scratchpad,
            revision_round=revision_round,
            tracer=tracer,
            verbose=verbose,
        )
        reviews.append(review)

        if review.approved:
            if verbose:
                print(f"  Critic approved on round {revision_round}.")
            break

        if revision_round == MAX_REVISIONS:
            if verbose:
                print(
                    f"  Hit MAX_REVISIONS ({MAX_REVISIONS}). "
                    f"Shipping current verdict with {len(review.issues)} unresolved issue(s)."
                )
            break

        # Stuck-loop detection: if the new revisions are the same as last
        # round's, treat as stuck and ship.
        if revision_round > 0:
            prev_issues = {
                (i.type.value, i.description) for i in reviews[-2].issues
            }
            curr_issues = {(i.type.value, i.description) for i in review.issues}
            if prev_issues == curr_issues:
                if verbose:
                    print("  Critic feedback unchanged from last round — treating as stuck. Shipping.")
                break

        # Prepend revision feedback for the next attempt.
        feedback_block = _format_revision_feedback(review)
        augmented_input = user_input + feedback_block

    return verdict, reports, reviews


def archive_existing_run(slug: str) -> Path | None:
    """If output/<slug>/ has top-level files from a previous run, move them
    into output/<slug>/_history/<YYYY-MM-DD_HH-MM-SS>/ before the new run writes.

    Returns the archive path (or None if there was nothing to archive).
    """
    folder = OUTPUT_DIR / slug
    if not folder.exists():
        return None

    existing = [f for f in folder.iterdir() if f.is_file()]
    if not existing:
        return None

    archive_dir = folder / "_history" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for f in existing:
        f.rename(archive_dir / f.name)
    return archive_dir


def save_run_output(fact_sheet, slug: str) -> Path:
    """Write a FactSheet (object OR dict) to output/<slug>/fact_sheet.json.

    Accepts either form because M1 produces a FactSheet object directly while
    M2's Coordinator captures the analyst's output as a dict during a tool call.
    """
    folder = OUTPUT_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)

    data = (
        fact_sheet.model_dump(mode="json")
        if isinstance(fact_sheet, FactSheet)
        else fact_sheet
    )
    (folder / "fact_sheet.json").write_text(json.dumps(data, indent=2))
    return folder


def render_verdict_markdown(verdict: Verdict, fact_sheet: dict | None = None) -> str:
    """Render a Verdict as the human-readable verdict.md."""
    lines: list[str] = []
    lines.append("# DealScout Verdict")
    lines.append("")
    lines.append(f"**Recommendation:** `{verdict.recommendation.value.upper()}`  ")
    lines.append(f"**Confidence:** {verdict.confidence.value}")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(verdict.headline)
    lines.append("")
    lines.append("## Reasoning")
    lines.append("")
    lines.append(verdict.reasoning)
    lines.append("")

    if verdict.key_concerns:
        lines.append("## Key concerns")
        lines.append("")
        for concern in verdict.key_concerns:
            lines.append(f"- {concern}")
        lines.append("")

    if verdict.disagreements:
        lines.append("## Disagreements between specialists")
        lines.append("")
        for d in verdict.disagreements:
            lines.append(f"- **Between** {', '.join(d.between)} — {d.on}  ")
            lines.append(f"  *Resolution:* {d.resolution}")
        lines.append("")

    lines.append("## Specialists consulted")
    lines.append("")
    for spec in verdict.specialists_consulted:
        lines.append(f"- {spec}")
    lines.append("")

    if fact_sheet:
        lines.append("## Fact sheet summary")
        lines.append("")
        car = " ".join(
            str(p) for p in [
                fact_sheet.get("year"),
                fact_sheet.get("make"),
                fact_sheet.get("model"),
                fact_sheet.get("variant") or "",
            ] if p
        )
        lines.append(f"- {car}")
        if isinstance(fact_sheet.get("km_driven"), int):
            lines.append(
                f"- {fact_sheet['km_driven']:,} km, "
                f"owner #{fact_sheet.get('owner_count', '?')}"
            )
        if isinstance(fact_sheet.get("asking_price"), int):
            lines.append(f"- Asking price: ₹{fact_sheet['asking_price']:,}")
        loc = fact_sheet.get("location") or {}
        if loc.get("city"):
            lines.append(f"- Location: {loc['city']}, {loc.get('state', '')}")
        if fact_sheet.get("reg_no"):
            lines.append(f"- Reg: {fact_sheet['reg_no']}")
        if fact_sheet.get("missing_fields"):
            lines.append(
                f"- Missing fields: {', '.join(fact_sheet['missing_fields'])}"
            )

    return "\n".join(lines) + "\n"


def save_verdict(verdict: Verdict, fact_sheet_dict: dict | None, slug: str) -> Path:
    """Write verdict.json and verdict.md into output/<slug>/."""
    folder = OUTPUT_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "verdict.json").write_text(
        json.dumps(verdict.model_dump(mode="json"), indent=2)
    )
    (folder / "verdict.md").write_text(
        render_verdict_markdown(verdict, fact_sheet_dict)
    )
    return folder


def save_all_reports(reports: dict[str, dict | None], slug: str) -> Path:
    """Write each specialist's report to its own JSON file under output/<slug>/."""
    folder = OUTPUT_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)

    file_map = {
        "fact_sheet": "fact_sheet.json",
        "visual_inspection": "visual_inspection.json",
        "price_audit": "price_audit.json",
        "history_report": "history_report.json",
    }

    for key, filename in file_map.items():
        if reports.get(key) is not None:
            (folder / filename).write_text(json.dumps(reports[key], indent=2))

    return folder


if __name__ == "__main__":
    # M2: Run the Coordinator on every fixture. Save fact_sheet.json,
    # verdict.json and verdict.md into the run folder.
    fixture_urls = [
        "mock://swift_clean",
        "mock://honda_sparse",
        "mock://i20_noisy",
        "mock://innova_dealer",
    ]

    for url in fixture_urls:
        slug = url.replace("mock://", "")

        print("\n" + "=" * 70)
        print(f"COORDINATING: {url}")
        print("=" * 70)

        # Archive prior run BEFORE we start writing this one.
        archived = archive_existing_run(slug)
        if archived:
            print(f"  archived previous run -> {archived.relative_to(OUTPUT_DIR.parent)}")

        # Per-run observability: scratchpad + trace.jsonl in the listing folder.
        listing_folder = OUTPUT_DIR / slug
        listing_folder.mkdir(parents=True, exist_ok=True)
        scratchpad = Scratchpad()
        with TraceLogger(listing_folder / "trace.jsonl") as tracer:
            verdict, reports, reviews = orchestrate(
                f"Analyse this listing: {url}",
                scratchpad=scratchpad,
                tracer=tracer,
            )

        save_all_reports(reports, slug)
        folder = save_verdict(verdict, reports.get("fact_sheet"), slug)

        # Snapshot scratchpad to disk.
        (folder / "scratchpad.json").write_text(
            json.dumps(scratchpad.to_dict(), indent=2)
        )

        # M6: persist the full revision history of Critic reviews.
        (folder / "critic_notes.json").write_text(
            json.dumps(
                [r.model_dump(mode="json") for r in reviews],
                indent=2,
            )
        )

        final_review = reviews[-1]
        critic_status = (
            "approved"
            if final_review.approved
            else f"shipped with {len(final_review.issues)} unresolved issue(s)"
        )

        print(f"  saved -> {folder.relative_to(OUTPUT_DIR.parent)}")
        print(f"  verdict:  {verdict.recommendation.value.upper()} "
              f"({verdict.confidence.value} confidence)")
        print(f"  headline: {verdict.headline}")
        print(f"  critic:   {critic_status} after {len(reviews)} round(s)")
        print(f"  trace events: {len(tracer)}, scratchpad entries: {len(scratchpad)}")

    print("\n" + "=" * 70)
    print("M3 complete. Inspect verdict.md, fact_sheet.json, "
          "visual_inspection.json, price_audit.json under output/<run>/.")
    print("=" * 70)
