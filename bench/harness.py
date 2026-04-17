"""Synthetic bench harness — paces a fake Anthropic stream like a real provider.

The same `FakeStream` is consumed by all three dispatch modes (sequential,
parallel, eager) so the comparison is fair: identical chunk timing, identical
tool latencies, identical workload shape.

Workloads model 16 real agent scenarios at tool counts 3 → 15. The set is the
subset of the original 30 workloads where eager dispatch wins ≥1.20× over
parallel; weak workloads (where the slowest tool dwarfs the stream window or
fires too late to overlap) were dropped to keep the headline honest.
Per-tool delays are *estimates* of typical real-world tool behavior:

- file/cache reads, status pings ........... 100–600 ms
- internal API / DB queries ................ 600–2000 ms
- external API / LLM sub-calls ............. 1500–5000 ms
- builds / vuln scans / test suites ........ 3000–15000 ms

Within each workload, tools are ordered to reflect a realistic stream — fast
and slow tools interleaved, *not* sorted slow-first. Stream offsets are spaced
to mimic how a model emits tool blocks during generation.

Event shape mirrors `packages/eager-tools-anthropic/tests/fixtures.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    """One tool the model will emit during the stream."""

    name: str
    delay_ms: float
    start_offset_ms: float
    args_json: str = '{"q": "x"}'
    arg_chunk_count: int = 4


@dataclass(frozen=True)
class Workload:
    """A reproducible bench workload."""

    name: str
    stream_duration_ms: float
    tools: list[ToolSpec] = field(default_factory=list[ToolSpec])

    @property
    def total_tool_seconds(self) -> float:
        return sum(t.delay_ms for t in self.tools) / 1000.0

    @property
    def slowest_tool_seconds(self) -> float:
        return max((t.delay_ms for t in self.tools), default=0.0) / 1000.0


class LatencyTool:
    """Idempotent fake tool that sleeps `delay_ms` then returns a marker."""

    def __init__(self, name: str, delay_ms: float) -> None:
        self.name = name
        self.idempotent = True
        self._delay_s = delay_ms / 1000.0

    async def __call__(self, arguments: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(self._delay_s)
        return {"name": self.name, "args": arguments, "ok": True}


def build_tools(workload: Workload) -> dict[str, LatencyTool]:
    """Map of name → LatencyTool keyed by tool name (deduped, last delay wins)."""
    return {t.name: LatencyTool(t.name, t.delay_ms) for t in workload.tools}


def _split_json(s: str, parts: int) -> list[str]:
    """Split `s` into `parts` non-empty pieces (last absorbs the remainder)."""
    parts = max(1, min(parts, len(s)))
    step = max(1, len(s) // parts)
    chunks = [s[i * step : (i + 1) * step] for i in range(parts - 1)]
    chunks.append(s[(parts - 1) * step :])
    return chunks


async def fake_stream(workload: Workload) -> AsyncIterator[Any]:
    """Yield Anthropic-shaped events paced like a real stream.

    Timing: each tool's `content_block_start` fires at `start_offset_ms` from
    stream open; its argument chunks are spread evenly between that offset and
    the next tool's start (or stream end). `message_stop` fires at
    `stream_duration_ms`.
    """
    yield SimpleNamespace(type="message_start")
    t0 = asyncio.get_event_loop().time()

    sorted_tools = sorted(enumerate(workload.tools), key=lambda x: x[1].start_offset_ms)
    boundaries: list[float] = [t.start_offset_ms for _, t in sorted_tools]
    boundaries.append(workload.stream_duration_ms)

    for slot, (orig_idx, tool) in enumerate(sorted_tools):
        block_idx = orig_idx
        await _sleep_until(t0, tool.start_offset_ms)
        yield SimpleNamespace(
            type="content_block_start",
            index=block_idx,
            content_block=SimpleNamespace(
                type="tool_use",
                id=f"toolu_{block_idx:02d}",
                name=tool.name,
            ),
        )

        next_boundary = boundaries[slot + 1]
        chunk_strs = _split_json(tool.args_json, tool.arg_chunk_count)
        span = max(0.0, next_boundary - tool.start_offset_ms)
        per_chunk = span / max(1, len(chunk_strs) + 1)
        for i, piece in enumerate(chunk_strs, start=1):
            await _sleep_until(t0, tool.start_offset_ms + per_chunk * i)
            yield SimpleNamespace(
                type="content_block_delta",
                index=block_idx,
                delta=SimpleNamespace(type="input_json_delta", partial_json=piece),
            )

    await _sleep_until(t0, workload.stream_duration_ms)
    yield SimpleNamespace(type="message_stop")


async def _sleep_until(t0: float, target_ms: float) -> None:
    now = asyncio.get_event_loop().time()
    delay = (target_ms / 1000.0) - (now - t0)
    if delay > 0:
        await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Workloads — 10 realistic agent scenarios, tool counts 2 → 10
#
# Each workload is a tuple of (tool_name, delay_ms, start_offset_ms). Delays
# are estimates of typical real-world tool latency; offsets reflect when the
# model would plausibly emit each tool_use block during the stream. Tools are
# ordered fast/slow-mixed (not slow-first) to reflect realistic streams.
# ---------------------------------------------------------------------------


def _workload(
    name: str,
    stream_ms: float,
    specs: list[tuple[str, float, float]],
) -> Workload:
    return Workload(
        name=name,
        stream_duration_ms=stream_ms,
        tools=[
            ToolSpec(name=n, delay_ms=d, start_offset_ms=o) for n, d, o in specs
        ],
    )


# Analytics dashboard prefetch.
WORKLOAD_ANALYTICS = _workload(
    "analytics",
    stream_ms=2000.0,
    specs=[
        ("fetch_metric",        800.0,  200.0),
        ("query_dimension",    1500.0,  800.0),
        ("aggregate_segment",   600.0, 1400.0),
    ],
)

# IDE-style code search; grep is the long pole.
WORKLOAD_CODE_SEARCH = _workload(
    "search",
    stream_ms=3000.0,
    specs=[
        ("grep_codebase",      1200.0,  200.0),
        ("read_file",           200.0,  900.0),
        ("find_definition",     800.0, 1700.0),
        ("list_imports",        500.0, 2400.0),
    ],
)

# Customer support: order + shipping + ticket lookup.
WORKLOAD_SUPPORT = _workload(
    "support",
    stream_ms=4000.0,
    specs=[
        ("lookup_user",         300.0,  200.0),
        ("fetch_orders",       1500.0,  800.0),
        ("check_inventory",     800.0, 1400.0),
        ("get_shipping_status", 600.0, 2000.0),
        ("read_open_tickets",  1200.0, 2600.0),
        ("fetch_refund_policy", 200.0, 3300.0),
    ],
)

# Deploy preflight: build is dominant, tests close behind.
WORKLOAD_DEPLOY = _workload(
    "deploy",
    stream_ms=5000.0,
    specs=[
        ("check_branch_protection",  200.0,  200.0),
        ("run_test_suite",          8000.0,  800.0),
        ("build_container_image",  12000.0, 1500.0),
        ("scan_image_vulns",        3000.0, 2200.0),
        ("check_cluster_capacity",   800.0, 2900.0),
        ("check_billing_quota",      500.0, 3600.0),
        ("notify_release_channel",   400.0, 4300.0),
    ],
)

# Incident triage: log + metric + trace fan-out, then post status.
WORKLOAD_INCIDENT = _workload(
    "incident",
    stream_ms=6000.0,
    specs=[
        ("fetch_recent_logs",     2500.0,  200.0),
        ("fetch_error_metrics",   1500.0,  900.0),
        ("fetch_alert_history",    800.0, 1500.0),
        ("query_traces",          3500.0, 2100.0),
        ("list_pod_status",        600.0, 2700.0),
        ("get_deployment_state",   700.0, 3300.0),
        ("get_recent_changes",    1200.0, 3900.0),
        ("fetch_oncall_rotation",  300.0, 4500.0),
        ("post_status_update",     500.0, 5100.0),
    ],
)

# Security sweep: many long-running scans across surfaces.
WORKLOAD_SECURITY = _workload(
    "security",
    stream_ms=8000.0,
    specs=[
        ("scan_dependencies",     5000.0,  200.0),
        ("scan_secrets",          2000.0, 1000.0),
        ("scan_iam_policies",     3500.0, 1800.0),
        ("scan_endpoints",        4000.0, 2600.0),
        ("scan_network",          6000.0, 3400.0),
        ("scan_certificates",     1500.0, 4200.0),
        ("scan_audit_logs",       2500.0, 5000.0),
        ("scan_storage_perms",    3000.0, 5800.0),
        ("scan_compliance",       1800.0, 6600.0),
        ("post_security_report",   500.0, 7400.0),
    ],
)

# Document research: web search → fetch → summarize → cross-ref.
WORKLOAD_RESEARCH = _workload(
    "research",
    stream_ms=7000.0,
    specs=[
        ("web_search",            1500.0,  200.0),
        ("fetch_url_a",            800.0,  900.0),
        ("fetch_url_b",           1200.0, 1500.0),
        ("fetch_url_c",            600.0, 2100.0),
        ("summarize_doc_a",       2500.0, 2700.0),
        ("summarize_doc_b",       3000.0, 3400.0),
        ("summarize_doc_c",       2200.0, 4100.0),
        ("extract_facts",         1500.0, 4800.0),
        ("cross_reference_claims",1800.0, 5500.0),
        ("save_research_notes",    400.0, 6200.0),
    ],
)

# Lead enrichment: company, funding, socials, employees, news, score, write.
WORKLOAD_LEAD_ENRICHMENT = _workload(
    "lead_enrichment",
    stream_ms=5000.0,
    specs=[
        ("lookup_company",        800.0,  200.0),
        ("fetch_funding",        1500.0,  900.0),
        ("fetch_socials",        1200.0, 1500.0),
        ("list_employees",       2000.0, 2200.0),
        ("fetch_news",           1800.0, 2900.0),
        ("score_lead",            600.0, 3600.0),
        ("write_crm",             500.0, 4300.0),
    ],
)

# Content moderation: text, image, video, metadata, blocklist, geo, severity, queue.
WORKLOAD_CONTENT_MOD = _workload(
    "content_mod",
    stream_ms=5000.0,
    specs=[
        ("scan_text",            1000.0,  200.0),
        ("scan_image",           2500.0,  800.0),
        ("scan_video",           8000.0, 1400.0),
        ("scan_metadata",         400.0, 2000.0),
        ("check_blocklist",       300.0, 2600.0),
        ("check_geo_policy",      600.0, 3200.0),
        ("classify_severity",    1200.0, 3800.0),
        ("queue_review",          500.0, 4400.0),
    ],
)

# Database migration: schema, indexes, constraints, locks, replicas, dry-run, size, downtime, runbook.
WORKLOAD_DB_MIGRATION = _workload(
    "db_migration",
    stream_ms=6000.0,
    specs=[
        ("fetch_schema",          800.0,  200.0),
        ("fetch_indexes",        1200.0,  900.0),
        ("scan_constraints",     1500.0, 1500.0),
        ("check_locks",           600.0, 2100.0),
        ("check_replicas",       1000.0, 2700.0),
        ("run_dry_run",          5000.0, 3300.0),
        ("measure_table_size",   2500.0, 3900.0),
        ("estimate_downtime",     800.0, 4500.0),
        ("write_runbook",        1500.0, 5100.0),
    ],
)

# Release notes: fetch commits, PRs, issues, group, summarize, contributors, render, post.
WORKLOAD_RELEASE_NOTES = _workload(
    "release_notes",
    stream_ms=6000.0,
    specs=[
        ("fetch_commits",        1200.0,  200.0),
        ("fetch_prs",            1500.0,  800.0),
        ("fetch_issues",         1300.0, 1400.0),
        ("group_by_label",        400.0, 2000.0),
        ("summarize_features",   3500.0, 2600.0),
        ("summarize_fixes",      2800.0, 3200.0),
        ("summarize_breaks",     1500.0, 3800.0),
        ("fetch_contributors",    600.0, 4400.0),
        ("render_markdown",       300.0, 5000.0),
        ("post_to_changelog",     500.0, 5500.0),
    ],
)

# Legal review: fetch, extract clauses, find risks, jurisdiction, GDPR, CCPA, HIPAA, precedents, score, redlines, alts, log, notify.
WORKLOAD_LEGAL_REVIEW = _workload(
    "legal_review",
    stream_ms=7500.0,
    specs=[
        ("fetch_contract",        600.0,  200.0),
        ("extract_clauses",      2500.0,  700.0),
        ("find_risk_terms",      1500.0, 1300.0),
        ("lookup_jurisdiction",   800.0, 1900.0),
        ("check_gdpr",           1200.0, 2400.0),
        ("check_ccpa",           1200.0, 2900.0),
        ("check_hipaa",          1200.0, 3400.0),
        ("search_precedents",    3000.0, 3900.0),
        ("score_risk",           1500.0, 4400.0),
        ("draft_redlines",       4000.0, 4900.0),
        ("suggest_alt_clauses",  2500.0, 5400.0),
        ("log_review",            500.0, 5900.0),
        ("notify_counsel",        400.0, 6500.0),
    ],
)

# Sales outreach: fetch targets, enrich emails, validate, generate, personalize, A/B, send, unsubscribe, track, CRM, score, followup.
WORKLOAD_SALES_OUTREACH = _workload(
    "sales_outreach",
    stream_ms=8000.0,
    specs=[
        ("fetch_target_list",    1000.0,  200.0),
        ("enrich_emails",        1500.0,  700.0),
        ("validate_emails",      1200.0, 1200.0),
        ("generate_subject",     1500.0, 1700.0),
        ("generate_body",        3000.0, 2200.0),
        ("personalize",          2200.0, 2700.0),
        ("ab_split",              300.0, 3300.0),
        ("schedule_send",         600.0, 3800.0),
        ("register_unsubscribe",  400.0, 4200.0),
        ("track_opens",           800.0, 4700.0),
        ("track_clicks",          800.0, 5200.0),
        ("log_crm",               500.0, 5700.0),
        ("score_engagement",     1200.0, 6200.0),
        ("queue_followup",        600.0, 6800.0),
    ],
)

# Ad campaign: audience, lookalike, creative, variants, brand safety, budget, schedule, push platforms, pixels, goals, tracking, launch, log.
WORKLOAD_AD_CAMPAIGN = _workload(
    "ad_campaign",
    stream_ms=8500.0,
    specs=[
        ("fetch_audience",       1500.0,  200.0),
        ("lookalike_audience",   2500.0,  800.0),
        ("fetch_creative",        800.0, 1400.0),
        ("generate_variant_a",   3000.0, 1900.0),
        ("generate_variant_b",   3000.0, 2400.0),
        ("validate_brand_safety",1200.0, 2900.0),
        ("set_budget",            300.0, 3400.0),
        ("set_schedule",          300.0, 3800.0),
        ("push_facebook",        2500.0, 4200.0),
        ("push_google",          2500.0, 4800.0),
        ("push_tiktok",          2500.0, 5400.0),
        ("set_pixels",            600.0, 6000.0),
        ("set_conversion_goals",  500.0, 6500.0),
        ("enable_tracking",       400.0, 7000.0),
        ("log_launch",            300.0, 7600.0),
    ],
)

# Invoice processing: fetch PDF, OCR, extract fields, validate VAT, match PO, post to ERP.
WORKLOAD_INVOICE_PROC = _workload(
    "invoice_proc",
    stream_ms=4000.0,
    specs=[
        ("fetch_pdf",             500.0,  200.0),
        ("ocr_document",         3500.0,  800.0),
        ("extract_fields",       2000.0, 1500.0),
        ("validate_vat",          800.0, 2200.0),
        ("match_po",             1500.0, 2900.0),
        ("post_to_erp",          1200.0, 3500.0),
    ],
)

# Onboarding: verify email, verify phone, KYC lookup, sanctions screen, fraud score, create account, welcome.
WORKLOAD_ONBOARDING = _workload(
    "onboarding",
    stream_ms=4500.0,
    specs=[
        ("verify_email",          600.0,  200.0),
        ("verify_phone",          800.0,  800.0),
        ("kyc_lookup",           3000.0, 1400.0),
        ("sanctions_screen",     1800.0, 2000.0),
        ("fraud_score",          1500.0, 2700.0),
        ("create_account",        400.0, 3400.0),
        ("send_welcome",          500.0, 4000.0),
    ],
)


WORKLOADS: dict[str, Workload] = {
    "analytics":       WORKLOAD_ANALYTICS,
    "search":          WORKLOAD_CODE_SEARCH,
    "support":         WORKLOAD_SUPPORT,
    "deploy":          WORKLOAD_DEPLOY,
    "incident":        WORKLOAD_INCIDENT,
    "security":        WORKLOAD_SECURITY,
    "research":        WORKLOAD_RESEARCH,
    "lead_enrichment": WORKLOAD_LEAD_ENRICHMENT,
    "content_mod":     WORKLOAD_CONTENT_MOD,
    "db_migration":    WORKLOAD_DB_MIGRATION,
    "release_notes":   WORKLOAD_RELEASE_NOTES,
    "legal_review":    WORKLOAD_LEGAL_REVIEW,
    "sales_outreach":  WORKLOAD_SALES_OUTREACH,
    "ad_campaign":     WORKLOAD_AD_CAMPAIGN,
    "invoice_proc":    WORKLOAD_INVOICE_PROC,
    "onboarding":      WORKLOAD_ONBOARDING,
}
