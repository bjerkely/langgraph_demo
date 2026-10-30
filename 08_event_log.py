"""08 - Event log -> structured incident report (exercise).

A lithography tool writes an event log: hundreds of lines, a dozen subsystems,
vendor-specific format. The question is not "what does line 412 say" but "what
happened, what caused it, and what do we do about it" -- one typed object.

The pipeline separates what code is good at from what models are good at:

    parse (your adapter)  ->  aggregate (Python)  ->  label (LLM, structured)
    ->  correlate (LLM, structured)  ->  verify + fill in (Python)

Consequences worth noticing:

  * The model never counts and never formats timestamps. Python does that.
    The model returns event ids, labels and causal links.
  * Because aggregation happens before prompting, log length is almost free:
    ~600 events become ~23 rows, and the run costs two calls.
  * Format is a config problem, not a model problem. `--format` accepts
    bracket | jsonl | kv | auto | path/to/spec.json.

Run:
    python 08_event_log.py                 # synthetic fixture, 2 LLM calls
    python 08_event_log.py --dry-run       # parse + aggregate only, no LLM
    python 08_event_log.py mytool.log --format auto
    python 08_event_log.py --out events.jsonl

Exercise (see TUTORIAL.md, Part 8): write a FormatSpec for your own log, then
break the verifier and watch a fabricated event id survive.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from common import ROOT, banner, structured

DATA = ROOT / "data"
DEFAULT_LOG = DATA / "litho_events.log"

# The synthetic fixture has a known answer. Real logs do not, which is why this
# is only used for a soft check and never to change the verdict.
FIXTURE_CHAIN = ["THM-1200", "PO-5502", "PO-7710", "ALN-3200", "WS-6213", "SAF-7777"]

SEVERITY_ORDER = {"DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40, "CRITICAL": 50}
SEVERITY_ALIASES = {
    "WARNING": "WARN",
    "ERR": "ERROR",
    "CRIT": "CRITICAL",
    "FATAL": "CRITICAL",
    "NOTICE": "INFO",
    "TRACE": "DEBUG",
}
TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


# --------------------------------------------------------------------------- #
# 1. Format layer: everything format-specific lives here
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FormatSpec:
    """How to turn one log line into a canonical Event.

    `pattern` uses named groups: ts, subsystem, severity, event_id and an
    optional `rest` group holding the payload. `field_map` is only used by the
    jsonl kind (canonical name -> key in the JSON object).
    """

    name: str
    kind: Literal["regex", "jsonl"] = "regex"
    pattern: str = ""
    timestamp_formats: tuple[str, ...] = TIMESTAMP_FORMATS
    payload: Literal["kv", "json", "none"] = "kv"
    field_map: dict[str, str] = field(default_factory=dict)


BUILTIN_SPECS: dict[str, FormatSpec] = {
    "bracket": FormatSpec(
        name="bracket",
        pattern=(
            r"^\[(?P<ts>[^\]]+)\]\[(?P<subsystem>[^\]]+)\]"
            r"\[(?P<severity>[^\]]+)\]\[(?P<event_id>[^\]]+)\]\s*(?P<rest>.*)$"
        ),
    ),
    "kv": FormatSpec(
        name="kv",
        pattern=(
            r"^(?P<ts>\S+)\s+(?P<severity>[A-Za-z]+)\s+"
            r"(?P<subsystem>[\w.-]+)\s+(?P<event_id>[\w.-]+)\s*(?P<rest>.*)$"
        ),
    ),
    "jsonl": FormatSpec(name="jsonl", kind="jsonl", payload="json"),
}


@dataclass(slots=True)
class Event:
    ts: datetime
    subsystem: str
    severity: str
    event_id: str
    payload: dict[str, Any]
    raw: str
    line_no: int


@dataclass
class ParseReport:
    parsed: int = 0
    skipped: int = 0
    samples: list[tuple[int, str]] = field(default_factory=list)

    def skip(self, line_no: int, raw: str) -> None:
        self.skipped += 1
        if len(self.samples) < 5:
            self.samples.append((line_no, raw.strip()[:120]))


@lru_cache(maxsize=32)
def _compiled(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


_KV = re.compile(r"""([A-Za-z_][\w.-]*)=("([^"]*)"|'([^']*)'|(\S+))""")


def _coerce(value: str) -> Any:
    text = value.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def normalize_severity(value: Any) -> str:
    text = str(value).strip().upper()
    return SEVERITY_ALIASES.get(text, text)


def parse_timestamp(value: Any, formats: tuple[str, ...] = TIMESTAMP_FORMATS) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in formats:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        return None
    # Mixed aware/naive logs would break sorting and comparisons later on.
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _parse_payload(rest: str, spec: FormatSpec) -> dict[str, Any]:
    if spec.payload == "none" or not rest.strip():
        return {}
    if spec.payload == "json":
        try:
            loaded = json.loads(rest)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {"value": loaded}
    payload: dict[str, Any] = {}
    for match in _KV.finditer(rest):
        raw = match.group(3) or match.group(4) or match.group(5) or ""
        payload[match.group(1)] = _coerce(raw)
    return payload


def _parse_json_line(text: str, spec: FormatSpec) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    flat = {str(k).lower(): v for k, v in obj.items()}

    def pick(canonical: str, *candidates: str) -> Any:
        keys = (spec.field_map[canonical],) if canonical in spec.field_map else candidates
        for key in keys:
            if key in flat and flat[key] not in (None, ""):
                return flat[key]
        return None

    ts = parse_timestamp(pick("ts", "ts", "timestamp", "time", "@timestamp"), spec.timestamp_formats)
    event_id = pick("event_id", "event_id", "event", "code", "id")
    if ts is None or event_id is None:
        return None
    consumed = set(spec.field_map.values()) | {
        "ts", "timestamp", "time", "@timestamp", "event_id", "event", "code", "id",
        "subsystem", "subsys", "component", "source", "severity", "level",
    }
    payload = {k: v for k, v in obj.items() if k.lower() not in consumed}
    return {
        "ts": ts,
        "subsystem": str(pick("subsystem", "subsystem", "subsys", "component", "source") or "UNKNOWN").upper(),
        "severity": normalize_severity(pick("severity", "severity", "level") or "INFO"),
        "event_id": str(event_id),
        "payload": payload,
    }


def parse_line(spec: FormatSpec, raw: str, line_no: int) -> Event | None:
    text = raw.strip()
    if not text:
        return None

    if spec.kind == "jsonl":
        fields = _parse_json_line(text, spec)
        if fields is None:
            return None
        return Event(raw=raw, line_no=line_no, **fields)

    match = _compiled(spec.pattern).match(text)
    if match is None:
        return None
    groups = match.groupdict()
    ts = parse_timestamp(groups.get("ts"), spec.timestamp_formats)
    if ts is None:
        return None
    return Event(
        ts=ts,
        subsystem=str(groups.get("subsystem") or "UNKNOWN").strip().upper(),
        severity=normalize_severity(groups.get("severity") or "INFO"),
        event_id=str(groups.get("event_id") or "").strip(),
        payload=_parse_payload(groups.get("rest") or "", spec),
        raw=raw,
        line_no=line_no,
    )


def parse_lines(lines: list[str], spec: FormatSpec) -> tuple[list[Event], ParseReport]:
    events: list[Event] = []
    report = ParseReport()
    for line_no, raw in enumerate(lines, start=1):
        event = parse_line(spec, raw, line_no)
        if event is None:
            report.skip(line_no, raw)
        else:
            events.append(event)
    report.parsed = len(events)
    return events, report


def load_spec_from_file(path: Path) -> FormatSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    return FormatSpec(
        name=data.get("name", path.stem),
        kind=data.get("kind", "regex"),
        pattern=data.get("pattern", ""),
        timestamp_formats=tuple(data.get("timestamp_formats") or TIMESTAMP_FORMATS),
        payload=data.get("payload", "kv"),
        field_map=data.get("field_map", {}),
    )


def detect_format(lines: list[str], sample_size: int = 200) -> list[tuple[float, str, int]]:
    """Score every built-in spec on a sample; returns [(ratio, name, matches)]."""
    sample = [line for line in lines if line.strip()][:sample_size]
    scores: list[tuple[float, str, int]] = []
    for name, spec in BUILTIN_SPECS.items():
        matches = sum(1 for i, line in enumerate(sample, 1) if parse_line(spec, line, i))
        scores.append((matches / len(sample) if sample else 0.0, name, matches))
    scores.sort(reverse=True)
    return scores


# --------------------------------------------------------------------------- #
# 2. Aggregation: the reason a 600-line log costs two calls, not six
# --------------------------------------------------------------------------- #
@dataclass
class EventGroup:
    event_id: str
    subsystems: list[str]
    count: int
    first: datetime
    last: datetime
    worst: str
    samples: list[str]
    numeric: dict[str, tuple[float, float]]


def aggregate(events: list[Event], window_seconds: int = 0) -> list[EventGroup]:
    buckets: dict[tuple[str, int], list[Event]] = {}
    for event in events:
        bucket = int(event.ts.timestamp()) // window_seconds if window_seconds else 0
        buckets.setdefault((event.event_id, bucket), []).append(event)

    groups: list[EventGroup] = []
    for (event_id, _), bucket_events in buckets.items():
        bucket_events.sort(key=lambda e: e.ts)
        numeric: dict[str, tuple[float, float]] = {}
        for event in bucket_events:
            for key, value in event.payload.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                low, high = numeric.get(key, (float(value), float(value)))
                numeric[key] = (min(low, float(value)), max(high, float(value)))
        groups.append(
            EventGroup(
                event_id=event_id,
                subsystems=sorted({e.subsystem for e in bucket_events}),
                count=len(bucket_events),
                first=bucket_events[0].ts,
                last=bucket_events[-1].ts,
                worst=max((e.severity for e in bucket_events), key=lambda s: SEVERITY_ORDER.get(s, 0)),
                samples=[e.raw.strip()[:200] for e in bucket_events[:2]],
                numeric=dict(sorted(numeric.items())[:5]),
            )
        )
    groups.sort(key=lambda g: (g.first, g.event_id))
    return groups


def render_groups(groups: list[EventGroup]) -> str:
    rows = []
    for g in groups:
        ranges = " ".join(f"{key}[{low:g}..{high:g}]" for key, (low, high) in g.numeric.items())
        rows.append(
            f"{g.event_id} | {','.join(g.subsystems)} | {g.worst} | n={g.count} | "
            f"{_clock(g.first)}..{_clock(g.last)} | {ranges}\n"
            f"    sample: {g.samples[0] if g.samples else ''}"
        )
    return "\n".join(rows)


def _clock(value: datetime) -> str:
    return f"{value:%H:%M:%S}.{value.microsecond // 1000:03d}"


# --------------------------------------------------------------------------- #
# 3. The two structured calls
# --------------------------------------------------------------------------- #
class EventLabel(BaseModel):
    """What one event id means for the tool, independent of its wording."""

    event_id: str = Field(description="Event id exactly as listed, e.g. THM-1200.")
    category: Literal[
        "thermal", "mechanical", "optical", "electrical", "fluid", "vacuum",
        "control", "safety", "process", "maintenance", "noise", "unknown",
    ] = Field(description="Physical domain this event belongs to.")
    significance: Literal["noise", "expected", "degraded", "fault", "safety"] = Field(
        description=(
            "noise/expected = harmless or routine; degraded = performance slipping; "
            "fault = a real failure; safety = interlock or product-hold."
        )
    )
    physical_effect: str = Field(
        description="One sentence on the physical effect. Write 'none' for noise/expected."
    )
    root_cause_candidate: bool = Field(
        description="True only if this event could START a fault chain, not if it is a symptom."
    )


class LabelSet(BaseModel):
    labels: list[EventLabel] = Field(description="One label per event id in the summary.")


class CausalLink(BaseModel):
    cause_event: str = Field(description="Event id that comes first in time.")
    effect_event: str = Field(description="Event id it produces.")
    mechanism: str = Field(description="How the cause physically produces the effect, one sentence.")


class IncidentReport(BaseModel):
    """The answer: what happened, why, and what to do. References only."""

    title: str = Field(description="One line naming the failure, e.g. 'Chiller drift degraded overlay'.")
    severity: Literal["low", "medium", "high", "critical"]
    root_cause_event: str = Field(description="Single event id from the summary that started it.")
    causal_chain: list[CausalLink] = Field(description="Ordered cause -> effect links, event ids only.")
    contributing_events: list[str] = Field(description="Other event ids that made it worse.")
    wafer_impact: Literal["none", "possible", "confirmed"]
    containment: str = Field(description="What the operator should do now, two sentences max.")
    escalate: bool = Field(description="True if severity is high or critical.")


def label_events(groups: list[EventGroup]) -> dict[str, EventLabel]:
    prompt = (
        "You are a reliability engineer reading an aggregated event summary from a "
        "lithography machine. Every row is one event id; n= is how many times it fired, "
        "and the bracketed ranges are min..max of its numeric payload fields.\n\n"
        "Classify every event id. Judge the physical effect on the tool, not the wording "
        "of the message. Routine housekeeping is 'expected' or 'noise'. Only events that "
        "can move overlay, dose, throughput or safety are 'degraded', 'fault' or 'safety'. "
        "Mark root_cause_candidate only for events that could start a fault chain -- a "
        "consequence of another event is candidate=false.\n\n"
        "Numeric context matters more than log level: compare values against setpoint or "
        "limit fields on the same row. A roughly 1 K chiller deviation, or a differential "
        "pressure above its limit, is a real thermal fault even when the line says WARN. "
        "A late ERROR is often a symptom of an early WARN.\n\n"
        "physical_effect must be one full sentence naming the physical quantity and its "
        "consequence (for example 'lens temperature exceeds its limit, so aberration "
        "correction runs out of headroom'), never a single category word.\n\n"
        f"EVENT SUMMARY\n{render_groups(groups)}"
    )
    labels = structured(LabelSet).invoke(prompt).labels
    return {label.event_id: label for label in labels}


def correlate(
    groups: list[EventGroup],
    labels: dict[str, EventLabel],
    failures: list[str] | None = None,
) -> IncidentReport:
    table = "\n".join(
        f"{g.event_id} | {label.category if (label := labels.get(g.event_id)) else '?'} | "
        f"{label.significance if label else '?'} | "
        f"root_candidate={label.root_cause_candidate if label else '?'} | "
        f"{label.physical_effect if label else '?'}"
        for g in groups
    )
    timeline = "\n".join(
        f"{g.event_id} | {_clock(g.first)}..{_clock(g.last)} | n={g.count} | {','.join(g.subsystems)}"
        for g in groups
    )
    prompt = (
        "You are the incident reviewer for a lithography machine. Using only the event ids "
        "below, pick the single root cause and the causal chain that produced the product "
        "impact. Each link must connect two listed ids and must be ordered in time.\n\n"
        "Severity is not causality. The root cause is normally the earliest event that can "
        "physically drive the others: a WARN-level thermal, fluid, vacuum or electrical "
        "excursion can cause later ERROR-level symptoms. Prefer the chain that explains "
        "every product-impacting symptom over one that starts at a mid-chain error.\n\n"
        "Do not invent ids. Do not report counts or timestamps -- they are filled in later.\n\n"
        f"LABELS\n{table}\n\nTIMELINE\n{timeline}"
    )
    if failures:
        prompt += (
            "\n\nYour previous answer failed these checks. Fix them and answer again:\n- "
            + "\n- ".join(failures)
        )
    report = structured(IncidentReport).invoke(prompt)
    return report


# --------------------------------------------------------------------------- #
# 4. Verification and fill-in: the model's answer is a hypothesis until checked
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    count: int
    first: datetime
    last: datetime
    subsystems: list[str]


def summarize(groups: list[EventGroup]) -> dict[str, Stats]:
    return {
        g.event_id: Stats(g.count, g.first, g.last, g.subsystems)
        for g in groups
    }


def referenced_ids(report: IncidentReport) -> set[str]:
    ids = {report.root_cause_event, *report.contributing_events}
    for link in report.causal_chain:
        ids.add(link.cause_event)
        ids.add(link.effect_event)
    return {i for i in ids if i}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hard: bool = True


def verify(
    report: IncidentReport,
    stats: dict[str, Stats],
    labels: dict[str, EventLabel],
    expected_chain: list[str] | None = None,
) -> list[Check]:
    used = referenced_ids(report)
    unknown = sorted(used - stats.keys())
    unlabelled = sorted(used - labels.keys())
    root = labels.get(report.root_cause_event)

    checks = [
        Check("ids exist in log", not unknown, f"unknown: {unknown}" if unknown else f"{len(used)} ids cited"),
        Check(
            "cited ids are labelled",
            not unlabelled,
            f"unlabelled: {unlabelled}" if unlabelled else "all cited ids labelled",
        ),
        Check(
            "root cause was a candidate",
            root is not None and root.root_cause_candidate,
            f"{report.root_cause_event} candidate={getattr(root, 'root_cause_candidate', None)}",
        ),
        Check("chain is non-empty", bool(report.causal_chain), f"{len(report.causal_chain)} links"),
        Check(
            "escalation matches severity",
            report.escalate == (report.severity in {"high", "critical"}),
            f"severity={report.severity} escalate={report.escalate}",
        ),
    ]

    backwards = [
        f"{link.cause_event}->{link.effect_event}"
        for link in report.causal_chain
        if link.cause_event in stats
        and link.effect_event in stats
        and stats[link.cause_event].first > stats[link.effect_event].first
    ]
    checks.append(
        Check(
            "chain follows the clock",
            not backwards,
            f"backwards in time: {backwards}" if backwards else "every cause precedes its effect",
        )
    )

    if expected_chain:
        # The fixture has a known answer. This is a soft check: a real log does not.
        root_ok = report.root_cause_event == expected_chain[0]
        recovered = [event_id for event_id in expected_chain if event_id in used]
        checks.append(
            Check(
                "fixture chain recovered",
                root_ok and len(recovered) == len(expected_chain),
                f"root={report.root_cause_event} (expected {expected_chain[0]}), "
                f"{len(recovered)}/{len(expected_chain)} chain ids cited",
                hard=False,
            )
        )
    return checks


def _event_line(event_id: str, prefix: str, stats: dict[str, Stats], labels: dict[str, EventLabel]) -> str:
    stat = stats.get(event_id)
    label = labels.get(event_id)
    where = f"{_clock(stat.first)}..{_clock(stat.last)}, n={stat.count}" if stat else "?"
    what = label.physical_effect if label else "?"
    return f"{prefix}{event_id:<12} {where:<28} {what}"


def _is_path(links: list[CausalLink]) -> bool:
    return all(links[i].effect_event == links[i + 1].cause_event for i in range(len(links) - 1))


def render_report(report: IncidentReport, stats: dict[str, Stats], labels: dict[str, EventLabel]) -> str:
    lines = [
        report.title,
        f"severity: {report.severity}    wafer impact: {report.wafer_impact}    escalate: {'yes' if report.escalate else 'no'}",
        "",
    ]
    if _is_path(report.causal_chain):
        for index, link in enumerate(report.causal_chain):
            if index == 0:
                lines.append(_event_line(link.cause_event, "  ", stats, labels))
            lines.append(_event_line(link.effect_event, "    -> ", stats, labels))
            lines.append(f"{' ' * 7}mechanism: {link.mechanism}")
    else:
        for link in report.causal_chain:
            lines.append(_event_line(link.cause_event, "  ", stats, labels))
            lines.append(_event_line(link.effect_event, "    -> ", stats, labels))
            lines.append(f"{' ' * 7}mechanism: {link.mechanism}")
            lines.append("")

    root = stats.get(report.root_cause_event)
    if root:
        lines.append("")
        lines.append(
            f"root cause: {report.root_cause_event} ({root.subsystems[0]}, {_clock(root.first)}, n={root.count})"
        )
    lines.append(f"containment: {report.containment}")
    return "\n".join(lines)


def referenced_payload(events: list[Event], report: IncidentReport, key: str) -> list[str]:
    wanted = referenced_ids(report)
    values: list[str] = []
    for event in events:
        if event.event_id not in wanted:
            continue
        raw = event.payload.get(key)
        if raw is None:
            continue
        for value in str(raw).split(","):
            value = value.strip()
            if value and value not in values:
                values.append(value)
    return values


def write_jsonl(path: Path, events: list[Event], labels: dict[str, EventLabel]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            label = labels.get(event.event_id)
            handle.write(
                json.dumps(
                    {
                        "ts": event.ts.isoformat(),
                        "line_no": event.line_no,
                        "subsystem": event.subsystem,
                        "severity": event.severity,
                        "event_id": event.event_id,
                        "payload": event.payload,
                        "category": label.category if label else None,
                        "significance": label.significance if label else None,
                        "physical_effect": label.physical_effect if label else None,
                    },
                    default=str,
                )
                + "\n"
            )


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #
def resolve_spec(name: str, lines: list[str]) -> FormatSpec:
    if name == "auto":
        scores = detect_format(lines)
        for ratio, spec_name, matches in scores:
            print(f"[auto] {spec_name:<8} {ratio:6.1%} match ({matches} sample lines)")
        best = next((s for r, s, _ in scores if r > 0), None)
        if best is None:
            raise SystemExit("auto-detect found no matching format; pass --format-file")
        print(f"[auto] using {best}")
        return BUILTIN_SPECS[best]
    if name in BUILTIN_SPECS:
        return BUILTIN_SPECS[name]
    path = Path(name)
    if path.exists():
        return load_spec_from_file(path)
    raise SystemExit(f"unknown format {name!r}; choose {', '.join(BUILTIN_SPECS)}, auto, or a spec file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_LOG, help="event log to read")
    parser.add_argument("--format", default="bracket", help="bracket | jsonl | kv | auto | spec.json")
    parser.add_argument("--min-severity", default="DEBUG", choices=sorted(SEVERITY_ORDER))
    parser.add_argument("--group-window", type=int, default=0, metavar="SECONDS",
                        help="split groups into time buckets (0 = whole log)")
    parser.add_argument("--limit", type=int, default=0, help="read only the first N lines")
    parser.add_argument("--out", type=Path, help="write the per-event typed table as JSONL")
    parser.add_argument("--dry-run", action="store_true", help="stop after aggregation; no LLM calls")
    parser.add_argument("--list-formats", action="store_true", help="show built-in format specs and exit")
    args = parser.parse_args()

    if args.list_formats:
        for spec in BUILTIN_SPECS.values():
            print(f"{spec.name:<8} kind={spec.kind:<5} payload={spec.payload}")
        print("\nSpec files accept: name, kind, pattern, timestamp_formats, payload, field_map")
        return

    lines = args.path.read_text(encoding="utf-8", errors="replace").splitlines()
    if args.limit:
        lines = lines[: args.limit]

    banner(f"1. Parse: {args.path}")
    spec = resolve_spec(args.format, lines)
    events, report = parse_lines(lines, spec)
    print(f"{report.parsed} events parsed, {report.skipped} lines skipped  (format: {spec.name})")
    for line_no, sample in report.samples:
        print(f"  skipped line {line_no}: {sample!r}")
    if not events:
        raise SystemExit("no events parsed; try --format auto or a spec file")

    floor = SEVERITY_ORDER[args.min_severity]
    kept = [event for event in events if SEVERITY_ORDER.get(event.severity, 0) >= floor]
    if floor > SEVERITY_ORDER["DEBUG"]:
        print(f"severity filter {args.min_severity}: kept {len(kept)}/{len(events)} events")

    banner("2. Aggregate: many lines -> few rows")
    groups = aggregate(kept, args.group_window)
    print(f"{len(kept)} events -> {len(groups)} groups")
    print()
    print(render_groups(groups))
    if args.dry_run:
        print("\n[dry-run] stopping before the LLM calls")
        return

    banner("3. Label: one structured call over the summary")
    labels = label_events(groups)
    for label in sorted(labels.values(), key=lambda l: l.significance):
        print(f"{label.event_id:<12} {label.category:<11} {label.significance:<9} "
              f"root={str(label.root_cause_candidate):<5} {label.physical_effect}")

    banner("4. Correlate: one structured call over the labels")
    report_obj = correlate(groups, labels)

    stats = summarize(groups)
    expected = FIXTURE_CHAIN if args.path.resolve() == DEFAULT_LOG.resolve() else None
    checks = verify(report_obj, stats, labels, expected)
    if any(not check.ok and check.hard for check in checks):
        print("verification failed, asking for one repair:")
        for check in checks:
            if not check.ok and check.hard:
                print(f"  - {check.name}: {check.detail}")
        report_obj = correlate(groups, labels, [f"{c.name}: {c.detail}" for c in checks if not c.ok and c.hard])
        checks = verify(report_obj, stats, labels, expected)

    banner("5. Verify (Python, not the model)")
    for check in checks:
        flag = "ok  " if check.ok else ("FAIL" if check.hard else "warn")
        print(f"[{flag}] {check.name:<28} {check.detail}")

    banner("6. Report")
    print(render_report(report_obj, stats, labels))
    if expected:
        print(f"expected chain (fixture): {' -> '.join(expected)}")
    lots = referenced_payload(events, report_obj, "lots") or referenced_payload(events, report_obj, "lot")
    if lots:
        print(f"lots in evidence: {', '.join(lots)}")

    if args.out:
        write_jsonl(args.out, events, labels)
        print(f"\n[done] per-event records written to {args.out}")

    if any(not check.ok for check in checks):
        print("\nnote: at least one check still fails; that is the exercise working, not a crash.")
        sys.exit(0)


if __name__ == "__main__":
    main()
