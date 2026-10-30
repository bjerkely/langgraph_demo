"""Generate the synthetic lithography-tool event log used by 08_event_log.py.

The log is deliberately fake: one 8-minute window, ~550 events across 12
subsystems, one cross-subsystem fault chain, several red herrings and a handful
of malformed lines. It exists so the exercise is deterministic and free to run;
point 08 at a real log with `--format` when you have one.

Format (vendor-style, brackets, key=value payload):

    [2026-10-27 09:12:04.118][THERMAL][WARN][THM-1200] chiller_temp=19.4 setpoint=20.0 dev_k=-0.6

Usage:
    python data/generate_litho_events.py
    python data/generate_litho_events.py --seed 11 --out /tmp/big.log
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START = datetime(2026, 10, 27, 9, 10, 0)
END = START + timedelta(minutes=8, seconds=30)

LOTS = ["LOT-4468", "LOT-4469", "LOT-4470", "LOT-4471", "LOT-4472"]


def _ts(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.") + f"{value.microsecond // 1000:03d}"


def _event(ts: datetime, subsystem: str, level: str, event_id: str, **payload: object) -> str:
    body = " ".join(f"{k}={v}" for k, v in payload.items())
    return f"[{_ts(ts)}][{subsystem}][{level}][{event_id}] {body}".rstrip()


def _series(start: datetime, end: datetime, step: float, rng: random.Random, jitter: float = 0.0):
    """Yield timestamps every `step` seconds, optionally jittered."""
    offset = 0.0
    while True:
        base = start + timedelta(seconds=offset)
        if base >= end:
            return
        yield base + timedelta(seconds=rng.uniform(-jitter, jitter) if jitter else 0.0)
        offset += step


def build(seed: int) -> list[str]:
    rng = random.Random(seed)
    rows: list[tuple[datetime, int, str]] = []
    seq = 0

    def add(ts: datetime, line: str) -> None:
        nonlocal seq
        rows.append((ts, seq, line))
        seq += 1

    # --- routine traffic -----------------------------------------------------
    # One noisy event id dominates: aggregation should collapse ~340 lines into
    # a single row the labeller can dismiss as noise.
    for ts in _series(START, END, 1.5, rng, jitter=0.4):
        add(ts, _event(ts, "CONTROL", "WARN", "CTL-0012",
                       deadline_miss_us=rng.randint(280, 940), core=rng.choice(["rt3", "rt7"])))

    for i, ts in enumerate(_series(START, END, 18, rng, jitter=1.0)):
        add(ts, _event(ts, "SOURCE", "INFO", "SRC-0100",
                       rep_rate_hz=6000, energy_mj=round(rng.uniform(5.0, 5.4), 3)))

    for i, ts in enumerate(_series(START, END, 40, rng, jitter=2.0)):
        add(ts, _event(ts, "CONTROL", "INFO", "CTL-0200",
                       lot=LOTS[min(i // 3, len(LOTS) - 1)],
                       recipe=f"RCP-{4100 + i % 3}"))

    for i, ts in enumerate(_series(START, END, 7, rng, jitter=0.5)):
        add(ts, _event(ts, "WAFER_HANDLER", "INFO", "WH-2200",
                       slot=i % 25 + 1, wafer=f"W{44680 + i:05d}"))

    for ts in _series(START, END, 25, rng, jitter=1.5):
        add(ts, _event(ts, "ILLUMINATION", "INFO", "ILL-1100",
                       dose_setpoint_mj_cm2=round(rng.uniform(31.5, 32.0), 2)))

    for ts in _series(START, END, 30, rng, jitter=1.0):
        add(ts, _event(ts, "VACUUM", "INFO", "VAC-9000",
                       pressure_pa=round(rng.uniform(0.008, 0.014), 4)))

    for ts in _series(START, END, 10, rng, jitter=0.5):
        add(ts, _event(ts, "PROJECTION_OPTICS", "INFO", "PO-5400",
                       lens_temp_c=round(rng.uniform(31.0, 31.7), 2)))

    for ts in _series(START, END, 55, rng, jitter=3.0):
        add(ts, _event(ts, "ALIGNMENT", "INFO", "ALN-3001",
                       residual_nm=round(rng.uniform(1.4, 2.1), 2), marks=64))

    for ts in _series(START, START + timedelta(minutes=2), 20, rng):
        add(ts, _event(ts, "THERMAL", "INFO", "THM-1300",
                       chiller_temp=20.0, setpoint=20.0, dev_k=0.0))

    # --- red herrings --------------------------------------------------------
    add(START + timedelta(seconds=30), _event(
        START + timedelta(seconds=30), "CONTROL", "INFO", "CTL-0100",
        pm_window_scheduled="2026-10-28T02:00Z", duration_h=4))

    for i, ts in enumerate([START + timedelta(seconds=70), START + timedelta(seconds=110),
                            START + timedelta(seconds=150)]):
        add(ts, _event(ts, "VACUUM", "WARN", "VAC-9100",
                       pressure_rise_pa=round(0.6 + i * 0.1, 2)))

    add(START + timedelta(seconds=165), _event(
        START + timedelta(seconds=165), "VACUUM", "INFO", "VAC-9101", vacuum_recovered_pa=0.01))

    add(START + timedelta(seconds=55), _event(
        START + timedelta(seconds=55), "WAFER_HANDLER", "WARN", "WH-2100", grip_retry_attempt=1))
    add(START + timedelta(seconds=65), _event(
        START + timedelta(seconds=65), "WAFER_HANDLER", "WARN", "WH-2100", grip_retry_attempt=2))
    add(START + timedelta(seconds=68), _event(
        START + timedelta(seconds=68), "WAFER_HANDLER", "INFO", "WH-2101", grip_ok=1))

    # --- the incident chain --------------------------------------------------
    # THM-1200 -> PO-5502 -> PO-7710 -> ALN-3200 -> WS-6213 -> SAF-7777
    # The filter differential pressure is the actual cause; the temperature
    # deviation is the symptom. It is deliberately only a WARN event: severity
    # is not causality.
    for i, ts in enumerate(_series(datetime(2026, 10, 27, 9, 12, 4),
                                   datetime(2026, 10, 27, 9, 17, 55), 25, rng)):
        add(ts, _event(ts, "THERMAL", "WARN", "THM-1200",
                       chiller_temp=round(19.4 - i * 0.03, 2), setpoint=20.0,
                       dev_k=round(-0.6 - i * 0.03, 2),
                       filter_dp_kpa=182 + i * 5, limit_kpa=150))

    for ts, temp in [(datetime(2026, 10, 27, 9, 12, 5), 32.8),
                     (datetime(2026, 10, 27, 9, 14, 10), 33.1),
                     (datetime(2026, 10, 27, 9, 16, 20), 33.6)]:
        add(ts, _event(ts, "PROJECTION_OPTICS", "ERROR", "PO-5502",
                       lens_temp_c=temp, limit_c=32.0))

    for ts in [datetime(2026, 10, 27, 9, 12, 5, 117000),
               datetime(2026, 10, 27, 9, 16, 22, 440000)]:
        add(ts, _event(ts, "PROJECTION_OPTICS", "ERROR", "PO-7710",
                       aberr_corr_pct=100, status="AT_LIMIT"))

    for ts, resid in [(datetime(2026, 10, 27, 9, 12, 7), 3.8),
                      (datetime(2026, 10, 27, 9, 13, 30), 4.1),
                      (datetime(2026, 10, 27, 9, 15, 10), 4.4),
                      (datetime(2026, 10, 27, 9, 16, 40), 4.9),
                      (datetime(2026, 10, 27, 9, 17, 40), 5.1)]:
        add(ts, _event(ts, "ALIGNMENT", "WARN", "ALN-3200",
                       grid_residual_nm=resid, limit_nm=2.5))

    for ts, err in [(datetime(2026, 10, 27, 9, 12, 9), 4.6),
                    (datetime(2026, 10, 27, 9, 14, 12), 5.1),
                    (datetime(2026, 10, 27, 9, 16, 23), 5.8),
                    (datetime(2026, 10, 27, 9, 17, 44), 6.2)]:
        add(ts, _event(ts, "WAFER_STAGE", "ERROR", "WS-6213",
                       following_error_nm=err, limit_nm=2.0))

    add(datetime(2026, 10, 27, 9, 17, 58), _event(
        datetime(2026, 10, 27, 9, 17, 58), "SAFETY", "CRITICAL", "SAF-7777",
        interlock="LOT_HOLD", lots="LOT-4471,LOT-4472"))

    add(datetime(2026, 10, 27, 9, 18, 1), _event(
        datetime(2026, 10, 27, 9, 18, 1), "CONTROL", "ERROR", "CTL-0400",
        matching_aborted="INTERLOCK"))

    add(datetime(2026, 10, 27, 9, 18, 3), _event(
        datetime(2026, 10, 27, 9, 18, 3), "SAFETY", "INFO", "SAF-7778",
        lot_hold_ack="OP-2211"))

    add(datetime(2026, 10, 27, 9, 18, 20), _event(
        datetime(2026, 10, 27, 9, 18, 20), "THERMAL", "INFO", "THM-1201",
        chiller_recovered=1, dev_k=0.0))

    # --- deliberate format noise --------------------------------------------
    add(START, "--- log rotated: litho01-events.1 ---")
    add(datetime(2026, 10, 27, 9, 13, 0), "[2026-10-27 09:13:00.000][THERMAL][WARN]")
    add(datetime(2026, 10, 27, 9, 13, 10), _event(
        datetime(2026, 10, 27, 9, 13, 10), "control", "warn", "CTL-0012",
        deadline_miss_us=812, core="rt7"))

    rows.sort(key=lambda row: (row[0], row[1]))
    return [line for _, _, line in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "litho_events.log")
    args = parser.parse_args()

    lines = build(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
