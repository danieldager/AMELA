#!/usr/bin/env python3
"""
Build a reproducible symlinked dataset organized as:
  out_root/
    MANIFEST.json
    build_config.json
    index.jsonl            # or assignment.csv if --plan-only
    000/
      chunk_0000/
        00000__foo.wav -> <relative link to original>
        ...
      chunk_0001/
    001/
      ...

Core rules implemented:
- 10-minute chunks (target <= chunk_sec + chunk_eps), strictly continuous by (top, mid, sequence+1).
- A chunk never crosses top_file.
- After each chunk, rotate to the next top_file (round-robin) if available.
- ~5-hour segments (<= segment_sec + segment_eps) made of whole chunks.
- Deterministic order: always (top, mid, sequence) ascending. Never re-use a file.
- Plan-only mode: emit assignment.csv (segment, chunk, order_in_chunk, ...), no symlinks.

Tested on macOS/Linux. Windows symlinks may require dev mode/admin.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------- Tunable defaults ----------
TARGET_SEGM = 600.0  # 10 minutes in seconds
ERROR_SEGM = 5.0  # allow tiny overshoot per chunk
TARGET_CHUNK = 18000.0  # 5 hours in seconds
ERROR_CHUNK = 60.0  # allow small overshoot per segment

# Row = Tuple[Path, float, str, str, int]  # (src_path, duration, top, mid, seq)
Row = Tuple[str, str, int, float]  # (top, mid, seq, duration)


# ---------- Small utilities ----------
def sha256_file(p: Path) -> str:
    """Hash a file (for reproducibility manifest)."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _first_present(d: dict, names: List[str], required: bool = True):
    """Return the first key in names present in dict d (value)."""
    for k in names:
        if k in d and d[k] not in (None, ""):
            return d[k]
    if required:
        raise KeyError(f"Missing one of required fields: {names}")
    return None


def read_csv_rows(csv_path: Path, root: Path) -> List[Row]:
    """
    Load your metadata CSV. Flexible field names:
      path field:     one of ['path','filepath','file','abs_path']
      duration field: one of ['duration_sec','duration','secs','seconds']
      top field:      one of ['top_file','top-file','top']
      mid field:      one of ['mid_file','mid-file','mid']
      seq field:      one of ['sequence','seq','index']
    Returns sorted rows by (top, mid, sequence).
    """
    rows: List[Row] = []
    with csv_path.open(newline="") as f:
        r = csv.DictReader(f)
        for line_no, d in enumerate(r, start=2):  # header is line 1
            # Extract columns
            # src_raw  = _first_present(d, ['path','filepath','file','abs_path'])
            dur = _first_present(d, ["duration_sec", "duration", "secs", "seconds"])
            top = _first_present(d, ["top_file", "top-file", "top"])
            mid = _first_present(d, ["mid_file", "mid-file", "mid"])
            seq = _first_present(d, ["sequence", "seq", "index"])

            # # Normalize/parse
            # src = Path(src_raw)
            # if not src.is_absolute():
            #     src = (root / src).resolve()

            # Check types
            if (
                not isinstance(top, str)
                or not isinstance(mid, str)
                or not isinstance(seq, int)
                or not isinstance(dur, float)
            ):
                raise ValueError(
                    f"Bad types at CSV line {line_no}: top({type(top)}), mid({type(mid)}), seq({type(seq)}), dur({type(dur)})"
                )

            # rows.append( (src, dur, top, mid, seq) )
            rows.append((top, mid, seq, dur))

    # Deterministic ordering is crucial for reproducibility
    rows.sort(key=lambda x: (x[0], x[1], x[2]))  # (top, mid, sequence)
    return rows


def build_runs(rows_by_top_mid: Dict[Tuple[str, str], List[Row]]):
    """
    For each (top, mid), precompute 'runs' of strict continuity where sequence increments by 1.
    Output: dict[(top,mid)] = List[List[int]], where inner list holds indices into the per-(top,mid) array.
    """
    runs_by_top_mid: Dict[Tuple[str, str], List[List[int]]] = {}
    for key, lst in rows_by_top_mid.items():
        runs: List[List[int]] = []
        if lst:
            run = [0]
            for i in range(1, len(lst)):
                if lst[i][4] == lst[i - 1][4] + 1:  # sequence + 1
                    run.append(i)
                else:
                    runs.append(run)
                    run = [i]
            runs.append(run)
        runs_by_top_mid[key] = runs
    return runs_by_top_mid


def make_rel_symlink(src: Path, link: Path):
    """Create a relative symlink (portable within the project tree)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, start=link.parent)
    try:
        os.symlink(rel, link)
    except FileExistsError:
        # Idempotent behavior: if a link already exists, keep going.
        pass


# ---------- Planner + builder ----------
def main():
    ap = argparse.ArgumentParser(
        description="Plan and/or build a symlinked dataset into 10-min chunks and ~5-hour segments."
    )
    ap.add_argument("--csv", required=True, type=Path, help="Path to metadata CSV.")
    ap.add_argument(
        "--original-root",
        required=True,
        type=Path,
        help="Root used to resolve relative CSV paths.",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        type=Path,
        help="Output directory for the symlinked dataset.",
    )
    ap.add_argument(
        "--chunk-sec",
        type=float,
        default=TARGET_SEGM,
        help="Target seconds per chunk (default 600).",
    )
    ap.add_argument(
        "--chunk-eps",
        type=float,
        default=ERROR_SEGM,
        help="Allowed chunk overshoot in seconds (default 5).",
    )
    ap.add_argument(
        "--segment-sec",
        type=float,
        default=TARGET_CHUNK,
        help="Target seconds per segment (default 18000).",
    )
    ap.add_argument(
        "--segment-eps",
        type=float,
        default=ERROR_CHUNK,
        help="Allowed segment overshoot in seconds (default 60).",
    )
    ap.add_argument(
        "--plan-only",
        action="store_true",
        help="Only write assignment.csv (no directories or symlinks).",
    )
    ap.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Refuse to overwrite an existing out-root.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and plan, but do not write anything.",
    )
    args = ap.parse_args()

    # ---- Prepare output dir atomically (build in .tmp then rename) ----
    tmp_out = args.out_root.with_suffix(".tmp")
    if args.fail_if_exists and args.out_root.exists():
        print(f"[ERR] Output exists: {args.out_root}", file=sys.stderr)
        sys.exit(2)
    if tmp_out.exists():
        print(f"[ERR] Temporary exists: {tmp_out}", file=sys.stderr)
        sys.exit(2)
    if not args.dry_run:
        tmp_out.mkdir(parents=True, exist_ok=False)

    # ---- Load + hash CSV (reproducibility anchor) ----
    csv_hash = sha256_file(args.csv)
    rows = read_csv_rows(args.csv, args.original_root)

    # ---- Group by (top, mid), preserving order ----
    by_top_mid: Dict[Tuple[str, str], List[Row]] = defaultdict(list)
    for r in rows:
        by_top_mid[(r[2], r[3])].append(r)

    # ---- Precompute continuous runs within each (top,mid) ----
    runs_by_top_mid = build_runs(by_top_mid)

    # ---- Initialize pointers (run_idx, elem_idx) per (top,mid) ----
    ptrs: Dict[Tuple[str, str], List[int]] = {}
    for key in runs_by_top_mid.keys():
        ptrs[key] = [0, 0]

    # ---- Build the round-robin top_file queue ----
    top_files = sorted({r[2] for r in rows})

    # Keep tops that actually have at least one run with content
    def top_has_content(t: str) -> bool:
        for tt, _m in runs_by_top_mid.keys():
            if tt != t:
                continue
            run_idx, elem_idx = ptrs[(tt, _m)]
            runs = runs_by_top_mid[(tt, _m)]
            # advance to next non-empty run if needed (read-only check)
            while run_idx < len(runs) and elem_idx >= len(runs[run_idx]):
                run_idx += 1
                elem_idx = 0
            if run_idx < len(runs):
                return True
        return False

    top_queue = deque([t for t in top_files if top_has_content(t)])
    mids_by_top: Dict[str, List[str]] = {
        t: sorted({k[1] for k in runs_by_top_mid if k[0] == t}) for t in top_files
    }

    # ---- Helpers to fetch/advance the next file within one top_file ----
    def next_file_in_top(t: str) -> Optional[Tuple[Row, Tuple[str, str], int, int]]:
        """
        Return the next available Row inside 't', respecting continuity:
        we stay within the current run; if run exhausted, we move to next run of same mid,
        and if mid exhausted, we move to next mid (still inside 't').
        """
        for m in mids_by_top.get(t, []):
            key = (t, m)
            runs = runs_by_top_mid.get(key, [])
            if not runs:
                continue
            run_idx, elem_idx = ptrs[key]
            # slide to next run if the current run is exhausted
            while run_idx < len(runs) and elem_idx >= len(runs[run_idx]):
                run_idx += 1
                elem_idx = 0
            if run_idx >= len(runs):
                continue
            # get the actual Row (index i in the per-(top,mid) list)
            i = runs[run_idx][elem_idx]
            return by_top_mid[key][i], key, run_idx, elem_idx
        return None

    def advance_ptr(key: Tuple[str, str], run_idx: int, elem_idx: int):
        """Advance pointer by one element; if run ends, move to next run."""
        runs = runs_by_top_mid[key]
        elem_idx += 1
        if elem_idx >= len(runs[run_idx]):
            run_idx += 1
            elem_idx = 0
        ptrs[key] = [run_idx, elem_idx]

    # ---- Writers: index.jsonl and/or assignment.csv ----
    # We always plan; we optionally link.
    index_path = tmp_out / "index.jsonl"
    assign_path = tmp_out / "assignment.csv"
    json_f = None
    csv_f = None
    csv_w = None

    if not args.dry_run:
        if args.plan_only:
            csv_f = assign_path.open("w", newline="")
            csv_w = csv.writer(csv_f)
            csv_w.writerow(
                [
                    "segment",
                    "chunk",
                    "order_in_chunk",
                    "src_path",
                    "duration_sec",
                    "top_file",
                    "mid_file",
                    "sequence",
                ]
            )
        else:
            json_f = index_path.open("w", encoding="utf-8")

    # ---- Segment/chunk assembly loop ----
    seg_idx = 0
    seg_sum = 0.0
    chunk_global = 0

    def start_new_segment():
        nonlocal seg_sum
        seg_sum = 0.0

    start_new_segment()

    while top_queue:
        t = top_queue.popleft()

        # Build ONE chunk from this top (<= chunk_sec + chunk_eps)
        chunk_sum = 0.0
        chunk_rows: List[Tuple[Row, Tuple[str, str], int, int]] = []

        while True:
            nxt = next_file_in_top(t)
            if nxt is None:
                break  # this top exhausted
            row, key, run_idx, elem_idx = nxt
            dur = row[1]
            if chunk_sum + dur <= args.chunk_sec + args.chunk_eps:
                chunk_rows.append((row, key, run_idx, elem_idx))
                chunk_sum += dur
                advance_ptr(key, run_idx, elem_idx)
                # If we reached target (within tolerance), we can stop this chunk
                if chunk_sum >= args.chunk_sec - 1e-9:
                    break
            else:
                # adding this file would overshoot chunk beyond tolerance -> stop here
                break

        if not chunk_rows:
            # No more content in this top; do NOT requeue it.
            continue

        # Segment boundary logic: keep chunks intact
        if (
            seg_sum >= args.segment_sec - 1e-9
            and seg_sum + chunk_sum > args.segment_sec + args.segment_eps
        ):
            # Start a new segment BEFORE placing this chunk
            seg_idx += 1
            start_new_segment()

        # ----- Materialize this chunk: either write assignment.csv OR symlinks + index.jsonl -----
        # Build paths
        seg_dir = tmp_out / f"{seg_idx:03d}"
        chunk_dir = seg_dir / f"chunk_{chunk_global:04d}"

        if not args.dry_run and not args.plan_only:
            chunk_dir.mkdir(parents=True, exist_ok=True)

        order = 0
        for row, _key, _run_idx, _elem_idx in chunk_rows:
            if args.plan_only:
                # Plan only: write one CSV line per file
                if csv_w:
                    csv_w.writerow(
                        [
                            seg_idx,
                            chunk_global,
                            order,
                            str(row[0]),
                            row[1],
                            row[2],
                            row[3],
                            row[4],
                        ]
                    )
            else:
                # Build symlink + write JSONL record
                link = chunk_dir / f"{order:05d}__{row[0].name}"
                if not args.dry_run:
                    make_rel_symlink(row[0], link)
                    rec = {
                        "segment": seg_idx,
                        "chunk": chunk_global,
                        "order_in_chunk": order,
                        "link_path": str(link.relative_to(tmp_out)),
                        "src_path": str(row[0]),
                        "duration_sec": row[1],
                        "top_file": row[2],
                        "mid_file": row[3],
                        "sequence": row[4],
                    }
                    json_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            order += 1

        seg_sum += chunk_sum
        chunk_global += 1

        # Requeue this top if it still has remaining content
        if next_file_in_top(t) is not None:
            top_queue.append(t)

    # ---- Close writers ----
    if json_f:
        json_f.close()
    if csv_f:
        csv_f.close()

    # ---- Write manifest + build_config (for perfect reproducibility) ----
    manifest = {
        "csv": str(args.csv),
        "csv_sha256": csv_hash,
        "original_root": str(args.original_root.resolve()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version,
        "platform": sys.platform,
    }
    config = {
        "chunk_sec": args.chunk_sec,
        "chunk_eps": args.chunk_eps,
        "segment_sec": args.segment_sec,
        "segment_eps": args.segment_eps,
        "plan_only": args.plan_only,
    }

    if not args.dry_run:
        (tmp_out / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (tmp_out / "build_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        # Finalize atomically
        if args.out_root.exists():
            if args.fail_if_exists:
                print(f"[ERR] Output exists: {args.out_root}", file=sys.stderr)
                sys.exit(2)
            else:
                print(
                    f"[ERR] Output exists: {args.out_root}. Refusing to overwrite.",
                    file=sys.stderr,
                )
                sys.exit(2)
        tmp_out.rename(args.out_root)
    else:
        # Dry-run cleanup
        import shutil

        shutil.rmtree(tmp_out, ignore_errors=True)


if __name__ == "__main__":
    main()
