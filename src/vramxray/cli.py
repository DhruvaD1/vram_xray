from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable

from . import __version__
from .frag import Explanation, explain, if_freed
from .snapshot import Frame, Snapshot, fmt_bytes, load, user_frames


def _stack(frames: Iterable[Frame], n: int = 3) -> str:
    # drop torch's own frames so the user's line comes first
    keep = user_frames(frames) or list(frames)
    if not keep:
        return "(no stack, enable record_memory_history)"
    parts = [f"{f.filename.rsplit('/', 1)[-1]}:{f.line} {f.name}" for f in keep[:n]]
    return " from ".join(parts)


def _size_arg(text: str) -> int:
    # accepts 1.5G, 512M, 4K, or plain bytes
    unit = {"g": 1 << 30, "m": 1 << 20, "k": 1 << 10}.get(text[-1].lower())
    return int(float(text[:-1]) * unit) if unit else int(text)


def render(snap: Snapshot, ex: Explanation, top: int) -> str:
    b = fmt_bytes
    d = ex.device
    lines = [
        f"vramxray analyze: cuda:{d}",
        f"  reserved {b(ex.reserved)} · live {b(ex.live)} · free in segments "
        f"{b(ex.reserved - ex.live)} · segments {len(snap.for_device(d))} "
        f"· trace entries {len(snap.traces.get(d, []))}",
    ]
    if ex.request is not None:
        driver = (
            f" · driver reported {b(ex.device_free)} free" if ex.device_free is not None else ""
        )
        lines.append(f"  request {b(ex.request)} on stream {ex.stream}{driver}")
    lines += [
        f"  usable for this request: {b(ex.free_in_segments)} free · largest hole "
        f"{b(ex.largest_hole)} · trapped {b(ex.trapped)}",
        "",
        f"  verdict: {ex.verdict}",
        f"  {ex.verdict_text}",
        "",
    ]
    if ex.holes:
        lines.append(f"  holes (top {min(top, len(ex.holes))} of {len(ex.holes)}):")
        for h in ex.holes[:top]:
            seg = h.segment
            lo = f"{b(h.left.size)} live below" if h.left else "segment start"
            hi = f"{b(h.right.size)} live above" if h.right else "segment end"
            tag = " (expandable)" if seg.is_expandable else ""
            lines.append(
                f"    {b(h.size):>10}  in {b(seg.total_size)} segment 0x{seg.address:x}{tag}"
                f"  [{lo} | {hi}]"
            )
        lines.append("")
    if ex.pins:
        lines.append(
            f"  pinning blocks (top {min(top, len(ex.pins))} of {len(ex.pins)}), "
            "ranked by wasted-bytes per byte held:"
        )
        for p in ex.pins[:top]:
            age = f"age {p.age} allocs" if p.age is not None else "age ?"
            lines.append(
                f"    {b(p.block.size):>10}  holds {b(p.wasted):>10} of holes  {age:>16}"
                f"  {_stack(p.frames)}"
            )
        lines.append("")
    if ex.verdict == "fragmentation":
        lines.append("  what would help:")
        fits_if_merged = ex.request is not None and ex.free_in_segments >= ex.request
        if fits_if_merged and ex.expandable_recoverable and not ex.expandable_on:
            lines.append(
                f"    {ex.alloc_conf_var}=expandable_segments:True   free pages from different "
                f"holes can back one block: {b(ex.free_in_segments)} becomes usable as a unit "
                f"instead of {b(ex.largest_hole)}"
            )
        for k in (1, 2, 3):
            cand = ex.pins[:k]
            if not cand:
                break
            new_largest = if_freed(ex.holes, [p.block for p in cand])
            if ex.request is not None and new_largest >= ex.request:
                plural = "s" if k > 1 else ""
                lines.append(
                    f"    freeing or relocating the top {k} pinning block{plural} above would "
                    f"give a {b(new_largest)} hole, enough for this request"
                )
                break
    return "\n".join(lines)


def to_json(ex: Explanation) -> dict:
    return {
        "device": ex.device,
        "request": ex.request,
        "stream": ex.stream,
        "device_free": ex.device_free,
        "reserved": ex.reserved,
        "live": ex.live,
        "free_in_segments": ex.free_in_segments,
        "largest_hole": ex.largest_hole,
        "trapped": ex.trapped,
        "verdict": ex.verdict,
        "verdict_text": ex.verdict_text,
        "expandable_recoverable": ex.expandable_recoverable,
        "cap": ex.cap,
        "alloc_conf_var": ex.alloc_conf_var,
        "expandable_on": ex.expandable_on,
        "sites": [{"where": s.where, "bytes": s.bytes, "count": s.count} for s in ex.sites],
        "holes": [
            {
                "segment": hex(h.segment.address),
                "address": hex(h.address),
                "size": h.size,
                "left": h.left.size if h.left else None,
                "right": h.right.size if h.right else None,
            }
            for h in ex.holes
        ],
        "pins": [
            {
                "address": hex(p.block.address),
                "size": p.block.size,
                "wasted": p.wasted,
                "age": p.age,
                "frames": [str(f) for f in p.frames],
            }
            for p in ex.pins
        ],
    }


def cmd_analyze(a: argparse.Namespace) -> int:
    snap = load(a.snapshot)
    out = {}
    devices = [a.device] if a.device is not None else snap.devices()
    for d in devices:
        ex = explain(snap, d, request=a.request)
        print(render(snap, ex, a.top))
        out[d] = to_json(ex)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.json}")
    return 0


def cmd_merge(a: argparse.Namespace) -> int:
    """One table across the JSON reports of a multi-process job."""
    from .snapshot import fmt_bytes as b

    rows = []
    for path in a.reports:
        with open(path) as f:
            d = json.load(f)
        acc = d.get("accounting") or {}
        libs = acc.get("libs") or {}
        nccl = sum(v for k, v in libs.items() if "nccl" in k.lower())
        rows.append(
            {
                "rank": d.get("rank"),
                "file": path,
                "request": d.get("request"),
                "verdict": (d.get("explanation") or {}).get("verdict"),
                "torch": acc.get("torch_reserved") or 0,
                "nccl": nccl,
                "libs": sum(libs.values()) - nccl,
                "other": acc.get("unattributed") or 0,
            }
        )
    rows.sort(key=lambda r: (r["rank"] is None, r["rank"]))
    print(
        f"{'rank':>4}  {'request':>10}  {'verdict':<13} {'torch':>10}  {'nccl':>10}  "
        f"{'other libs':>10}  {'outside':>10}"
    )
    for r in rows:
        rank = "?" if r["rank"] is None else r["rank"]
        print(
            f"{rank:>4}  {b(r['request']):>10}  {(r['verdict'] or '-'):<13} {b(r['torch']):>10}  "
            f"{b(r['nccl']):>10}  {b(r['libs']):>10}  {b(r['other']):>10}"
        )
    nccls = [r["nccl"] for r in rows if r["nccl"]]
    if len(nccls) > 1:
        med = sorted(nccls)[len(nccls) // 2]
        skew = [r for r in rows if med and r["nccl"] > 1.5 * med]
        if skew:
            ranks = ", ".join(str(r["rank"]) for r in skew)
            print(f"\nNCCL skew: rank(s) {ranks} hold more than 1.5x the median ({b(med)})")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    import runpy

    import vramxray

    vramxray.watch(stacks=a.stacks, report_dir=a.report_dir)
    sys.argv = [a.script, *a.args]
    sys.path.insert(0, str(__import__("pathlib").Path(a.script).resolve().parent))
    runpy.run_path(a.script, run_name="__main__")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="vramxray", description="See the GPU memory PyTorch can't show you."
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    an = sub.add_parser("analyze", help="explain a torch.cuda.memory snapshot pickle")
    an.add_argument("snapshot")
    an.add_argument("--device", type=int)
    an.add_argument(
        "--request",
        type=_size_arg,
        help="hypothetical request size, e.g. 1.5G or 512M (default: last OOM in the trace)",
    )
    an.add_argument("--top", type=int, default=8)
    an.add_argument("--json")
    an.set_defaults(fn=cmd_analyze)
    rn = sub.add_parser("run", help="run a script with vramxray.watch() already on")
    rn.add_argument("script")
    rn.add_argument("args", nargs=argparse.REMAINDER)
    rn.add_argument("--stacks", choices=["python", "all"], help="record allocation stacks")
    rn.add_argument("--report-dir")
    rn.set_defaults(fn=cmd_run)
    mg = sub.add_parser("merge", help="one table across per-rank OOM report JSON files")
    mg.add_argument("reports", nargs="+")
    mg.set_defaults(fn=cmd_merge)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
