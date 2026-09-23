"""A single self contained HTML page for a report.

The centrepiece is the segment map. Every segment torch holds is drawn to scale as a strip of
live blocks and holes, which is the one thing a wall of text cannot show: a 4 MiB block sitting
in the middle of half a gigabyte of free space is obvious the moment you see it.

No scripts from the network and no images, because this page gets attached to bug reports and
opened on machines that are not the one that produced it.
"""

from __future__ import annotations

import html

from .accounting import Accounting
from .frag import Explanation
from .snapshot import Segment, fmt_bytes

# a block thinner than this would be invisible, so neighbours get merged into one sliver
MIN_BLOCK_FRACTION = 0.004
MAX_SEGMENTS_DRAWN = 40

CSS = """
:root {
  --bg: #fbfbfa; --fg: #1a1a18; --muted: #6b6b66; --rule: #e0e0dc;
  --live: #3d6ea8; --live-pin: #c2410c; --hole: #d9d9d4; --hole-big: #b8ccb0;
  --card: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17171a; --fg: #e8e8e4; --muted: #8f8f88; --rule: #2e2e33;
    --live: #5b8fc9; --live-pin: #f97316; --hole: #26262b; --hole-big: #38513a;
    --card: #1e1e22;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 32px 16px 64px; background: var(--bg); color: var(--fg);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
main { max-width: 900px; margin: 0 auto; }
h1 { font-size: 19px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 13px; margin: 32px 0 10px; font-weight: 600; letter-spacing: .04em;
     text-transform: uppercase; color: var(--muted); }
p { margin: 0 0 10px; }
.sub { color: var(--muted); margin-bottom: 24px; }
.verdict { border-left: 3px solid var(--live-pin); padding: 2px 0 2px 14px; margin: 0 0 8px; }
.verdict b { text-transform: uppercase; letter-spacing: .05em; font-size: 12px; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
table { border-collapse: collapse; width: 100%; }
td { padding: 3px 8px 3px 0; vertical-align: baseline; }
td.n { text-align: right; font-family: ui-monospace, Menlo, monospace; white-space: nowrap; }
td.g { width: 40%; }
.bar { height: 9px; border-radius: 2px; background: var(--live); }
.seg { margin-bottom: 14px; }
.wrap { position: relative; }
.mark { position: absolute; top: -4px; transform: translateX(-50%); color: var(--live-pin);
        font-size: 10px; line-height: 1; white-space: nowrap; pointer-events: none; }
.mark b { display: block; text-align: center; font-weight: 600; }
.marks { height: 15px; }
.seg .cap { display: flex; justify-content: space-between; color: var(--muted);
            font-size: 11.5px; margin-bottom: 2px; }
.strip { display: flex; height: 22px; border-radius: 3px; overflow: hidden;
         border: 1px solid var(--rule); }
.blk { height: 100%; }
.blk.live { background: var(--live); }
.blk.pin  { background: var(--live-pin); }
.blk.free { background: var(--hole); }
.blk.bigfree { background: var(--hole-big); }
.key { display: flex; gap: 18px; color: var(--muted); font-size: 12px; margin: 10px 0 2px; }
.key i { display: inline-block; width: 10px; height: 10px; border-radius: 2px;
         margin-right: 5px; vertical-align: -1px; }
.note { color: var(--muted); font-size: 12.5px; }
.stack { border: 1px solid var(--rule); background: var(--card); border-radius: 6px;
         padding: 12px 14px; }
svg { display: block; width: 100%; height: auto; }
.tip { position: fixed; pointer-events: none; background: var(--fg); color: var(--bg);
       padding: 4px 8px; border-radius: 4px; font-size: 12px; opacity: 0; transition: opacity .1s;
       font-family: ui-monospace, Menlo, monospace; z-index: 9; white-space: pre; }
"""

JS = """
const tip = document.createElement('div');
tip.className = 'tip';
document.body.appendChild(tip);
document.addEventListener('mousemove', (e) => {
  const el = e.target.closest('[data-tip]');
  if (!el) { tip.style.opacity = 0; return; }
  tip.textContent = el.dataset.tip;
  tip.style.opacity = 1;
  const pad = 14;
  let x = e.clientX + pad, y = e.clientY + pad;
  const box = tip.getBoundingClientRect();
  if (x + box.width > innerWidth) x = e.clientX - box.width - pad;
  if (y + box.height > innerHeight) y = e.clientY - box.height - pad;
  tip.style.left = x + 'px';
  tip.style.top = y + 'px';
});
"""


def _e(text: object) -> str:
    return html.escape(str(text))


def _site_of(block) -> str:
    from .snapshot import user_frames

    frames = user_frames(block.frames)
    if not frames:
        return "no stack"
    f = frames[0]
    return f"{f.filename.rsplit('/', 1)[-1]}:{f.line} {f.name}"


def _strip(seg: Segment, pins: set[int], request: int | None) -> str:
    """One segment drawn to scale, with the blocks that matter kept visible."""
    total = max(seg.total_size, 1)
    pieces: list[tuple[str, int, str]] = []  # (kind, bytes, tooltip)
    for b in seg.blocks:
        if b.live:
            kind = "pin" if b.address in pins else "live"
            tip = f"{fmt_bytes(b.size)} live\n{_site_of(b)}"
        else:
            big = request is not None and b.size >= request
            kind = "bigfree" if big else "free"
            tip = f"{fmt_bytes(b.size)} free" + (" (fits the request)" if big else "")
        pieces.append((kind, b.size, tip))

    # merge runs of slivers so the browser is not asked to draw a thousand hairlines
    merged: list[tuple[str, int, str]] = []
    for kind, size, tip in pieces:
        thin = size / total < MIN_BLOCK_FRACTION
        if thin and merged and merged[-1][1] / total < MIN_BLOCK_FRACTION:
            prev = merged[-1]
            merged[-1] = (prev[0], prev[1] + size, f"{fmt_bytes(prev[1] + size)} of small blocks")
        else:
            merged.append((kind, size, tip))

    out = [
        f'<div class="blk {k}" style="width:{100 * n / total:.4f}%" data-tip="{_e(t)}"></div>'
        for k, n, t in merged
    ]
    tag = " expandable" if seg.is_expandable else ""
    head = (
        f'<div class="cap"><span>{fmt_bytes(seg.total_size)} segment{tag} '
        f"at 0x{seg.address:x}</span><span>{fmt_bytes(seg.free)} free</span></div>"
    )

    # a block holding a big hole open can be a hairline at this scale, so point at it instead
    # of widening it, which would make the picture lie about where the memory is
    def mark(b) -> str:
        middle = 100 * (b.address - seg.address + b.size / 2) / total
        return f'<span class="mark" style="left:{middle:.3f}%"><b>▾</b>{fmt_bytes(b.size)}</span>'

    marks = "".join(mark(b) for b in seg.blocks if b.address in pins)
    marks = f'<div class="marks"><div class="wrap">{marks}</div></div>' if marks else ""
    return f'<div class="seg">{head}{marks}<div class="strip">{"".join(out)}</div></div>'


def _device_bar(acc: Accounting) -> str:
    """The whole card as one bar, so the memory torch cannot see is impossible to miss."""
    if acc.nvml is None or not acc.nvml.total:
        return ""
    total = acc.nvml.total
    parts = [("torch reserved", acc.torch_reserved, "var(--live)")]
    for name, size in sorted(acc.libs.items(), key=lambda kv: -kv[1])[:4]:
        parts.append((name, size, "var(--live-pin)"))
    if acc.kernel_images:
        parts.append(("kernel images", acc.kernel_images, "var(--live-pin)"))
    if acc.context_known:
        parts.append(("CUDA context", acc.baseline, "var(--muted)"))
    if acc.unattributed:
        parts.append(("outside torch", acc.unattributed, "var(--muted)"))
    used = sum(p[1] for p in parts)
    parts.append(("free", max(total - used, 0), "var(--hole)"))

    cells = "".join(
        f'<div class="blk" style="width:{100 * n / total:.4f}%;background:{c};'
        f'opacity:{0.55 if c == "var(--muted)" else 1}" '
        f'data-tip="{_e(name)}: {fmt_bytes(n)}"></div>'
        for name, n, c in parts
        if n > 0
    )
    rows = "".join(
        f'<tr><td>{_e(name)}</td><td class="n">{fmt_bytes(n)}</td></tr>'
        for name, n, _ in parts
        if n > 0
    )
    return (
        f'<div class="cap"><span>{fmt_bytes(acc.nvml.used)} used</span>'
        f"<span>{fmt_bytes(total)} on the card</span></div>"
        f'<div class="strip">{cells}</div><table style="margin-top:10px">{rows}</table>'
    )


def _timeline(rows: list) -> str:
    """reserved, live and the largest free block over the run."""
    rows = [r for r in rows if r.reserved]
    if len(rows) < 3:
        return ""
    w, h, pad = 880, 160, 6
    top = max(max(r.reserved for r in rows), 1)
    t0, t1 = rows[0].t, max(rows[-1].t, rows[0].t + 1e-6)

    def path(get, only_known=False):
        pts = []
        for r in rows:
            v = get(r)
            if only_known and v < 0:
                continue
            x = pad + (w - 2 * pad) * (r.t - t0) / (t1 - t0)
            y = h - pad - (h - 2 * pad) * min(v / top, 1)
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    layers = [
        ("reserved", path(lambda r: r.reserved), "var(--live)", 0.18),
        ("live", path(lambda r: r.live), "var(--live-pin)", 0.0),
        ("largest free block", path(lambda r: r.largest_free, True), "var(--hole-big)", 0.0),
    ]
    svg = [f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" role="img">']
    for name, pts, colour, fill in layers:
        if not pts:
            continue
        if fill:
            svg.append(
                f'<polygon points="{pad},{h - pad} {pts} {w - pad},{h - pad}" '
                f'fill="{colour}" opacity="{fill}"/>'
            )
        svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="1.6" '
            f'vector-effect="non-scaling-stroke"><title>{_e(name)}</title></polyline>'
        )
    svg.append("</svg>")
    key = "".join(
        f'<span><i style="background:{c}"></i>{_e(n)}</span>' for n, p, c, _f in layers if p
    )
    return (
        f'<div class="cap"><span>0 s</span><span>peak {fmt_bytes(top)}</span>'
        f"<span>{t1 - t0:.0f} s</span></div>{''.join(svg)}"
        f'<div class="key">{key}</div>'
    )


def _sites(ex: Explanation) -> str:
    if not ex.sites:
        return ""
    top = max((s.bytes for s in ex.sites), default=1)
    rows = "".join(
        f'<tr><td class="n">{fmt_bytes(s.bytes)}</td>'
        f'<td class="g"><div class="bar" style="width:{100 * s.bytes / top:.1f}%"></div></td>'
        f'<td class="mono">{_e(s.where)}</td>'
        f'<td class="n note">{s.count} blocks</td></tr>'
        for s in ex.sites
    )
    return f"<table>{rows}</table>"


def render(
    ex: Explanation,
    acc: Accounting | None = None,
    rows: list | None = None,
    suggestions: list | None = None,
    title: str = "vramxray",
) -> str:
    pins = {p.block.address for p in ex.pins[:6]}
    segs = sorted(ex.segments, key=lambda s: -s.total_size)
    drawn = segs[:MAX_SEGMENTS_DRAWN]
    more = (
        f'<p class="note">showing the {len(drawn)} largest of {len(segs)} segments</p>'
        if len(segs) > len(drawn)
        else ""
    )

    head = f"cuda:{ex.device}"
    if ex.request is not None:
        head += f" ran out of memory asking for {fmt_bytes(ex.request)}"
    else:
        head += " memory report"

    blocks = [f"<h1>{_e(head)}</h1>"]
    blocks.append(
        f'<p class="sub mono">reserved {fmt_bytes(ex.reserved)} · live {fmt_bytes(ex.live)} · '
        f"largest free block {fmt_bytes(ex.largest_hole)}</p>"
    )
    blocks.append(f'<p class="verdict"><b>{_e(ex.verdict)}</b><br>{_e(ex.verdict_text)}</p>')
    if suggestions:
        items = "".join(f"<li>{_e(str(s))}</li>" for s in suggestions)
        blocks.append(f'<h2>What to try</h2><ul class="stack">{items}</ul>')
    if acc is not None:
        blocks.append(f"<h2>The whole card</h2>{_device_bar(acc)}")
    if rows:
        tl = _timeline(rows)
        if tl:
            blocks.append(f"<h2>Over the run</h2>{tl}")
    blocks.append("<h2>Segment map</h2>")
    blocks.append(
        '<div class="key">'
        '<span><i style="background:var(--live)"></i>live</span>'
        '<span><i style="background:var(--live-pin)"></i>holding a hole open</span>'
        '<span><i style="background:var(--hole)"></i>free</span>'
        '<span><i style="background:var(--hole-big)"></i>free and big enough</span></div>'
    )
    blocks.append("".join(_strip(s, pins, ex.request) for s in drawn) + more)
    sites = _sites(ex)
    if sites:
        blocks.append(f"<h2>Live memory by call site</h2>{sites}")

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_e(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{''.join(blocks)}</main><script>{JS}</script></body></html>"
    )


def write(path: str, **kw) -> str:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render(**kw))
    return path
