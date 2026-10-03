#!/usr/bin/env python3
"""Generate the ChurnGuard architecture diagram as a standalone SVG.

Embeds the official AWS architecture icons from ../../aws-icons (base64-inlined
so the output SVG is fully portable) and lays out the real resource topology
returned by the backend's GET /api/architecture route:

    dataset -> pipeline -> registry -> {events, bluegreen -> realtime}
    dataset -> {realtime, serverless, async, batch, mme}

Routing is ORTHOGONAL with rounded corners along dedicated channels so no edge
overlaps another edge or passes over an icon tile:
  * the top row (dataset -> pipeline -> registry -> events) is a clean chain;
  * registry -> bluegreen -> realtime drops down the right-hand channel;
  * dataset -> the five serving nodes drops into a horizontal "data bus" that
    runs in the clear band above the inference lane, with a vertical riser into
    the TOP edge of each endpoint tile (risers sit in the gaps between tiles).

Run:  python3 docs/generate_architecture_diagram.py
Output: docs/architecture.svg  (+ render to PNG with rsvg-convert)
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ICONS = REPO.parent / "aws-icons"
OUT = HERE / "architecture.svg"

SVC = ICONS / "Architecture-Service-Icons_04302026"

ICON_FILES = {
    "sagemaker": SVC / "Arch_Artificial-Intelligence/64/Arch_Amazon-SageMaker-AI_64.svg",
    "s3": SVC / "Arch_Storage/64/Arch_Amazon-Simple-Storage-Service_64.svg",
    "sns": SVC / "Arch_Application-Integration/64/Arch_Amazon-Simple-Notification-Service_64.svg",
    "eventbridge": SVC / "Arch_Application-Integration/64/Arch_Amazon-EventBridge_64.svg",
    "cloudwatch": SVC / "Arch_Management-Tools/64/Arch_Amazon-CloudWatch_64.svg",
    "lambda": SVC / "Arch_Compute/64/Arch_AWS-Lambda_64.svg",
}


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


ICON_URI = {k: data_uri(v) for k, v in ICON_FILES.items()}

# ---- canvas -------------------------------------------------------------
W, H = 1320, 860
ICON = 60
TILE = 92
R = 14                 # corner radius for orthogonal routing

# palette
SQUID = "#232F3E"
INK = "#16191F"
MUTE = "#5A6B7B"
ORANGE = "#ED7100"
TEAL = "#01A88D"
PURPLE = "#8C4FFF"
PINK = "#E7157B"
GREEN = "#2BB673"

# Row Y coordinates
TOP_Y = 210          # MLOps row
MID_Y = 450          # dataset + blue/green + realtime
BOT_Y = 700          # inference row
BUS_Y = 585          # horizontal data-bus between mid and inference rows

# node id -> (centre x, centre y, icon key, title, subtitle, tile colour)
NODES = {
    "dataset":    (150, MID_Y, "s3",          "Telco Churn Dataset",  "S3 data bucket",             "#7AA116"),
    "pipeline":   (470, TOP_Y, "sagemaker",   "SageMaker Pipeline",   "Process-Train-Eval-Register","#01A88D"),
    "registry":   (720, TOP_Y, "sagemaker",   "Model Registry",       "churnguard-churn",           "#01A88D"),
    "events":     (970, TOP_Y, "eventbridge", "EventBridge + Lambda", "model-approved rule",        "#E7157B"),
    "bluegreen":  (970, MID_Y, "cloudwatch",  "Blue/Green Deploy",    "canary / linear + rollback", "#E7157B"),
    "realtime":   (1200, MID_Y, "sagemaker",  "Real-time Endpoint",   "3x ml.m5.large",             "#01A88D"),
    "serverless": (470, BOT_Y, "sagemaker",   "Serverless Endpoint",  "scale-to-zero",              "#01A88D"),
    "async":      (660, BOT_Y, "sagemaker",   "Async Endpoint",       "queued, large payload",      "#01A88D"),
    "sns":        (850, BOT_Y, "sns",         "SNS",                  "success / error",            "#E7157B"),
    "batch":      (1040, BOT_Y, "sagemaker",  "Batch Transform",      "offline job",                "#01A88D"),
    "mme":        (1230, BOT_Y, "sagemaker",  "Multi-Model Endpoint", "churn-v1 / churn-v2",        "#01A88D"),
}

COLORS = {PURPLE: "pu", PINK: "pk", GREEN: "gr", TEAL: "te"}


def cx(nid: str) -> float: return NODES[nid][0]
def cy(nid: str) -> float: return NODES[nid][1]
def left(nid: str) -> float: return cx(nid) - TILE / 2
def right(nid: str) -> float: return cx(nid) + TILE / 2
def top(nid: str) -> float: return cy(nid) - TILE / 2
def bot(nid: str) -> float: return cy(nid) + TILE / 2


def rounded_path(pts: list[tuple[float, float]], color: str) -> str:
    """Build an orthogonal polyline through pts with rounded corners."""
    if len(pts) < 2:
        return ""
    d = [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"]
    for i in range(1, len(pts) - 1):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        # approach point before the corner
        def shorten(px, py, qx, qy):
            import math
            dx, dy = qx - px, qy - py
            dist = math.hypot(dx, dy) or 1
            r = min(R, dist / 2)
            return px + dx / dist * (dist - r), py + dy / dist * (dist - r), \
                px + dx / dist * r + px * 0, py + dy / dist * r
        import math
        # point entering corner (from x0,y0 -> x1,y1), stop short of x1,y1
        d01 = math.hypot(x1 - x0, y1 - y0) or 1
        r01 = min(R, d01 / 2)
        ex = x1 - (x1 - x0) / d01 * r01
        ey = y1 - (y1 - y0) / d01 * r01
        # point leaving corner (from x1,y1 -> x2,y2), start short after x1,y1
        d12 = math.hypot(x2 - x1, y2 - y1) or 1
        r12 = min(R, d12 / 2)
        lx = x1 + (x2 - x1) / d12 * r12
        ly = y1 + (y2 - y1) / d12 * r12
        d.append(f"L {ex:.1f} {ey:.1f}")
        d.append(f"Q {x1:.1f} {y1:.1f} {lx:.1f} {ly:.1f}")
    d.append(f"L {pts[-1][0]:.1f} {pts[-1][1]:.1f}")
    return (f'<path d="{" ".join(d)}" fill="none" stroke="{color}" '
            f'stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" '
            f'marker-end="url(#arrow-{COLORS[color]})" opacity="0.95"/>')


def build_edges() -> list[str]:
    paths: list[str] = []
    GAP = 8  # arrow standoff from tile edge

    # --- top chain: pipeline -> registry -> events (straight, same row) ---
    for a, b in [("pipeline", "registry"), ("registry", "events")]:
        paths.append(rounded_path(
            [(right(a) + GAP, cy(a)), (left(b) - GAP, cy(b))], PURPLE if a == "pipeline" else PINK))

    # dataset -> pipeline : up the left channel then right into pipeline left
    chx = (right("dataset") + left("pipeline")) / 2 - 40
    paths.append(rounded_path([
        (right("dataset"), cy("dataset") - 20),
        (chx, cy("dataset") - 20),
        (chx, cy("pipeline")),
        (left("pipeline") - GAP, cy("pipeline")),
    ], PURPLE))

    # registry -> bluegreen : exit the registry's RIGHT side, run through the
    # clear channel in the gap between the registry and events tiles, then drop
    # straight down into the top of bluegreen. Keeps the line off the registry
    # label and clear of every other edge.
    chan_x = (right("registry") + left("events")) / 2
    mid_y = (bot("registry") + top("bluegreen")) / 2
    paths.append(rounded_path([
        (right("registry") + GAP, cy("registry")),
        (chan_x, cy("registry")),
        (chan_x, mid_y),
        (cx("bluegreen"), mid_y),
        (cx("bluegreen"), top("bluegreen") - GAP),
    ], PINK))

    # bluegreen -> realtime : straight across the mid row
    paths.append(rounded_path([
        (right("bluegreen") + GAP, cy("bluegreen")),
        (left("realtime") - GAP, cy("realtime")),
    ], GREEN))

    # dataset -> realtime : up and over the top of everything via a high arc
    hy = TOP_Y - TILE / 2 - 55
    paths.append(rounded_path([
        (cx("dataset"), top("dataset") - GAP),
        (cx("dataset"), hy),
        (cx("realtime"), hy),
        (cx("realtime"), top("realtime") - GAP),
    ], TEAL))

    # dataset -> five bottom nodes via a horizontal data bus at BUS_Y.
    # Each riser enters the TOP of its endpoint; the dataset drops from its
    # BOTTOM to the bus. Risers sit at each node's x (gaps between tiles).
    bottoms = ["serverless", "async", "batch", "mme"]
    # dataset trunk down to bus
    dx_trunk = cx("dataset")
    paths.append(rounded_path([
        (dx_trunk, bot("dataset") + GAP),
        (dx_trunk, BUS_Y),
        (cx("serverless"), BUS_Y),
        (cx("serverless"), top("serverless") - GAP),
    ], TEAL))
    # branches off the bus to async, batch, mme (bus already drawn to serverless x;
    # extend bus segments and drop risers)
    bus_targets = ["async", "batch", "mme"]
    prev_x = cx("serverless")
    for t in bus_targets:
        paths.append(rounded_path([
            (prev_x, BUS_Y),
            (cx(t), BUS_Y),
            (cx(t), top(t) - GAP),
        ], TEAL))
        prev_x = cx(t)

    # async -> sns : straight across bottom row
    paths.append(rounded_path([
        (right("async") + GAP, cy("async")),
        (left("sns") - GAP, cy("sns")),
    ], PINK))

    return paths


def tile_svg(nid: str) -> str:
    x, y, key, title, sub, col = NODES[nid]
    tx, ty = x - TILE / 2, y - TILE / 2
    ix, iy = x - ICON / 2, y - ICON / 2
    label_y = y + TILE / 2 + 20
    sub_y = label_y + 15
    return f'''  <g filter="url(#shadow)">
    <rect x="{tx:.0f}" y="{ty:.0f}" width="{TILE}" height="{TILE}" rx="18"
          fill="#FFFFFF" stroke="{col}" stroke-width="2.5"/>
    <rect x="{tx:.0f}" y="{ty:.0f}" width="{TILE}" height="8" rx="4" fill="{col}"/>
    <image href="{ICON_URI[key]}" x="{ix:.0f}" y="{iy:.0f}" width="{ICON}" height="{ICON}"/>
  </g>
  <text x="{x:.0f}" y="{label_y:.0f}" text-anchor="middle" font-size="13.5"
        font-weight="700" fill="{INK}">{html.escape(title)}</text>
  <text x="{x:.0f}" y="{sub_y:.0f}" text-anchor="middle" font-size="11"
        fill="{MUTE}">{html.escape(sub)}</text>'''


def main() -> None:
    p: list[str] = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="Amazon Ember, Segoe UI, Arial, sans-serif">'
    )

    arrow_markers = "".join(
        f'''<marker id="arrow-{k}" markerWidth="9" markerHeight="9" refX="6.5" refY="3.2"
              orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6.4 L7,3.2 z" fill="{col}"/>
        </marker>'''
        for col, k in COLORS.items()
    )
    p.append(f'''  <defs>
    {arrow_markers}
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#FBFCFE"/>
      <stop offset="1" stop-color="#EEF3F8"/>
    </linearGradient>
    <linearGradient id="titlebar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#232F3E"/>
      <stop offset="1" stop-color="#394a60"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
      <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#1B2A3A" flood-opacity="0.18"/>
    </filter>
  </defs>''')

    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="url(#bg)"/>')
    p.append(f'<rect x="0" y="0" width="{W}" height="64" fill="url(#titlebar)"/>')
    p.append(f'<rect x="0" y="64" width="{W}" height="4" fill="{ORANGE}"/>')
    p.append(
        f'<text x="36" y="40" font-size="22" font-weight="700" fill="#FFFFFF">ChurnGuard</text>'
        f'<text x="166" y="40" font-size="16" fill="#CBD5E1">Deployment &amp; Orchestration of ML Workflows</text>'
    )
    p.append(
        f'<text x="{W-36:.0f}" y="40" text-anchor="end" font-size="12.5" fill="#9FB2C6">'
        f'Account 875692608981  -  us-east-1</text>'
    )

    # swim-lane bands
    p.append(f'<rect x="400" y="{TOP_Y-TILE/2-24:.0f}" width="{W-440}" height="{TILE+60}" rx="16" '
             f'fill="#8C4FFF" opacity="0.05"/>')
    p.append(f'<rect x="400" y="{BOT_Y-TILE/2-18:.0f}" width="{W-440}" height="{TILE+54}" rx="16" '
             f'fill="#01A88D" opacity="0.06"/>')
    p.append(f'<text x="420" y="{TOP_Y-TILE/2-6:.0f}" font-size="12" font-weight="700" '
             f'letter-spacing="1" fill="{PURPLE}">MLOPS / CI-CD LANE</text>')
    p.append(f'<text x="420" y="{BOT_Y-TILE/2-1:.0f}" font-size="12" font-weight="700" '
             f'letter-spacing="1" fill="{TEAL}">INFERENCE &amp; HOSTING LANE</text>')

    # edges under tiles
    for path in build_edges():
        p.append(path)

    # tiles
    for nid in NODES:
        p.append(tile_svg(nid))

    # legend
    legend = [
        (PURPLE, "Build pipeline"),
        (PINK, "Approve -> event / notify"),
        (GREEN, "Blue/green promotion"),
        (TEAL, "Model served from data"),
    ]
    lx = 36
    ly = H - 28
    p.append(f'<text x="{lx}" y="{ly-24:.0f}" font-size="11.5" font-weight="700" fill="{SQUID}">Flow legend</text>')
    for col, label in legend:
        p.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="{col}" '
                 f'stroke-width="3" stroke-linecap="round"/>')
        p.append(f'<text x="{lx+34}" y="{ly+4:.0f}" font-size="11" fill="{MUTE}">{html.escape(label)}</text>')
        lx += 34 + 9 * len(label) + 28

    p.append('</svg>')
    OUT.write_text("\n".join(p), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
