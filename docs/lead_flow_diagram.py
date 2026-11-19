"""Generate docs/lead-flow.svg, the animated lead-flow diagram in the README.

Run from the repo root:  .venv/bin/python docs/lead_flow_diagram.py

The SVG animates with SMIL, which GitHub plays inside README images. A lead
moves through the stages, the steps it passes light up, and a caption narrates
each stop. Scenarios play one after another and loop.

To mark a planned stage as built, move it out of PLANNED. To change what is
shown, edit NODES, EDGES, or SCENARIOS.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

OUTPUT = Path(__file__).resolve().parent / "lead-flow.svg"

WIDTH, HEIGHT = 1130, 560
SPEED = 200  # pixels per second
DWELL = 3.0  # seconds a lead pauses at each stop
FINAL_HOLD = 4.0
GAP = 1.0  # blank pause between scenarios

Point = Tuple[float, float]

# name: (center x, center y, width, height, title, subtitle)
NODES: Dict[str, Tuple[float, float, float, float, str, str]] = {
    "form": (85, 130, 124, 56, "Web form", "fields and note"),
    "schema": (250, 130, 124, 56, "Schema check", "POST /leads"),
    "screen": (415, 130, 124, 56, "Screening", "spam, noise, ICP"),
    "dup": (580, 130, 124, 56, "Same email?", "last 30 days"),
    "acct": (745, 130, 124, 56, "Existing account?", "email domain"),
    "filter": (905, 130, 124, 56, "Rep filter", "territory, hours, cap"),
    "buyer": (905, 350, 154, 56, "Buyer agent", "note, TheirStack, PostHog"),
    "capacity": (905, 450, 154, 56, "Capacity service", "pipeline, backlog, fit"),
    "routing": (720, 400, 124, 56, "Routing agent", "primary and backup"),
    "commit": (545, 400, 124, 56, "Validate, commit", "reps on shortlist"),
    "handoff": (370, 400, 124, 56, "Rep handoff", "brief and history"),
}

# name: (center x, center y, width, label)
STATUSES: Dict[str, Tuple[float, float, float, str]] = {
    "invalid": (250, 45, 104, "422 errors"),
    "dropped": (362, 225, 84, "dropped"),
    "self_serve": (472, 225, 92, "self_serve"),
    "duplicate": (580, 45, 96, "duplicate"),
    "preserve_owner": (688, 225, 118, "preserve_owner"),
    "ownership_exception": (835, 225, 150, "ownership_exception"),
    "no_eligible_reps": (905, 45, 132, "no_eligible_reps"),
    "new_prospect": (1050, 130, 116, "new_prospect"),
}

PLANNED = {"commit", "handoff"}

STATUS_HEIGHT = 30

# (from, to): waypoints between the two centers
EDGES: Dict[Tuple[str, str], List[Point]] = {
    ("form", "schema"): [],
    ("schema", "screen"): [],
    ("screen", "dup"): [],
    ("dup", "acct"): [],
    ("acct", "filter"): [],
    ("filter", "new_prospect"): [],
    ("schema", "invalid"): [],
    ("screen", "dropped"): [(415, 185), (362, 185)],
    ("screen", "self_serve"): [(415, 185), (472, 185)],
    ("dup", "duplicate"): [],
    ("acct", "preserve_owner"): [(745, 185), (688, 185)],
    ("acct", "ownership_exception"): [(745, 185), (835, 185)],
    ("filter", "no_eligible_reps"): [],
    ("new_prospect", "buyer"): [(1050, 350)],
    ("new_prospect", "capacity"): [(1050, 450)],
    ("buyer", "routing"): [(800, 350), (800, 400)],
    ("capacity", "routing"): [(800, 450), (800, 400)],
    ("routing", "commit"): [],
    ("commit", "handoff"): [],
}

# Each scenario: token label, caption prefix, stops in order with the caption
# shown while the lead is there, and an optional parallel branch that leaves
# with the lead from one stop and rejoins it at a later one.
SCENARIOS = [
    {
        "token": "Elena",
        "who": "Elena Rostova, FinPay (Spain)",
        "stops": [
            ("form", "Submits the form: Intercom contract ends in 2 weeks, needs 40 seats before Q1"),
            ("schema", "Schema valid: required fields, work email, phone, company size"),
            ("screen", "Passes screening: not a test, competitor, or non-ICP submission"),
            ("dup", "No earlier submission from elena@finpay.es"),
            ("acct", "No existing account for finpay.es, so this is a new prospect"),
            ("filter", "5 eligible reps in EMEA-Southern: Mid-Market, Spanish, active, under cap, in hours"),
            ("new_prospect", "The shortlist goes to the buyer agent and the capacity service at once"),
            ("buyer", "In parallel: intent 98, High (workbook: 95); capacity rates rep_003 High bandwidth, "
                      "strong vs Intercom"),
            ("routing", "The routing agent proposes rep_003, backup rep_027: High bandwidth, FinTech, strong vs Intercom"),
            ("commit", "Both reps are checked against the shortlist, then the assignment is committed"),
            ("handoff", "The rep gets the buyer brief, and the decision is saved"),
        ],
        "branch": {"from": "new_prospect", "via": "capacity", "to": "routing", "sync": "buyer"},
    },
    {
        "token": "Test Bot",
        "who": "Test Bot, qa-automation-bot@test-domain-internal.org",
        "stops": [
            ("form", "An automated staging test submits the form"),
            ("schema", "Schema valid"),
            ("screen", "Screening finds a test or bot address"),
            ("dropped", "Dropped before any agent or rep sees it"),
        ],
    },
    {
        "token": "Alex",
        "who": "Alex Student, gmail.com, 1-15 people",
        "stops": [
            ("form", "Asks about a chat widget for a college side project"),
            ("schema", "Schema valid"),
            ("screen", "Screening: company size 1-15 is below the sales-assisted segment"),
            ("self_serve", "Sent to self-serve"),
        ],
    },
    {
        "token": "Priya",
        "who": "Priya Shah, HarborPay (mock account)",
        "stops": [
            ("form", "Asks to buy more licenses"),
            ("schema", "Schema valid"),
            ("screen", "Passes screening"),
            ("dup", "No earlier submission from this email"),
            ("acct", "harborpay.com is an existing customer owned by Mark Rivera (rep_002)"),
            ("preserve_owner", "Owner kept and a handoff brief generated, with no agent calls"),
        ],
    },
    {
        "token": "Sam",
        "who": "Sam Lee, Atlas Group (mock account)",
        "stops": [
            ("form", "Asks about an enterprise compliance plan"),
            ("schema", "Schema valid"),
            ("screen", "Passes screening"),
            ("dup", "No earlier submission from this email"),
            ("acct", "atlasgroup.com has two accounts with different owners: rep_009 and rep_038"),
            ("ownership_exception", "Sent to RevOps as an ownership exception, listing both owners"),
        ],
    },
]

STYLE = """
  .bg { fill: #ffffff; }
  .title { fill: #1f2328; font: 600 15px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .lane { fill: #59636e; font: 500 11px system-ui, -apple-system, "Segoe UI", sans-serif; letter-spacing: .04em; }
  .node { fill: #dafbe1; stroke: #1a7f37; stroke-width: 1.5; }
  .node.planned { fill: #f6f8fa; stroke: #8c959f; stroke-dasharray: 5 4; }
  .ntitle { fill: #1f2328; font: 600 13px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .nsub { fill: #59636e; font: 400 11px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .status { fill: #fff8c5; stroke: #9a6700; stroke-width: 1.2; }
  .stext { fill: #6c4a00; font: 500 11.5px ui-monospace, SFMono-Regular, Menlo, monospace; }
  .edge { fill: none; stroke: #8c959f; stroke-width: 1.5; }
  .arrow { fill: #8c959f; }
  .lit-edge { fill: none; stroke: #0969da; stroke-width: 3.5; stroke-linecap: round; stroke-linejoin: round; }
  .lit-node { fill: none; stroke: #0969da; stroke-width: 2.5; stroke-opacity: .5; }
  .act-node { fill: #0969da; fill-opacity: .14; stroke: #0969da; stroke-width: 3.5; }
  .token { fill: #0969da; stroke: #ffffff; stroke-width: 2; }
  .tlabel { fill: #0969da; font: 600 12px system-ui, -apple-system, "Segoe UI", sans-serif;
            paint-order: stroke; stroke: #ffffff; stroke-width: 4px; stroke-linejoin: round; }
  .cbox { fill: #f6f8fa; stroke: #d1d9e0; }
  .cwho { fill: #0969da; font: 600 14px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .ctext { fill: #1f2328; font: 400 14px system-ui, -apple-system, "Segoe UI", sans-serif; }
  .ltext { fill: #59636e; font: 400 12px system-ui, -apple-system, "Segoe UI", sans-serif; }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #0d1117; }
    .title, .ntitle, .ctext { fill: #e6edf3; }
    .lane, .nsub, .ltext { fill: #9198a1; }
    .node { fill: #0f2e1b; stroke: #3fb950; }
    .node.planned { fill: #151b23; stroke: #656c76; }
    .status { fill: #2e2404; stroke: #bb8009; }
    .stext { fill: #e3b341; }
    .edge { stroke: #3d444d; }
    .arrow { fill: #3d444d; }
    .lit-edge, .lit-node { stroke: #4493f8; }
    .act-node { fill: #4493f8; stroke: #4493f8; fill-opacity: .2; }
    .token { fill: #4493f8; stroke: #0d1117; }
    .tlabel { fill: #4493f8; stroke: #0d1117; }
    .cbox { fill: #151b23; stroke: #3d444d; }
    .cwho { fill: #4493f8; }
  }
"""


def box(name: str) -> Tuple[float, float, float, float]:
    if name in NODES:
        x, y, w, h = NODES[name][:4]
    else:
        x, y, w = STATUSES[name][:3]
        h = STATUS_HEIGHT
    return x, y, w, h


def center(name: str) -> Point:
    return box(name)[:2]


def edge_points(a: str, b: str) -> List[Point]:
    return [center(a), *EDGES[(a, b)], center(b)]


def length(points: Sequence[Point]) -> float:
    return sum(abs(x2 - x1) + abs(y2 - y1) for (x1, y1), (x2, y2) in zip(points, points[1:]))


def clipped_end(points: List[Point], target: str) -> List[Point]:
    """Stop an edge at the target's border so its arrowhead stays visible."""
    (px, py), (ex, ey) = points[-2], points[-1]
    _, _, w, h = box(target)
    if px == ex:
        ey = ey - h / 2 - 3 if py < ey else ey + h / 2 + 3
    else:
        ex = ex - w / 2 - 3 if px < ex else ex + w / 2 + 3
    return [*points[:-1], (ex, ey)]


def path_d(points: Sequence[Point]) -> str:
    return "M" + " L".join(f"{x:g},{y:g}" for x, y in points)


class Timeline:
    """Collects visibility intervals and motion keyframes, then emits SMIL."""

    def __init__(self) -> None:
        self.intervals: Dict[str, List[Tuple[float, float]]] = {}
        self.cycle = 0.0

    def show(self, element: str, start: float, end: float) -> None:
        self.intervals.setdefault(element, []).append((start, end))

    def visibility(self, element: str) -> str:
        times, values = [0.0], ["0"]
        for start, end in sorted(self.intervals.get(element, [])):
            if start <= 0:
                values[0] = "1"
            else:
                times.append(start)
                values.append("1")
            times.append(end)
            values.append("0")
        return self._animate("opacity", times, values)

    def motion(self, route: Sequence[Point], frames: List[Tuple[float, float]]) -> str:
        total = length(route)
        if frames[0][0] > 0:
            frames = [(0.0, frames[0][1]), *frames]
        if frames[-1][0] < self.cycle:
            frames = [*frames, (self.cycle, frames[-1][1])]
        key_times = ";".join(f"{t / self.cycle:.6f}" for t, _ in frames)
        key_points = ";".join(f"{d / total:.6f}" for _, d in frames)
        return (
            f'<animateMotion dur="{self.cycle:.2f}s" repeatCount="indefinite" calcMode="linear" '
            f'keyTimes="{key_times}" keyPoints="{key_points}" path="{path_d(route)}"/>'
        )

    def _animate(self, attribute: str, times: List[float], values: List[str]) -> str:
        key_times = ";".join(f"{t / self.cycle:.6f}" for t in times)
        return (
            f'<animate attributeName="{attribute}" dur="{self.cycle:.2f}s" repeatCount="indefinite" '
            f'calcMode="discrete" keyTimes="{key_times}" values="{";".join(values)}"/>'
        )


def plan(scenarios: list) -> Tuple[Timeline, list]:
    """Lay out every scenario on one looping clock."""
    timeline = Timeline()
    tokens = []
    t = 0.0
    for index, scenario in enumerate(scenarios):
        stops = [name for name, _ in scenario["stops"]]
        route: List[Point] = [center(stops[0])]
        distance_at = {stops[0]: 0.0}
        for a, b in zip(stops, stops[1:]):
            route.extend(edge_points(a, b)[1:])
            distance_at[b] = length(route)

        start = t
        arrive: Dict[str, float] = {}
        depart: Dict[str, float] = {}
        frames = []
        for i, name in enumerate(stops):
            if i:
                t += length(edge_points(stops[i - 1], name)) / SPEED
            arrive[name] = t
            frames.append((t, distance_at[name]))
            t += FINAL_HOLD if i == len(stops) - 1 else DWELL
            depart[name] = t
            frames.append((t, distance_at[name]))
        end = t

        for i, (name, caption) in enumerate(scenario["stops"]):
            timeline.show(f"lit-{name}", arrive[name], end)
            timeline.show(f"act-{name}", arrive[name], depart[name] if i < len(stops) - 1 else end)
            timeline.show(f"cap-{index}-{i}", arrive[name], arrive[stops[i + 1]] if i < len(stops) - 1 else end)
            if i:
                timeline.show(f"lit-{stops[i - 1]}-{name}", depart[stops[i - 1]], end)
        timeline.show(f"token-{index}", start, end)
        timeline.show(f"who-{index}", start, end)
        tokens.append((f"token-{index}", scenario["token"], route, frames))

        branch = scenario.get("branch")
        if branch:
            a, via, b, sync = branch["from"], branch["via"], branch["to"], branch["sync"]
            first, second = edge_points(a, via), edge_points(via, b)
            branch_route = [*first, *second[1:]]
            reach = length(first)
            branch_frames = [
                (depart[a], 0.0),
                (arrive[sync], reach),
                (depart[sync], reach),
                (arrive[b], length(branch_route)),
            ]
            timeline.show(f"lit-{via}", arrive[sync], end)
            timeline.show(f"act-{via}", arrive[sync], depart[sync])
            timeline.show(f"lit-{a}-{via}", depart[a], end)
            timeline.show(f"lit-{via}-{b}", depart[sync], end)
            timeline.show(f"token-{index}b", depart[a], arrive[b])
            tokens.append((f"token-{index}b", "", branch_route, branch_frames))
        t += GAP
    timeline.cycle = t
    return timeline, tokens


def render() -> str:
    timeline, tokens = plan(SCENARIOS)
    out: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" '
        f'height="{HEIGHT}" role="img" aria-labelledby="t d">',
        "<title id=\"t\">How a lead flows through LeadFlow AI</title>",
        "<desc id=\"d\">Animated diagram. Example leads move from the web form through schema check, "
        "screening, the duplicate check, the existing-account check, and the rep filter. New prospects go "
        "to the buyer agent and capacity service in parallel, then the routing agent, validation and "
        "commit, and the rep handoff. Other leads stop at dropped, self_serve, duplicate, preserve_owner, "
        "ownership_exception, or no_eligible_reps.</desc>",
        f"<style>{STYLE}</style>",
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="arrow" d="M0,0 L10,5 L0,10 z"/></marker></defs>',
        f'<rect class="bg" width="{WIDTH}" height="{HEIGHT}" rx="12"/>',
        '<text class="title" x="24" y="34">How a lead is routed</text>',
        '<text class="lane" x="24" y="56">DETERMINISTIC TRIAGE</text>',
        '<text class="lane" x="308" y="322">AGENTS AND ROUTING</text>',
    ]

    for (a, b), _ in EDGES.items():
        out.append(f'<path class="edge" marker-end="url(#arrow)" d="{path_d(clipped_end(edge_points(a, b), b))}"/>')
    for (a, b), _ in EDGES.items():
        element = f"lit-{a}-{b}"
        if element in timeline.intervals:
            out.append(f'<path class="lit-edge" opacity="0" d="{path_d(edge_points(a, b))}">'
                       f"{timeline.visibility(element)}</path>")

    # Leads travel under the boxes, so a paused lead never covers a label.
    for element, label, route, frames in tokens:
        out.append(f'<g opacity="0">{timeline.visibility(element)}{timeline.motion(route, frames)}'
                   '<circle class="token" r="8"/>'
                   + (f'<text class="tlabel" x="12" y="-12">{escape(label)}</text>' if label else "")
                   + "</g>")

    for name, (x, y, w, h, title, subtitle) in NODES.items():
        planned = " planned" if name in PLANNED else ""
        out.append(f'<rect class="node{planned}" x="{x - w / 2:g}" y="{y - h / 2:g}" width="{w:g}" '
                   f'height="{h:g}" rx="10"/>')
        out.append(f'<text class="ntitle" x="{x:g}" y="{y - 3:g}" text-anchor="middle">{escape(title)}</text>')
        out.append(f'<text class="nsub" x="{x:g}" y="{y + 14:g}" text-anchor="middle">{escape(subtitle)}</text>')
    for name, (x, y, w, label) in STATUSES.items():
        h = STATUS_HEIGHT
        out.append(f'<rect class="status" x="{x - w / 2:g}" y="{y - h / 2:g}" width="{w:g}" height="{h:g}" '
                   f'rx="{h / 2:g}"/>')
        out.append(f'<text class="stext" x="{x:g}" y="{y + 4:g}" text-anchor="middle">{escape(label)}</text>')

    for kind in ("lit", "act"):
        for name in [*NODES, *STATUSES]:
            element = f"{kind}-{name}"
            if element in timeline.intervals:
                x, y, w, h = box(name)
                rx = h / 2 if name in STATUSES else 10
                out.append(f'<rect class="{kind}-node" opacity="0" x="{x - w / 2 - 3:g}" y="{y - h / 2 - 3:g}" '
                           f'width="{w + 6:g}" height="{h + 6:g}" rx="{rx + 3:g}">'
                           f"{timeline.visibility(element)}</rect>")

    out.extend(legend())

    out.append(f'<rect class="cbox" x="20" y="484" width="{WIDTH - 40}" height="62" rx="10"/>')
    for index, scenario in enumerate(SCENARIOS):
        out.append(f'<text class="cwho" x="40" y="508" opacity="0">{escape(scenario["who"])}'
                   f"{timeline.visibility(f'who-{index}')}</text>")
        for i, (_, caption) in enumerate(scenario["stops"]):
            element = f"cap-{index}-{i}"
            out.append(f'<text class="ctext" x="40" y="531" opacity="0">'
                       f"{escape(caption)}{timeline.visibility(element)}</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def legend() -> List[str]:
    x, y = 30, 360
    return [
        f'<rect class="node" x="{x}" y="{y}" width="26" height="16" rx="4"/>',
        f'<text class="ltext" x="{x + 36}" y="{y + 12}">Built, runs from POST /leads</text>',
        f'<rect class="node planned" x="{x}" y="{y + 28}" width="26" height="16" rx="4"/>',
        f'<text class="ltext" x="{x + 36}" y="{y + 40}">Planned</text>',
        f'<rect class="status" x="{x}" y="{y + 56}" width="26" height="16" rx="8"/>',
        f'<text class="ltext" x="{x + 36}" y="{y + 68}">Status the API returns</text>',
        f'<circle class="token" cx="{x + 13}" cy="{y + 92}" r="7"/>',
        f'<text class="ltext" x="{x + 36}" y="{y + 96}">Example lead</text>',
    ]


def main(output: Optional[Path] = None) -> None:
    target = output or OUTPUT
    target.write_text(render())
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
