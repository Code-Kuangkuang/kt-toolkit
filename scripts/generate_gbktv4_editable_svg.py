from pathlib import Path
from xml.sax.saxutils import escape


OUT = Path("figures/model_diagrams/gbktv4_model_editable.svg")
W, H = 2400, 1450


COLORS = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "green": "#8BCF8B",
    "green_light": "#EAF8EA",
    "green_mid": "#AADCA9",
    "gold": "#FFD700",
    "gold_light": "#FFF1CC",
    "orange": "#B56B00",
    "red": "#B64342",
    "red_light": "#FCE8EE",
    "violet": "#9A4D8E",
    "violet_light": "#EFE9FF",
    "gray": "#64748B",
    "gray2": "#CBD5E1",
    "ink": "#15202B",
    "panel": "#FFFFFF",
    "input": "#F4F7FB",
    "time": "#FFF7D6",
    "update": "#F6F6F6",
}


parts = []


def add(s):
    parts.append(s)


def attrs(**kw):
    out = []
    for k, v in kw.items():
        if v is None:
            continue
        k = k.replace("_", "-")
        out.append(f'{k}="{escape(str(v))}"')
    return " ".join(out)


def text(x, y, label, size=24, fill=None, weight=None, anchor="middle", italic=False, family="Georgia, 'Times New Roman', serif"):
    fill = fill or COLORS["ink"]
    style = []
    if weight:
        style.append(f"font-weight:{weight}")
    if italic:
        style.append("font-style:italic")
    style.append(f"font-family:{family}")
    lines = str(label).split("\n")
    add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" {" ".join(["style=\"" + ";".join(style) + "\"" if style else ""])}>')
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else size * 1.15
        add(f'<tspan x="{x}" dy="{dy}">{escape(line)}</tspan>')
    add("</text>")


def rect(x, y, w, h, fill="white", stroke=None, sw=2, rx=10, dash=None, opacity=1):
    add(
        f'<rect {attrs(x=x, y=y, width=w, height=h, rx=rx, ry=rx, fill=fill, stroke=stroke or COLORS["gray"], stroke_width=sw, stroke_dasharray=dash, opacity=opacity)} />'
    )


def line(x1, y1, x2, y2, stroke=None, sw=3, dash=None, marker=True):
    add(
        f'<line {attrs(x1=x1, y1=y1, x2=x2, y2=y2, stroke=stroke or COLORS["ink"], stroke_width=sw, stroke_dasharray=dash, marker_end="url(#arrow)" if marker else None)} />'
    )


def path(d, stroke=None, sw=3, fill="none", dash=None, marker=True):
    add(
        f'<path {attrs(d=d, fill=fill, stroke=stroke or COLORS["ink"], stroke_width=sw, stroke_dasharray=dash, marker_end="url(#arrow)" if marker else None)} />'
    )


def plus(x, y, r=18):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{COLORS["ink"]}" stroke-width="2"/>')
    line(x - r * 0.55, y, x + r * 0.55, y, sw=2, marker=False)
    line(x, y - r * 0.55, x, y + r * 0.55, sw=2, marker=False)


def times(x, y, r=18):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{COLORS["ink"]}" stroke-width="2"/>')
    line(x - r * 0.45, y - r * 0.45, x + r * 0.45, y + r * 0.45, sw=2, marker=False)
    line(x - r * 0.45, y + r * 0.45, x + r * 0.45, y - r * 0.45, sw=2, marker=False)


def box(x, y, w, h, title, fill, stroke=None, size=22, weight=None):
    rect(x, y, w, h, fill=fill, stroke=stroke or COLORS["gray"], sw=2.2, rx=10)
    text(x + w / 2, y + h / 2 - (size * 0.42 if "\n" in title else -size * 0.25), title, size=size, weight=weight)


def header_panel(x, y, w, h, title, header_color=None, dash=True):
    header_color = header_color or COLORS["blue"]
    rect(x, y, w, h, fill="white", stroke=COLORS["blue"], sw=2.2, rx=12, dash="8 8" if dash else None)
    rect(x, y, w, 45, fill=header_color, stroke=header_color, sw=0, rx=10)
    text(x + w / 2, y + 31, title, size=26, fill="white", weight="bold")


def matrix(x, y, cols=4, rows=4, cell=20, colors=None, stroke="#758CA6"):
    colors = colors or ["#E8F3FF", "#D6E8FF", "#BDD8F6", "#8FB8E7"]
    for r in range(rows):
        for c in range(cols):
            fill = colors[(r + c) % len(colors)]
            add(f'<rect x="{x + c * cell}" y="{y + r * cell}" width="{cell}" height="{cell}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')


def vector(x, y, n=5, cell=20, colors=None):
    matrix(x, y, cols=1, rows=n, cell=cell, colors=colors)


def ball(cx, cy, r, kind="student", label=None, sub=None):
    grad = "gradStudent" if kind == "student" else "gradItem"
    edge = "#2E7D32" if kind == "student" else COLORS["orange"]
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#{grad})" stroke="{edge}" stroke-width="3"/>')
    add(f'<circle cx="{cx}" cy="{cy}" r="{r * 0.70}" fill="none" stroke="{edge}" stroke-width="2" stroke-dasharray="8 6" opacity="0.75"/>')
    add(f'<circle cx="{cx}" cy="{cy}" r="9" fill="{edge}"/>')
    line(cx, cy, cx + r * 0.72, cy - r * 0.10, stroke=edge, sw=2.2, marker=True)
    text(cx + r * 0.50, cy - r * 0.21, "r_h" if kind == "student" else "r_d", size=20, fill=edge, weight="bold")
    if label:
        text(cx, cy - r - 22, label, size=21, fill=edge, weight="bold")
    if sub:
        text(cx, cy + r + 28, sub, size=20, fill=edge, italic=True)


def doc_icon(x, y, color):
    rect(x, y, 38, 48, fill=color, stroke=COLORS["gray"], sw=1.8, rx=2)
    add(f'<path d="M{x+26},{y} L{x+38},{y+12} L{x+26},{y+12} Z" fill="#FFFFFF" stroke="{COLORS["gray"]}" stroke-width="1.3"/>')
    line(x + 8, y + 18, x + 28, y + 18, sw=1.3, stroke="#8AA0B7", marker=False)
    line(x + 8, y + 27, x + 31, y + 27, sw=1.3, stroke="#8AA0B7", marker=False)
    line(x + 8, y + 36, x + 24, y + 36, sw=1.3, stroke="#8AA0B7", marker=False)


def checkmark(x, y, ok=True):
    if ok:
        add(f'<path d="M{x},{y} l10,12 l24,-30" fill="none" stroke="#0FA958" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>')
    else:
        line(x, y, x + 28, y + 28, stroke="#E11D48", sw=6, marker=False)
        line(x + 28, y, x, y + 28, stroke="#E11D48", sw=6, marker=False)


def small_legend_item(x, y, label, kind):
    if kind == "matrix":
        matrix(x, y - 18, 3, 3, 14, ["#F2F8EF", "#DDF3DE", "#B8D9B5"])
    elif kind == "plus":
        plus(x + 20, y - 2, 14)
    elif kind == "times":
        times(x + 20, y - 2, 14)
    elif kind == "radius":
        add(f'<circle cx="{x+18}" cy="{y-4}" r="18" fill="#EAF8EA" stroke="#2E7D32" stroke-width="2" stroke-dasharray="6 4"/>')
    elif kind == "center":
        add(f'<circle cx="{x+18}" cy="{y-4}" r="9" fill="#2E7D32"/>')
    elif kind == "conditioning":
        add(f'<circle cx="{x+12}" cy="{y-4}" r="18" fill="#DDF3DE" stroke="#2E7D32" stroke-width="1.5" opacity="0.7"/>')
        add(f'<circle cx="{x+32}" cy="{y-4}" r="18" fill="#E8F3FF" stroke="#3775BA" stroke-width="1.5" opacity="0.55"/>')
    text(x + 64, y + 3, label, size=16, anchor="start")


def build():
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    add(
        """
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M2,2 L10,6 L2,10 Z" fill="#15202B"/>
  </marker>
  <radialGradient id="gradStudent" cx="33%" cy="28%" r="70%">
    <stop offset="0%" stop-color="#F2FFF1"/>
    <stop offset="25%" stop-color="#AADCA9"/>
    <stop offset="62%" stop-color="#6FA86B"/>
    <stop offset="100%" stop-color="#2E5F2D"/>
  </radialGradient>
  <radialGradient id="gradItem" cx="33%" cy="28%" r="70%">
    <stop offset="0%" stop-color="#FFF8C4"/>
    <stop offset="28%" stop-color="#FFD700"/>
    <stop offset="68%" stop-color="#D99400"/>
    <stop offset="100%" stop-color="#8A4B00"/>
  </radialGradient>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.12"/>
  </filter>
</defs>
"""
    )
    add('<rect x="0" y="0" width="2400" height="1450" fill="white"/>')
    text(W / 2, 52, "GBKTV4-Clean: Granular Ball Knowledge Tracing", size=42, weight="bold")
    text(
        W / 2,
        92,
        "Time-aware acquisition, distance-aware retrieval, concept-conditioned readout, and IRT ball-to-ball prediction",
        size=22,
        italic=True,
        fill=COLORS["gray"],
    )

    # Panels
    header_panel(16, 120, 620, 620, "1. Interaction Sequence Input")
    header_panel(652, 120, 1080, 620, "2. Knowledge Internalization Modeling")
    header_panel(1748, 120, 636, 620, "3. Prediction Module")
    header_panel(16, 770, 700, 570, "4. Temporal Information Mining", COLORS["gold"])
    header_panel(732, 770, 760, 570, "5. Concept-Guided Historical Retrieval", "#5B45B6")
    header_panel(1508, 770, 876, 570, "6. Granular Ball Dynamic Update", "#6B7280")

    # Panel 1
    x0, y0 = 16, 120
    text(x0 + 40, y0 + 80, "Current step  t", size=20, fill=COLORS["blue"], anchor="start", italic=True)
    labels = [("q_t", "#E8F3FF"), ("c_t", "#FFF1CC"), ("r_t", "#EAF8EA"), ("time", "#E8F3FF")]
    for i, (lab, col) in enumerate(labels):
        y = y0 + 120 + i * 78
        text(x0 + 70, y + 30, lab, size=26, anchor="start", italic=True)
        doc_icon(x0 + 130, y, col)
        line(x0 + 182, y + 24, x0 + 260, y + 24)
    box(x0 + 270, y0 + 130, 72, 300, "Embedding\nLayer", "#F6F6F6", size=20)
    for i, col in enumerate(["#E8F3FF", "#FFF1CC", "#EAF8EA", "#E8F3FF"]):
        y = y0 + 134 + i * 72
        line(x0 + 342, y + 24, x0 + 422, y + 24)
        matrix(x0 + 430, y + 10, cols=4, rows=1, cell=28, colors=[col, "#FFFFFF"])
    line(x0 + 20, y0 + 430, x0 + 600, y0 + 430, stroke=COLORS["gray2"], sw=2, dash="7 6", marker=False)
    text(x0 + 42, y0 + 470, "Next step (t + 1)", size=20, fill=COLORS["blue"], anchor="start", italic=True)
    for i, lab in enumerate(["q_{t+1}", "c_{t+1}", "?"]):
        x = x0 + 150 + i * 165
        text(x, y0 + 520, lab, size=24, italic=True)
        if lab == "?":
            rect(x - 28, y0 + 540, 56, 56, fill="#FFFFFF", stroke=COLORS["gray"], dash="8 6")
            text(x, y0 + 578, "?", size=28, weight="bold")
        else:
            doc_icon(x - 18, y0 + 540, "#E8F3FF" if i == 0 else "#FFF1CC")

    # Panel 2
    p2x, p2y = 652, 120
    steps = [
        (684, 190, 150, 330, "1) Q-C\nRepresentation", COLORS["input"]),
        (850, 190, 220, 330, "2) QDB:\nQuestion Difficulty Ball\nB_d=(mu_d,r_d)", COLORS["gold_light"]),
        (1088, 190, 220, 330, "3) KSB:\nKnowledge State Ball\nB_h=(mu_h,r_h)", COLORS["green_light"]),
        (1326, 190, 170, 330, "4) Time-aware\nKAB", "#FFF7E8"),
        (1512, 190, 190, 330, "5) State Update +\nHistorical Ball\nAttention", COLORS["green_light"]),
    ]
    for x, y, w, h, lab, fc in steps:
        rect(x, y, w, h, fill=fc, stroke=COLORS["gray"], sw=2, rx=9)
        text(x + w / 2, y + 42, lab, size=18)
    matrix(706, 330, 3, 3, 34)
    matrix(708, 444, 3, 2, 34, ["#FFF1CC", "#FFE599", "#F8D166"])
    ball(960, 385, 88, "item")
    ball(1198, 385, 88, "student")
    line(834, 355, 850, 355)
    line(1070, 355, 1088, 355)
    line(1308, 355, 1326, 355)
    box(1350, 300, 120, 70, "Time-aware\nAcquisition\n(KAB)", COLORS["green_light"], size=16)
    box(1365, 390, 90, 50, "GRU", "#F6F6F6", size=18)
    box(1365, 455, 90, 50, "Gate\nγ_t", "#F6F6F6", size=16)
    plus(1512, 350, 20)
    box(1535, 270, 140, 75, "State Update\n(GRU + Gate)", COLORS["green_light"], size=16)
    box(1535, 395, 140, 78, "Historical Ball\nAttention", COLORS["green_light"], size=16)
    path("M1015,520 L1015,570 L1410,570 L1410,520", stroke=COLORS["ink"], sw=2.5, marker=True)
    path("M1250,520 L1250,570", stroke=COLORS["ink"], sw=2.5, marker=True)
    line(1496, 350, 1510, 350)
    line(1496, 430, 1535, 430)
    rect(690, 600, 980, 110, fill="#FFFFFF", stroke=COLORS["blue"], sw=2, dash="8 8", rx=12)
    text(1180, 633, "Information Flow", size=20, fill=COLORS["blue"], weight="bold")
    text(850, 675, "Q-C Rep.", size=18)
    line(910, 668, 980, 668)
    text(1040, 675, "QDB (B_d)", size=18)
    line(1105, 668, 1170, 668)
    text(1230, 675, "KSB (B_h)", size=18)
    line(1295, 668, 1360, 668)
    text(1436, 675, "KAB (Time-aware)", size=18)
    plus(1530, 668, 17)
    text(1608, 660, "State Update\n+ Historical Attention", size=16)

    # Panel 3
    p3x, p3y = 1748, 120
    box(1782, 190, 250, 150, "6) Concept-conditioned\nReadout\nB_h -> B_h^c", COLORS["green_light"], size=17)
    ball(1910, 305, 42, "student")
    box(2125, 190, 210, 138, "7) BBP:\nIRT Ball-to-Ball\nPrediction\np_ball", COLORS["red_light"], stroke=COLORS["red"], size=16)
    line(2032, 265, 2125, 265)
    line(2230, 328, 2230, 390)
    box(1800, 398, 270, 110, "8) Concept Decoder\np_concept", COLORS["violet_light"], stroke=COLORS["violet"], size=18)
    matrix(1846, 468, 7, 1, 24, ["#E9D5FF", "#D8B4FE", "#C4B5FD"])
    line(1910, 340, 1910, 398)
    box(2120, 390, 220, 145, "9) Conservative\nResidual Fusion\nw_c <= 0.30", COLORS["red_light"], stroke=COLORS["red"], size=17)
    line(2070, 455, 2120, 455)
    box(1825, 575, 470, 85, "10) Output:  y = P(correct)", "#FFF7F7", stroke=COLORS["red"], size=22)
    matrix(1875, 630, 12, 1, 30, ["#AADCA9", "#DDF3DE", "#FFFFFF", "#F9CFA9", "#F97316"])
    line(2230, 535, 2230, 575)
    text(1866, 710, "Learner", size=18)
    line(1955, 695, 2080, 695)
    text(2138, 703, "y", size=25, italic=True)
    line(2170, 695, 2260, 695)
    doc_icon(2290, 670, "#F4F7FB")
    text(2318, 740, "Question (q_{t+1})", size=16)

    # Panel 4
    bx, by = 16, 770
    for i, label in enumerate(["1) Interval Time", "2) Log Time\nFeatures", "3) Time\nProjection", "4) Radius\nExpansion"]):
        x = bx + 28 + i * 166
        text(x + 66, by + 95, label, size=18)
        if i > 0:
            line(x - 18, by + 70, x - 18, by + 520, stroke=COLORS["gray2"], sw=1.8, dash="7 6", marker=False)
    add(f'<circle cx="{bx+90}" cy="{by+205}" r="35" fill="#FFFFFF" stroke="{COLORS["ink"]}" stroke-width="3"/>')
    line(bx + 90, by + 205, bx + 90, by + 178, sw=2, marker=False)
    line(bx + 90, by + 205, bx + 112, by + 222, sw=2, marker=False)
    text(bx + 150, by + 210, "Δ_t", size=24, italic=True)
    text(bx + 92, by + 320, "Use-time  u_t", size=19)
    add(f'<path d="M{bx+62},{by+350} L{bx+118},{by+350} L{bx+101},{by+415} L{bx+79},{by+415} Z" fill="#FFFFFF" stroke="{COLORS["ink"]}" stroke-width="3"/>')
    path(f"M{bx+235},{by+290} L{bx+235},{by+180} L{bx+360},{by+250}", stroke=COLORS["blue"], sw=3, marker=False)
    path(f"M{bx+235},{by+430} L{bx+235},{by+325} L{bx+360},{by+365}", stroke=COLORS["blue"], sw=3, marker=False)
    box(bx + 445, by + 172, 92, 70, "MLP\nProjection", "#E8F3FF", size=16)
    vector(bx + 475, by + 284, 5, 22)
    line(bx + 491, by + 242, bx + 491, by + 284)
    ball(bx + 620, by + 215, 48, "student")
    ball(bx + 620, by + 410, 70, "student")
    line(bx + 620, by + 270, bx + 620, by + 330)
    text(bx + 638, by + 313, "Expand", size=17)

    # Panel 5
    cx, cy = 732, 770
    box(cx + 25, cy + 90, 180, 110, "1) Next Concept\nEmbedding\nc_{t+1}", COLORS["violet_light"], stroke=COLORS["violet"], size=16)
    matrix(cx + 75, cy + 170, 5, 1, 24, ["#E9D5FF", "#D8B4FE", "#FFFFFF"])
    box(cx + 25, cy + 250, 180, 140, "2) Difficulty Ball\nB_d=(mu_d,r_d)", COLORS["gold_light"], stroke=COLORS["orange"], size=16)
    ball(cx + 116, cy + 340, 44, "item")
    box(cx + 268, cy + 166, 160, 125, "3) Concept-conditioned\nReadout\nB_h^c", "#F4F7FB", stroke=COLORS["blue2"], size=15)
    vector(cx + 337, cy + 260, 5, 22, ["#DDF3DE", "#AADCA9", "#8BCF8B"])
    box(cx + 465, cy + 168, 120, 160, "4) Attention\nWeights\nw_1\nw_2\n...\nw_n\nsoftmax", "#FAF5FF", stroke=COLORS["violet"], size=15)
    box(cx + 620, cy + 166, 150, 170, "5) Student x Concept\nAttention Score Matrix", "#F8FAFC", stroke=COLORS["blue2"], size=14)
    matrix(cx + 642, cy + 240, 5, 5, 22, ["#EAF8EA", "#DDF3DE", "#AADCA9", "#8BCF8B"])
    line(cx + 205, cy + 145, cx + 268, cy + 210)
    line(cx + 205, cy + 320, cx + 268, cy + 230)
    line(cx + 428, cy + 230, cx + 465, cy + 230)
    line(cx + 585, cy + 245, cx + 620, cy + 245)
    rect(cx + 78, cy + 440, 210, 65, fill="#F8FAFC", stroke=COLORS["gray"], sw=1.8, rx=8)
    text(cx + 183, cy + 466, "Historical States", size=16)
    matrix(cx + 128, cy + 476, 5, 1, 20, ["#DDF3DE", "#AADCA9", "#FFFFFF"])
    times(cx + 420, cy + 455, 18)
    path(f"M{cx+205},{cy+445} L{cx+405},{cy+455} L{cx+465},{cy+328}", stroke=COLORS["ink"], sw=2.2)

    # Panel 6
    dx, dy = 1508, 770
    text(dx + 28, dy + 84, "A. Knowledge Center Update  (mu_h)", size=20, anchor="start", weight="bold")
    text(dx + 34, dy + 128, "Current\nCenter", size=14)
    vector(dx + 70, dy + 150, 5, 24, ["#DDF3DE", "#AADCA9", "#8BCF8B"])
    text(dx + 150, dy + 218, "=", size=34)
    labels2 = ["State\nTransformation", "Acquisition", "Concept\nInfluence", "Bias"]
    xs = [dx + 235, dx + 420, dx + 610, dx + 790]
    cols = [["#E8F3FF", "#BDD8F6"], ["#FFF1CC", "#FFD700"], ["#EAF8EA", "#8BCF8B"], ["#F6F6F6", "#D9DEE6"]]
    for i, x in enumerate(xs):
        text(x + 50, dy + 128, labels2[i], size=14)
        if i == 3:
            vector(x + 42, dy + 150, 5, 24, cols[i])
        else:
            matrix(x, dy + 150, 4, 4, 28, cols[i])
        if i < 3:
            text(x + 137, dy + 218, "+", size=34)
    line(dx + 18, dy + 325, dx + 850, dy + 325, stroke=COLORS["gray"], sw=2, dash="8 6", marker=False)
    text(dx + 28, dy + 363, "B. Radius Update  (r_h)", size=20, anchor="start", weight="bold")
    text(dx + 34, dy + 405, "Current\nRadius", size=14)
    vector(dx + 72, dy + 425, 5, 21, ["#DDF3DE", "#AADCA9", "#8BCF8B"])
    text(dx + 150, dy + 490, "=", size=34)
    labels3 = ["Radius\nTransformation", "Uncertainty\nExpansion", "Acquisition\nGate", "Bias"]
    cols3 = [["#E8F3FF", "#BDD8F6"], ["#FFF1CC", "#F9CFA9"], ["#EAF8EA", "#8BCF8B"], ["#F6F6F6", "#D9DEE6"]]
    for i, x in enumerate(xs):
        text(x + 50, dy + 405, labels3[i], size=14)
        if i == 3:
            vector(x + 45, dy + 425, 5, 21, cols3[i])
        else:
            matrix(x, dy + 425, 4, 4, 24, cols3[i])
        if i < 3:
            text(x + 124, dy + 490, "+", size=34)
    rect(dx + 28, dy + 500, 810, 62, fill="#FFFFFF", stroke=COLORS["ink"], sw=1.8, dash="8 6", rx=8)
    text(dx + 52, dy + 526, "Notes:  W_c, W_r transformation matrices    •  g_t acquisition gate (0-1)", size=15, anchor="start")
    text(dx + 52, dy + 550, "        C_h^c concept-conditioned coefficients    •  E_t time-aware expansion", size=15, anchor="start")

    # Legend
    rect(24, 1370, 2352, 62, fill="white", stroke=COLORS["ink"], sw=1.8, dash="8 8", rx=9)
    small_legend_item(52, 1405, "Matrix / Embedding", "matrix")
    small_legend_item(360, 1405, "Element-wise sum", "plus")
    small_legend_item(650, 1405, "Similarity / weighted sum", "times")
    small_legend_item(1010, 1405, "Radius (uncertainty)", "radius")
    small_legend_item(1320, 1405, "Center (semantic position)", "center")
    small_legend_item(1625, 1405, "Conditioning / combination", "conditioning")
    text(2140, 1408, "Low", size=16, italic=True)
    add('<linearGradient id="heat" x1="0%" x2="100%" y1="0%" y2="0%"><stop offset="0%" stop-color="#EAF8EA"/><stop offset="55%" stop-color="#AADCA9"/><stop offset="100%" stop-color="#F97316"/></linearGradient>')
    add(f'<rect x="2185" y="1388" width="120" height="20" fill="url(#heat)" stroke="{COLORS["gray"]}" stroke-width="1"/>')
    text(2334, 1408, "High", size=16, italic=True)

    add("</svg>")


if __name__ == "__main__":
    build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUT.resolve()}")
