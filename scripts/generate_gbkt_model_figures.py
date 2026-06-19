from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path("figures/model_diagrams")


COLORS = {
    "input": "#F4F7FB",
    "embed": "#E8F3FF",
    "ball": "#FFF1CC",
    "state": "#EAF8EA",
    "pred": "#FCE8EE",
    "aux": "#EFE9FF",
    "time": "#E6F7F8",
    "loss": "#F6F6F6",
    "edge": "#263238",
    "muted": "#D8DEE9",
    "text": "#15202B",
}


def wrap(label, width=22):
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def add_box(ax, xy, wh, label, fc, ec="#4B5563", fontsize=9.5, lw=1.2, dashed=False):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        wrap(label),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        family="DejaVu Sans",
    )
    return patch


def center(xy, wh):
    return xy[0] + wh[0] / 2, xy[1] + wh[1] / 2


def side(xy, wh, where):
    x, y = xy
    w, h = wh
    if where == "left":
        return x, y + h / 2
    if where == "right":
        return x + w, y + h / 2
    if where == "top":
        return x + w / 2, y + h
    if where == "bottom":
        return x + w / 2, y
    raise ValueError(where)


def arrow(ax, start, end, label=None, color=None, rad=0.0, style="-"):
    color = color or COLORS["edge"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.15,
        color=color,
        linestyle=style,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.05,
            label,
            fontsize=8,
            ha="center",
            va="center",
            color=color,
            family="DejaVu Sans",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.86),
        )


def setup(title, subtitle=None):
    fig, ax = plt.subplots(figsize=(16, 8.2), dpi=160)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(
        0.4,
        7.62,
        title,
        ha="left",
        va="center",
        fontsize=18,
        weight="bold",
        color=COLORS["text"],
        family="DejaVu Sans",
    )
    if subtitle:
        ax.text(
            0.42,
            7.25,
            subtitle,
            ha="left",
            va="center",
            fontsize=10.5,
            color="#4B5563",
            family="DejaVu Sans",
        )
    return fig, ax


def save(fig, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=240, facecolor="white")
    plt.close(fig)


def add_panel(ax, xy, wh, title, fc="#FFFFFF", ec="#CBD5E1"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=1.25,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.28,
        y + h - 0.33,
        title,
        ha="left",
        va="center",
        fontsize=12,
        weight="bold",
        color=COLORS["text"],
        family="DejaVu Sans",
    )
    return patch


def add_tag(ax, xy, label, fc="#F8FAFC", ec="#CBD5E1", fontsize=8.4):
    x, y = xy
    width = max(0.48 + 0.07 * len(label), 1.0)
    height = 0.36
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=0.9,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        family="DejaVu Sans",
    )
    return patch


def draw_point_vs_ball_kt():
    fig, ax = setup(
        "From Point-Based KT to Ball-Space GBKT",
        "Point embeddings estimate where a state is; GBKT additionally models how uncertain that state is and uses it in prediction.",
    )

    left_panel = ((0.45, 0.62), (7.25, 6.32))
    right_panel = ((8.3, 0.62), (7.25, 6.32))
    add_panel(ax, *left_panel, "(a) Point-Based Deep KT")
    add_panel(ax, *right_panel, "(b) GBKT: Ball-Space KT")

    # Left panel: deterministic point representation.
    hist = ((0.85, 5.75), (1.75, 0.55))
    encoder = ((3.05, 5.62), (1.8, 0.78))
    point_state = ((5.45, 5.75), (1.55, 0.55))
    item_point = ((0.95, 1.1), (1.55, 0.55))
    pred = ((3.25, 1.0), (2.0, 0.75))
    limitation = ((5.65, 1.0), (1.45, 0.75))
    add_box(ax, *hist, "Learning History", COLORS["input"], fontsize=8.6)
    add_box(ax, *encoder, "Sequence Encoder", COLORS["embed"], fontsize=8.6)
    add_box(ax, *point_state, "Student Point h_t", COLORS["state"], fontsize=8.4)
    add_box(ax, *item_point, "Item Point d_q", COLORS["ball"], fontsize=8.4)
    add_box(ax, *pred, "Prediction f(h_t,d_q)", COLORS["pred"], fontsize=8.4)
    add_box(ax, *limitation, "Uncertainty\nimplicit", COLORS["loss"], fontsize=8.2)
    arrow(ax, side(*hist, "right"), side(*encoder, "left"))
    arrow(ax, side(*encoder, "right"), side(*point_state, "left"))
    arrow(ax, side(*point_state, "bottom"), side(*pred, "top"), rad=-0.1)
    arrow(ax, side(*item_point, "right"), side(*pred, "left"))
    arrow(ax, side(*pred, "right"), side(*limitation, "left"))

    # Latent space sketch.
    ax.add_patch(Rectangle((1.0, 2.15), 6.28, 2.82, linewidth=1.0, edgecolor="#CBD5E1", facecolor="#FFFFFF"))
    ax.text(1.18, 4.73, "Latent knowledge space", fontsize=8.4, color="#475569", family="DejaVu Sans")
    h_xy = (2.15, 3.75)
    d_xy = (5.75, 3.0)
    ax.scatter([h_xy[0]], [h_xy[1]], s=70, c="#4CAF50", edgecolors="#1B5E20", linewidths=1.0, zorder=3)
    ax.scatter([d_xy[0]], [d_xy[1]], s=70, c="#F9C74F", edgecolors="#A16207", linewidths=1.0, zorder=3)
    ax.plot([h_xy[0], d_xy[0]], [h_xy[1], d_xy[1]], linestyle="--", color="#64748B", linewidth=1.25)
    ax.text(h_xy[0] - 0.28, h_xy[1] + 0.27, "h_t", fontsize=9, weight="bold", color="#1B5E20")
    ax.text(d_xy[0] + 0.12, d_xy[1] - 0.25, "d_q", fontsize=9, weight="bold", color="#A16207")
    ax.text(3.55, 3.18, "distance only", fontsize=8, color="#475569", rotation=-12, family="DejaVu Sans")

    # Right panel: ball-space representation.
    r_hist = ((8.72, 5.72), (1.55, 0.58))
    r_readout = ((10.55, 5.62), (1.8, 0.78))
    r_balls = ((12.75, 5.72), (1.9, 0.58))
    r_pred = ((13.05, 0.95), (1.8, 0.7))
    add_box(ax, *r_hist, "History Balls", COLORS["input"], fontsize=8.5)
    add_box(ax, *r_readout, "Concept Readout", COLORS["state"], fontsize=8.5)
    add_box(ax, *r_balls, "Student + Item Balls", COLORS["ball"], fontsize=8.2)
    add_box(ax, *r_pred, "IRT-Grounded Prediction", COLORS["pred"], fontsize=8.2)
    arrow(ax, side(*r_hist, "right"), side(*r_readout, "left"))
    arrow(ax, side(*r_readout, "right"), side(*r_balls, "left"))
    arrow(ax, side(*r_balls, "bottom"), side(*r_pred, "top"))

    ax.add_patch(Rectangle((8.85, 2.28), 6.28, 2.72, linewidth=1.0, edgecolor="#CBD5E1", facecolor="#FFFFFF"))
    ax.text(9.03, 4.76, "Ball-space knowledge geometry", fontsize=8.4, color="#475569", family="DejaVu Sans")
    student_c = (10.55, 3.55)
    item_c = (13.25, 3.18)
    student_r = 0.92
    item_r = 0.62
    ax.add_patch(Circle(student_c, student_r, facecolor="#BFE8C4", edgecolor="#1B5E20", alpha=0.55, linewidth=1.5))
    ax.add_patch(Circle(item_c, item_r, facecolor="#FFE6A3", edgecolor="#A16207", alpha=0.62, linewidth=1.5))
    ax.scatter([student_c[0], item_c[0]], [student_c[1], item_c[1]], s=38, c=["#2E7D32", "#A16207"], zorder=3)
    ax.plot([student_c[0], item_c[0]], [student_c[1], item_c[1]], linestyle="--", color="#334155", linewidth=1.1)
    ax.text(student_c[0] - 0.55, student_c[1] + 1.1, "B_h=(mu_h,r_h)", fontsize=8.5, color="#1B5E20", weight="bold")
    ax.text(item_c[0] - 0.36, item_c[1] + 0.83, "B_d=(mu_d,r_d)", fontsize=8.5, color="#A16207", weight="bold")
    ax.text(11.45, 3.05, "center displacement", fontsize=7.8, color="#334155", rotation=-8, family="DejaVu Sans")
    arrow(ax, student_c, (student_c[0] + student_r, student_c[1]), label="r_h", color="#2E7D32")
    arrow(ax, item_c, (item_c[0] + item_r, item_c[1]), label="r_d", color="#A16207")

    equation = (
        "effective mismatch = ||mu_h - mu_d|| / (r_h + r_d)\n"
        "logit = a(theta - b)"
    )
    ax.text(
        9.05,
        2.34,
        equation,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color=COLORS["text"],
        family="DejaVu Sans",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F8FAFC", ec="#CBD5E1", lw=0.9),
    )

    attn = ((9.0, 0.95), (1.85, 0.62))
    fusion = ((11.08, 0.95), (1.35, 0.62))
    add_box(ax, *attn, "Radius-Gated History", COLORS["aux"], fontsize=8.0)
    add_box(ax, *fusion, "Fusion Gate", COLORS["aux"], fontsize=8.0)
    arrow(ax, (10.55, 2.62), side(*attn, "top"), color="#6D28D9", rad=0.08)
    arrow(ax, (13.25, 2.68), side(*fusion, "top"), color="#6D28D9", rad=-0.1)
    arrow(ax, side(*attn, "right"), side(*fusion, "left"), color="#6D28D9")
    arrow(ax, side(*fusion, "right"), side(*r_pred, "left"), color="#6D28D9")
    ax.text(
        10.1,
        1.86,
        "radius conditions evidence selection and residual fusion",
        fontsize=7.9,
        color="#6D28D9",
        family="DejaVu Sans",
    )

    save(fig, "point_vs_ball_kt")


def draw_gbkt():
    fig, ax = setup(
        "GBKT: Granular Ball Knowledge Tracing",
        "Base model: question difficulty ball, knowledge state ball, ball-to-ball prediction, and acquisition update.",
    )

    q_in = ((0.55, 5.95), (1.55, 0.62))
    c_in = ((0.55, 5.05), (1.55, 0.62))
    r_in = ((5.25, 1.95), (2.1, 0.62))
    emb = ((2.65, 5.18), (2.05, 1.05))
    qdb = ((5.25, 5.18), (2.1, 1.05))
    state = ((2.65, 3.2), (2.05, 0.95))
    kab = ((5.25, 3.2), (2.1, 0.95))
    update = ((7.9, 3.2), (2.1, 0.95))
    hist = ((7.9, 1.8), (2.1, 0.72))
    q_next = ((9.85, 5.65), (2.0, 0.72))
    qdb_next = ((12.25, 5.25), (2.0, 0.95))
    bbp = ((12.25, 3.25), (2.0, 0.95))
    out = ((12.25, 1.75), (2.0, 0.72))
    loss = ((6.0, 0.62), (3.1, 0.68))

    add_box(ax, *q_in, "Question q_t", COLORS["input"])
    add_box(ax, *c_in, "Concepts c_t", COLORS["input"])
    add_box(ax, *r_in, "Response r_t", COLORS["input"])
    add_box(ax, *emb, "Question-Concept Representation", COLORS["embed"])
    add_box(ax, *qdb, "QDB Difficulty Ball\nB_d^t = (mu_d, r_d)", COLORS["ball"])
    add_box(ax, *state, "KSB Knowledge Ball\nB_h^t = (mu_h, r_h)", COLORS["state"])
    add_box(ax, *kab, "KAB Acquisition Update", COLORS["state"])
    add_box(ax, *update, "GRU + Concept Gate", COLORS["state"])
    add_box(ax, *hist, "Historical Ball Attention", COLORS["state"])
    add_box(ax, *q_next, "Next q,c", COLORS["input"])
    add_box(ax, *qdb_next, "Next Difficulty Ball\nB_d^{t+1}", COLORS["ball"])
    add_box(ax, *bbp, "BBP Ball-to-Ball Prediction", COLORS["pred"])
    add_box(ax, *out, "p_{t+1}, theta, confidence", COLORS["pred"])
    add_box(ax, *loss, "Training Loss: BCE + theta/radius/confidence regs", COLORS["loss"], fontsize=8.8)

    arrow(ax, side(*q_in, "right"), side(*emb, "left"))
    arrow(ax, side(*c_in, "right"), side(*emb, "left"))
    arrow(ax, side(*emb, "right"), side(*qdb, "left"))
    arrow(ax, side(*r_in, "top"), side(*kab, "bottom"))
    arrow(ax, side(*qdb, "bottom"), side(*kab, "top"))
    arrow(ax, side(*state, "right"), side(*kab, "left"))
    arrow(ax, side(*kab, "right"), side(*update, "left"))
    arrow(ax, side(*state, "right"), side(*update, "left"), rad=-0.18)
    arrow(ax, side(*update, "bottom"), side(*hist, "top"))
    arrow(ax, side(*hist, "right"), side(*bbp, "left"), rad=-0.22)
    arrow(ax, side(*q_next, "right"), side(*qdb_next, "left"))
    arrow(ax, side(*qdb_next, "bottom"), side(*bbp, "top"))
    arrow(ax, side(*bbp, "bottom"), side(*out, "top"))
    arrow(ax, side(*out, "left"), side(*loss, "right"), rad=-0.12)

    save(fig, "gbkt_model")


def draw_gbktv2():
    fig, ax = setup(
        "GBKTV2: Concept-Supervised GBKT",
        "Conservative upgrade: keep GBKT ball dynamics and add a next-concept auxiliary decoder.",
    )

    hist = ((0.65, 4.7), (2.0, 0.8))
    backbone = ((3.2, 4.15), (2.35, 1.35))
    ball_pred = ((6.3, 5.0), (2.0, 0.78))
    concept_dec = ((6.3, 3.55), (2.0, 0.78))
    next_c = ((3.35, 2.7), (2.05, 0.68))
    fusion = ((9.1, 4.35), (2.15, 1.0))
    out = ((12.05, 4.48), (1.8, 0.72))
    loss = ((8.55, 2.0), (3.05, 0.72))
    note = ((0.85, 1.35), (3.9, 0.78))

    add_box(ax, *hist, "q_t, c_t, r_t sequence", COLORS["input"])
    add_box(ax, *backbone, "GBKT Backbone: QDB + KSB + KAB + BBP", COLORS["state"])
    add_box(ax, *ball_pred, "Ball Prediction p_ball", COLORS["pred"])
    add_box(ax, *concept_dec, "Next-Concept Decoder p_concept", COLORS["aux"])
    add_box(ax, *next_c, "Next Concept c_{t+1}", COLORS["input"])
    add_box(ax, *fusion, "Logit Fusion\nlogit(p_ball) + w logit(p_concept)", COLORS["pred"], fontsize=8.8)
    add_box(ax, *out, "Final p_{t+1}", COLORS["pred"])
    add_box(ax, *loss, "Loss: BCE(y) + ball/concept auxiliary BCE + GBKT regs", COLORS["loss"], fontsize=8.5)
    add_box(ax, *note, "Design goal: add concept-level supervision while keeping GBKT dynamics unchanged.", COLORS["loss"], fontsize=8.6)

    arrow(ax, side(*hist, "right"), side(*backbone, "left"))
    arrow(ax, side(*backbone, "right"), side(*ball_pred, "left"))
    arrow(ax, side(*backbone, "right"), side(*concept_dec, "left"))
    arrow(ax, side(*next_c, "top"), side(*concept_dec, "bottom"))
    arrow(ax, side(*ball_pred, "right"), side(*fusion, "left"))
    arrow(ax, side(*concept_dec, "right"), side(*fusion, "left"))
    arrow(ax, side(*fusion, "right"), side(*out, "left"))
    arrow(ax, side(*fusion, "bottom"), side(*loss, "top"))
    arrow(ax, side(*concept_dec, "bottom"), side(*loss, "top"), rad=0.12)

    save(fig, "gbktv2_model")


def draw_gbktv3():
    fig, ax = setup(
        "GBKTV3: Time-Aware Concept-Heavy GBKT",
        "Final optimized direction: time-aware uncertainty update, concept-conditioned readout, and concept-heavy residual fusion.",
    )

    seq = ((0.5, 5.75), (1.9, 0.78))
    time = ((0.5, 4.45), (1.9, 0.78))
    time_proj = ((2.95, 4.45), (2.0, 0.92))
    kab = ((5.45, 5.4), (2.0, 0.85))
    forget = ((5.45, 4.05), (2.0, 0.85))
    update = ((7.9, 4.75), (2.05, 0.92))
    next_c = ((7.9, 3.1), (2.05, 0.72))
    readout = ((10.25, 3.95), (2.05, 0.95))
    ball = ((12.1, 5.25), (1.85, 0.72))
    concept = ((12.1, 3.65), (1.85, 0.72))
    fusion = ((13.75, 4.0), (2.1, 1.05))
    out = ((13.95, 2.65), (1.75, 0.68))
    loss = ((5.0, 1.2), (3.35, 0.72))

    add_box(ax, *seq, "q_t, c_t, r_t", COLORS["input"])
    add_box(ax, *time, "Interval it_t and Use-time ut_t", COLORS["time"])
    add_box(ax, *time_proj, "Log Time Features + Time Projection", COLORS["time"])
    add_box(ax, *forget, "Time Forgetting: radius expansion", COLORS["time"])
    add_box(ax, *kab, "Time-aware KAB", COLORS["state"])
    add_box(ax, *update, "State Update + Historical Attention", COLORS["state"])
    add_box(ax, *readout, "Concept-Conditioned Readout\nB_h -> B_h^c", COLORS["state"])
    add_box(ax, *next_c, "Next Concept and Difficulty Ball", COLORS["input"])
    add_box(ax, *ball, "Ball Prediction p_ball", COLORS["pred"])
    add_box(ax, *concept, "Concept Decoder p_concept", COLORS["aux"])
    add_box(ax, *fusion, "Concept-heavy Residual Fusion\nball + w_c(concept - ball)", COLORS["pred"], fontsize=7.9)
    add_box(ax, *out, "Final p_{t+1}", COLORS["pred"])
    add_box(ax, *loss, "Loss: BCE(y) + ball/concept auxiliary BCE + GBKT regs", COLORS["loss"], fontsize=8.3)

    arrow(ax, side(*seq, "right"), side(*kab, "left"), rad=0.08)
    arrow(ax, side(*time, "right"), side(*time_proj, "left"))
    arrow(ax, side(*time_proj, "right"), side(*forget, "left"))
    arrow(ax, side(*time_proj, "top"), side(*kab, "bottom"), rad=0.18)
    arrow(ax, side(*forget, "top"), side(*kab, "bottom"), rad=-0.16)
    arrow(ax, side(*kab, "right"), side(*update, "left"))
    arrow(ax, side(*update, "right"), side(*readout, "left"), rad=0.12)
    arrow(ax, side(*next_c, "right"), side(*readout, "left"), rad=-0.12)
    arrow(ax, side(*readout, "right"), side(*ball, "left"), rad=0.12)
    arrow(ax, side(*readout, "right"), side(*concept, "left"), rad=-0.08)
    arrow(ax, side(*ball, "right"), side(*fusion, "left"), rad=0.08)
    arrow(ax, side(*concept, "right"), side(*fusion, "left"), rad=-0.08)
    arrow(ax, side(*fusion, "bottom"), side(*out, "top"))
    arrow(ax, side(*out, "left"), side(*loss, "right"), rad=-0.2)

    save(fig, "gbktv3_model")


def main():
    draw_gbkt()
    draw_gbktv2()
    draw_gbktv3()
    print(f"Wrote diagrams to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
