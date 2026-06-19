from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("figures/model_diagrams")


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1": "#F6CFCB",
    "red_2": "#E9A6A1",
    "red_strong": "#B64342",
    "neutral": "#CFCECE",
    "highlight": "#FFD700",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "ink": "#15202B",
    "edge": "#263238",
    "muted": "#64748B",
}


COLORS = {
    "input": "#F4F7FB",
    "embed": "#E8F3FF",
    "time": "#E6F7F8",
    "state": "#EAF8EA",
    "ball": "#FFF1CC",
    "pred": "#FCE8EE",
    "aux": "#EFE9FF",
    "loss": "#F8FAFC",
}


def apply_publication_style():
    plt.rcParams.update(
        {
            "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 15,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 2.2,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def wrap(label, width=24):
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def add_box(ax, xy, wh, label, fc, fontsize=8.6, lw=1.2, ec="#4B5563", dashed=False):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        wrap(label),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=PALETTE["ink"],
        zorder=3,
    )
    return patch


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


def arrow(ax, start, end, label=None, color=None, rad=0.0, style="-", lw=1.25):
    color = color or PALETTE["edge"]
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        linestyle=style,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=5,
        shrinkB=5,
        zorder=4,
    )
    ax.add_patch(patch)
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(
            mx,
            my + 0.08,
            label,
            fontsize=7.4,
            ha="center",
            va="center",
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.9),
            zorder=5,
        )


def color_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


def sphere_rgba(base_color, resolution=260, alpha=0.98, light_dir=(-0.45, 0.55, 0.78)):
    base = color_to_rgb(base_color)
    y, x = np.ogrid[-1:1:complex(resolution), -1:1:complex(resolution)]
    rr = x * x + y * y
    mask = rr <= 1.0
    z = np.sqrt(np.clip(1.0 - rr, 0.0, 1.0))

    normal = np.dstack([np.broadcast_to(x, z.shape), np.broadcast_to(y, z.shape), z])
    light = np.array(light_dir, dtype=float)
    light = light / np.linalg.norm(light)
    shade = np.clip((normal * light).sum(axis=2), 0.0, 1.0)

    rim = np.clip(1.0 - z, 0.0, 1.0)
    highlight = np.exp(-((x + 0.36) ** 2 + (y - 0.42) ** 2) / 0.055)
    shadow = np.exp(-((x - 0.35) ** 2 + (y + 0.38) ** 2) / 0.16)

    rgb = base * (0.48 + 0.48 * shade[..., None])
    rgb = rgb * (1.0 - 0.26 * rim[..., None])
    rgb = rgb * (1.0 - 0.18 * shadow[..., None])
    rgb = rgb + (1.0 - rgb) * (0.58 * highlight[..., None])
    rgb = np.clip(rgb, 0.0, 1.0)

    rgba = np.zeros((resolution, resolution, 4), dtype=float)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(mask, alpha, 0.0)
    return rgba


def add_sphere(ax, center, radius, color, edge, label, sublabel=None, label_side="top"):
    x, y = center
    img = sphere_rgba(color)
    ax.imshow(
        img,
        extent=(x - radius, x + radius, y - radius, y + radius),
        origin="lower",
        interpolation="bilinear",
        zorder=3,
    )
    ax.add_patch(
        Circle(
            center,
            radius,
            facecolor="none",
            edgecolor=edge,
            linewidth=1.6,
            zorder=4,
        )
    )
    ax.scatter([x], [y], s=18, c=edge, zorder=5)
    if label_side == "top":
        tx, ty, va = x, y + radius + 0.30, "bottom"
    elif label_side == "bottom":
        tx, ty, va = x, y - radius - 0.18, "top"
    else:
        tx, ty, va = x + radius + 0.18, y, "center"
    ax.text(tx, ty, label, ha="center" if label_side != "right" else "left", va=va, fontsize=8.2, weight="bold", color=edge)
    if sublabel:
        sub_y = ty - 0.18 if label_side == "top" else ty - 0.18
        ax.text(
            tx,
            sub_y,
            sublabel,
            ha="center" if label_side != "right" else "left",
            va="top",
            fontsize=7.0,
            color=PALETTE["muted"],
        )


def add_section_label(ax, xy, label, color):
    ax.text(
        xy[0],
        xy[1],
        label,
        ha="left",
        va="center",
        fontsize=9.5,
        weight="bold",
        color=color,
    )
    ax.plot([xy[0], xy[0] + 1.0], [xy[1] - 0.16, xy[1] - 0.16], color=color, lw=2.2)


def draw_gbktv4():
    apply_publication_style()
    fig, ax = plt.subplots(figsize=(16.8, 9.2), dpi=170)
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    ax.text(
        0.35,
        8.78,
        "GBKTV4-Clean: Ball-Space Knowledge Tracing with Time and Rasch Bias",
        ha="left",
        va="center",
        fontsize=18,
        weight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        0.37,
        8.38,
        "Final v4 path: time-aware acquisition, distance-aware history attention, concept readout, scalar item difficulty, and conservative residual fusion.",
        ha="left",
        va="center",
        fontsize=9.6,
        color="#475569",
    )

    # Inputs and representation.
    seq = ((0.45, 6.35), (1.75, 0.72))
    time = ((0.45, 5.28), (1.75, 0.72))
    emb = ((2.75, 6.0), (2.0, 1.02))
    q_repr = ((5.25, 6.0), (2.15, 1.02))
    qdb = ((7.95, 6.0), (1.9, 1.02))
    time_proj = ((2.75, 4.92), (2.0, 0.76))

    add_section_label(ax, (0.45, 7.64), "Input and difficulty encoding", PALETTE["blue_main"])
    add_box(ax, *seq, "q_t, c_t, r_t\nquestion, concept, response", COLORS["input"])
    add_box(ax, *time, "it_t, ut_t\ninterval and use time", COLORS["time"])
    add_box(ax, *emb, "Question/Concept Embedding", COLORS["embed"])
    add_box(ax, *q_repr, "Residual q_repr\n[q+c, q*c] + MLP", COLORS["embed"], fontsize=8.2)
    add_box(ax, *qdb, "QDB\nDifficulty Ball", COLORS["ball"], fontsize=8.4)
    add_box(ax, *time_proj, "log1p time features", COLORS["time"], fontsize=8.2)

    arrow(ax, side(*seq, "right"), side(*emb, "left"))
    arrow(ax, side(*emb, "right"), side(*q_repr, "left"))
    arrow(ax, side(*q_repr, "right"), side(*qdb, "left"))
    arrow(ax, side(*time, "right"), side(*time_proj, "left"))

    add_sphere(
        ax,
        (10.35, 6.52),
        0.34,
        PALETTE["highlight"],
        "#A16207",
        "B_d^t",
        "(mu_d, r_d)",
        label_side="right",
    )

    # State update loop.
    init_state = ((0.55, 3.58), (1.75, 0.72))
    forget = ((2.75, 3.9), (2.0, 0.76))
    kab = ((5.15, 3.9), (2.0, 0.76))
    update = ((7.55, 3.9), (2.05, 0.76))
    hist = ((10.0, 3.9), (2.2, 0.76))
    readout = ((12.65, 3.9), (2.0, 0.76))
    next_concept = ((12.65, 2.72), (2.0, 0.64))

    add_section_label(ax, (0.55, 4.8), "Knowledge-state update", PALETTE["green_3"])
    add_box(ax, *init_state, "Initial Knowledge Ball\nB_h^0", COLORS["state"])
    add_box(ax, *forget, "Time Forgetting\nradius expansion", COLORS["time"], fontsize=8.0)
    add_box(ax, *kab, "Time-aware KAB\nacquisition ball", COLORS["state"], fontsize=8.0)
    add_box(ax, *update, "GRU State Update\n+ concept gate", COLORS["state"], fontsize=8.0)
    add_box(ax, *hist, "Distance-aware\nHistory Attention", COLORS["state"], fontsize=8.0)
    add_box(ax, *readout, "Concept Readout\nB_h -> B_h^c", COLORS["state"], fontsize=8.0)
    add_box(ax, *next_concept, "Next c_{t+1}\nreadout key", COLORS["input"], fontsize=7.8)

    arrow(ax, side(*init_state, "right"), side(*forget, "left"))
    arrow(ax, side(*time_proj, "bottom"), side(*forget, "top"), color=PALETTE["teal"], rad=0.12)
    arrow(ax, side(*qdb, "bottom"), side(*kab, "top"), color="#A16207", rad=-0.18, label="B_d^t")
    arrow(ax, side(*forget, "right"), side(*kab, "left"))
    arrow(ax, side(*kab, "right"), side(*update, "left"))
    arrow(ax, side(*update, "right"), side(*hist, "left"), color=PALETTE["green_3"])
    arrow(ax, side(*hist, "right"), side(*readout, "left"))
    arrow(ax, side(*next_concept, "top"), side(*readout, "bottom"), color=PALETTE["blue_secondary"], rad=-0.06)

    # Prediction plane with planar 3D balls.
    plane = FancyBboxPatch(
        (0.55, 0.58),
        8.85,
        1.9,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        linewidth=1.15,
        edgecolor="#CBD5E1",
        facecolor="#FFFFFF",
        zorder=1,
    )
    ax.add_patch(plane)
    ax.text(0.62, 2.68, "Ball-to-ball geometry: planar 3D ball rendering", fontsize=8.4, color="#475569", weight="bold")

    student_c = (2.25, 1.38)
    item_c = (5.65, 1.32)
    add_sphere(ax, student_c, 0.73, PALETTE["green_2"], "#2E7D32", "Student state", "B_h^c", label_side="top")
    add_sphere(ax, item_c, 0.54, PALETTE["highlight"], "#A16207", "Next item", "B_d^{t+1}", label_side="top")
    ax.plot([student_c[0], item_c[0]], [student_c[1], item_c[1]], "--", lw=1.15, color="#334155", zorder=2)
    arrow(ax, student_c, (student_c[0] + 0.73, student_c[1]), label="r_h", color="#2E7D32", lw=1.0)
    arrow(ax, item_c, (item_c[0] + 0.54, item_c[1]), label="r_d", color="#A16207", lw=1.0)
    ax.text(3.38, 1.06, "||mu_h - mu_d|| / (r_h + r_d)", fontsize=7.6, color="#334155", rotation=-1)

    bbp = ((7.15, 1.0), (1.8, 0.82))
    add_box(ax, *bbp, "BBP + IRT\nlogit_ball", COLORS["pred"], fontsize=7.9)
    arrow(ax, (6.22, 1.32), side(*bbp, "left"), color=PALETTE["red_strong"])

    # Prediction and fusion.
    rasch = ((9.9, 1.38), (1.95, 0.72))
    concept = ((9.9, 0.45), (1.95, 0.72))
    fusion = ((12.45, 0.9), (2.2, 1.02))
    out = ((15.05, 1.02), (1.25, 0.72))
    loss = ((12.45, 2.28), (2.2, 0.64))

    add_section_label(ax, (9.85, 2.75), "Prediction heads", PALETTE["red_strong"])
    add_box(ax, *rasch, "Scalar Rasch\nitem difficulty", COLORS["ball"], fontsize=7.9)
    add_box(ax, *concept, "Concept-next\nauxiliary head", COLORS["aux"], fontsize=7.9)
    add_box(ax, *fusion, "Dynamic Residual Fusion\nw_c <= 0.30", COLORS["pred"], fontsize=8.0)
    add_box(ax, *out, "Final y\nP(correct)", COLORS["pred"], fontsize=8.2)
    add_box(ax, *loss, "Clean loss: BCE + ball + concept + theta + item-L2", COLORS["loss"], fontsize=7.2)

    arrow(ax, side(*bbp, "right"), side(*rasch, "left"), color=PALETTE["red_strong"])
    arrow(ax, side(*readout, "bottom"), side(*concept, "top"), color=PALETTE["violet"], rad=0.18)
    arrow(ax, side(*rasch, "right"), side(*fusion, "left"), color=PALETTE["red_strong"], rad=0.08, label="y_ball")
    arrow(ax, side(*concept, "right"), side(*fusion, "left"), color=PALETTE["violet"], rad=-0.08, label="y_concept")
    arrow(ax, side(*fusion, "right"), side(*out, "left"), color=PALETTE["red_strong"])
    arrow(ax, side(*fusion, "top"), side(*loss, "bottom"), color="#475569")

    ax.text(
        11.45,
        7.22,
        "Final v4 keeps the GBKTV3 main path,\nremoves question branch execution by default,\nand adds two clean gains: sequence-distance attention\nand scalar Rasch item difficulty.",
        ha="left",
        va="top",
        fontsize=8.1,
        color="#475569",
        bbox=dict(boxstyle="round,pad=0.32", fc="#F8FAFC", ec="#CBD5E1", lw=0.9),
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            OUT_DIR / f"gbktv4_model.{ext}",
            bbox_inches="tight",
            pad_inches=0.08,
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def main():
    draw_gbktv4()
    print(f"Wrote GBKTV4 diagram to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
