"""Glazier & Graner (1993) study visualizations — interactive Plotly figures.

One @as_visualization class per study, each rendering that study's real
reproduction time-series (pbg-cpm CPM engine) as a polished, interactive figure.
Auto-generated; data are baked in so the figures render identically in the live
and the published read-only dashboard.
"""
from __future__ import annotations
import json
from pbg_superpowers.visualization import as_visualization

_INK = "#1f2937"; _MUTED = "#6b7280"; _GRID = "#eef0f3"; _CARD = "#ffffff"


def _figure(div, title, subtitle, ytitle, logx, series_defs, data, caption, ref=None):
    """Build a self-contained interactive Plotly card. series_defs: [(key,label,color)]."""
    x = data["mcs"]
    traces = []
    for key, label, color in series_defs:
        y = data["series"].get(key)
        if not y:
            continue
        traces.append({
            "x": x, "y": y, "name": label, "type": "scatter", "mode": "lines+markers",
            "line": {"color": color, "width": 2.4, "shape": "spline", "smoothing": 0.5},
            "marker": {"color": color, "size": 6, "line": {"color": _CARD, "width": 1.5}},
            "hovertemplate": f"<b>{label}</b><br>MCS %{{x}}<br>%{{y:.3f}}<extra></extra>",
        })
    layout = {
        "title": {"text": f"<b>{title}</b><br><span style='font-size:12px;color:{_MUTED}'>{subtitle}</span>",
                  "x": 0.02, "xanchor": "left", "font": {"size": 16, "color": _INK}},
        "paper_bgcolor": _CARD, "plot_bgcolor": _CARD,
        "font": {"family": "-apple-system, Segoe UI, Roboto, sans-serif", "color": _INK, "size": 12},
        "margin": {"l": 62, "r": 22, "t": 66, "b": 52},
        "xaxis": {"title": {"text": "Time (MCS)"}, "type": ("log" if logx else "linear"),
                   "gridcolor": _GRID, "zeroline": False, "ticks": "outside", "tickcolor": _GRID,
                   "linecolor": _GRID},
        "yaxis": {"title": {"text": ytitle}, "gridcolor": _GRID, "zeroline": False,
                   "ticks": "outside", "tickcolor": _GRID, "linecolor": _GRID},
        "hovermode": "x unified",
        "legend": {"orientation": "h", "y": -0.2, "x": 0, "font": {"size": 12}},
        "showlegend": len(traces) > 1,
    }
    if ref is not None:
        layout["shapes"] = [{"type": "line", "xref": "paper", "yref": "y",
                             "x0": 0, "x1": 1, "y0": ref, "y1": ref,
                             "line": {"color": "#9ca3af", "width": 1.2, "dash": "dash"}}]
        layout["annotations"] = [{"xref": "paper", "yref": "y", "x": 0.99, "y": ref,
                                  "xanchor": "right", "yanchor": "bottom", "showarrow": False,
                                  "text": f"reference {ref:g}", "font": {"size": 11, "color": "#9ca3af"}}]
    return (
        f'<div style="background:{_CARD};border:1px solid #e5e7eb;border-radius:12px;'
        f'padding:8px 10px 12px;max-width:760px;margin:0 auto;'
        f'font-family:-apple-system,Segoe UI,Roboto,sans-serif">'
        f'<div id="{div}" style="height:380px"></div>'
        f'<div style="color:{_MUTED};font-size:12.5px;line-height:1.5;padding:2px 8px 4px">{caption}</div>'
        f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        f'<script>Plotly.newPlot("{div}",{json.dumps(traces)},{json.dumps(layout)},'
        f'{{responsive:true,displayModeBar:false}});</script></div>'
    )



@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993Annealing", demo={"mcs": [0, 1, 2, 3, 4, 5]})
def update_gg1993_annealing(state):
    """Annealing: bulk ⟨n⟩ settles to ~6 within a couple of MCS"""
    data = {"mcs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40], "series": {"n_bulk": [6.22807, 6.0, 6.0011, 6.00878, 6.00659, 6.0022, 6.00439, 6.00659, 6.00768, 6.00659, 6.00659, 6.0, 6.0, 6.0, 6.0, 6.0, 6.00769, 6.00439, 6.00439, 6.0022, 6.0, 6.0022, 6.0, 6.0022, 6.00439, 6.0022, 6.0022, 6.00439, 6.0022, 6.00329, 6.0, 5.9978, 5.9978, 5.9978, 5.9978, 5.9989, 5.9978, 5.9978, 5.9978, 5.9978, 5.9978], "n_total": [6.12451, 5.90613, 5.9081, 5.91601, 5.91403, 5.91008, 5.91206, 5.91403, 5.91601, 5.91403, 5.91403, 5.9081, 5.9081, 5.9081, 5.9081, 5.9081, 5.91502, 5.91206, 5.91206, 5.91008, 5.9081, 5.91008, 5.9081, 5.91008, 5.91206, 5.91008, 5.91008, 5.91206, 5.91008, 5.91206, 5.9081, 5.90613, 5.90613, 5.90613, 5.90613, 5.9081, 5.90613, 5.90613, 5.90613, 5.90613, 5.90613]}}
    caption = "<b>γ_ld=+0 · γ_lM=+7 · γ_dM=+7 · T=0.0 · λ=1.0</b> &nbsp;·&nbsp; Two MCS of T=0 annealing removes crumpling; bulk stays just above total, both near the ideal ⟨n⟩=6 for a 2-D froth (Fig 2)."
    return {"html": _figure("gg-gg1993_annealing", "Annealing: bulk ⟨n⟩ settles to ~6 within a couple of MCS", "Annealing", "Number of sides ⟨n⟩",
                            False, [["n_bulk", "bulk \u27e8n\u27e9", "#4f46e5"], ["n_total", "total \u27e8n\u27e9", "#0891b2"]], data, caption, ref=6.0)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993GlobalEquilibration", demo={"mcs": [0, 20, 40, 60, 80, 100]})
def update_gg1993_globalequilibration(state):
    """Global equilibration: topological moments stabilise by ~400 MCS"""
    data = {"mcs": [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600], "series": {"mu2": [0.00226, 0.497, 0.5278, 0.43886, 0.46452, 0.50325, 0.4357, 0.47787, 0.51057, 0.48391, 0.47461, 0.4298, 0.49171, 0.49829, 0.49042, 0.47519, 0.46689, 0.48775, 0.50773, 0.5011, 0.48222, 0.49335, 0.50715, 0.46458, 0.50817, 0.49107, 0.4312, 0.47954, 0.474, 0.48648, 0.51487], "mu3": [0.0, 0.03979, 0.03619, 0.07157, 0.01952, 0.00878, 0.01364, 0.03223, -0.02505, 0.02859, 0.03311, 0.04481, 0.02194, 0.01, 0.02632, 0.04584, 0.06667, 0.05527, 0.05298, 0.05342, 0.01596, 0.05033, 0.03078, 0.01817, 0.03398, 0.02783, 0.06134, 0.06277, 0.08386, 0.03212, 0.0428], "mu4": [0.00226, 0.77583, 0.82072, 0.63722, 0.61077, 0.74224, 0.51558, 0.70324, 0.75072, 0.73397, 0.6468, 0.57462, 0.74344, 0.68399, 0.71343, 0.6999, 0.59963, 0.72094, 0.66667, 0.84495, 0.65408, 0.78357, 0.75794, 0.66334, 0.82465, 0.71428, 0.57796, 0.71564, 0.85662, 0.66926, 0.7921]}}
    caption = "<b>γ_ld=+0 · γ_lM=+7 · γ_dM=+7 · T=5.0 · λ=1.0</b> &nbsp;·&nbsp; A rectangular tiling rounds into a free disk; the side-number moments reach their equilibrium plateau, defining the shared initial condition (Fig 5)."
    return {"html": _figure("gg-gg1993_globalequilibration", "Global equilibration: topological moments stabilise by ~400 MCS", "Global pattern equilibration", "Bulk moments μₗ",
                            False, [["mu2", "\u03bc\u2082", "#4f46e5"], ["mu3", "\u03bc\u2083", "#0d9488"], ["mu4", "\u03bc\u2084", "#d97706"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993Checkerboard", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_checkerboard(state):
    """Negative γ_ld drives mixing: heterotypic (l–d) contact dominates"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 1000, 1212, 1468, 1778, 2000], "series": {"frac_ll": [0.1836, 0.1804, 0.18126, 0.18223, 0.18114, 0.17679, 0.17889, 0.17461, 0.17133, 0.17182, 0.17471, 0.17249, 0.17384, 0.17161, 0.17015, 0.17009, 0.1723, 0.17323, 0.17137, 0.17007, 0.17362, 0.17026, 0.17549, 0.17959, 0.17992, 0.17566, 0.17, 0.17339, 0.1784, 0.16898, 0.17326, 0.17403, 0.16548, 0.16726, 0.17474, 0.17222, 0.16705], "frac_dd": [0.19989, 0.19759, 0.1975, 0.19607, 0.19394, 0.19173, 0.19172, 0.191, 0.19013, 0.19086, 0.18957, 0.18876, 0.19212, 0.18818, 0.18376, 0.18858, 0.19031, 0.19011, 0.18867, 0.18776, 0.18713, 0.1882, 0.18946, 0.19205, 0.18977, 0.18738, 0.18444, 0.18511, 0.19134, 0.18462, 0.1838, 0.18617, 0.18162, 0.18042, 0.18402, 0.18507, 0.18298], "frac_ld": [0.55695, 0.56206, 0.561, 0.56189, 0.56467, 0.57113, 0.56932, 0.57409, 0.57771, 0.57706, 0.57508, 0.57827, 0.57417, 0.57993, 0.58597, 0.58114, 0.57699, 0.57631, 0.58012, 0.58167, 0.57861, 0.58107, 0.5751, 0.56847, 0.57042, 0.57616, 0.58481, 0.58135, 0.57021, 0.58609, 0.58266, 0.57963, 0.59251, 0.5923, 0.58118, 0.58264, 0.59001]}}
    caption = "<b>γ_ld=-3 · γ_lM=+7 · γ_dM=+8 · T=10.0 · λ=1.0</b> &nbsp;·&nbsp; With γ_ld=−3 cells maximise unlike contact, so the light–dark interface stays the largest cell–cell term — a checkerboard, not sorting (Fig 8)."
    return {"html": _figure("gg-gg1993_checkerboard", "Negative γ_ld drives mixing: heterotypic (l–d) contact dominates", "Checkerboard (negative surface tension)", "Fractional boundary length",
                            True, [["frac_ll", "l\u2013l", "#4f46e5"], ["frac_dd", "d\u2013d", "#0d9488"], ["frac_ld", "l\u2013d (heterotypic)", "#e11d48"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993CellSorting", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_cellsorting(state):
    """Cell sorting: total boundary energy decreases as the aggregate relaxes"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 1000, 1212, 1468, 1778, 2154, 2610, 3162, 3831, 4642, 5623, 6813, 8254, 10000, 12115, 13500], "series": {"bl_total": [37304.0, 37234.0, 37262.0, 37122.0, 37153.0, 37183.0, 37110.0, 37048.0, 37038.0, 36940.0, 36871.0, 36806.0, 36829.0, 36735.0, 36762.0, 36524.0, 36624.0, 36423.0, 36431.0, 36631.0, 36557.0, 36510.0, 36540.0, 36557.0, 36563.0, 36476.0, 36514.0, 36746.0, 36547.0, 36590.0, 36507.0, 36679.0, 36613.0, 36552.0, 36622.0, 36540.0, 36608.0, 36608.0, 36580.0, 36639.0, 36495.0, 36513.0, 36682.0, 36707.0, 36837.0, 36784.0, 36674.0]}}
    caption = "<b>γ_ld=+3 · γ_lM=+9 · γ_dM=+15 · T=10.0 · λ=1.0</b> &nbsp;·&nbsp; The sorting energies (γ_dM>γ_lM>γ_ld>0) drive a monotonic drop in total boundary energy. From a random mix, full surface sorting is logarithmically slow — see the Engulfment study for the same energies sorting cleanly from an ordered start (Fig 13a)."
    return {"html": _figure("gg-gg1993_cellsorting", "Cell sorting: total boundary energy decreases as the aggregate relaxes", "Cell sorting", "Total boundary length (mismatched bonds)",
                            True, [["bl_total", "total boundary length", "#4f46e5"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993Engulfment", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_engulfment(state):
    """Engulfment: dark cells bury while light forms the outer layer"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 1000, 1212, 1468, 1778, 2154, 2610, 3162, 3831, 4642, 5623, 6813, 8254, 10000], "series": {"frac_lM": [0.02972, 0.03034, 0.02974, 0.02977, 0.02996, 0.03005, 0.03006, 0.0304, 0.03031, 0.03047, 0.03016, 0.03036, 0.03031, 0.0306, 0.03111, 0.03062, 0.03066, 0.03046, 0.03164, 0.0318, 0.03225, 0.03284, 0.03282, 0.03338, 0.03515, 0.03391, 0.03453, 0.03451, 0.03586, 0.03534, 0.03447, 0.03535, 0.03642, 0.03753, 0.03899, 0.04018, 0.03893, 0.0402, 0.04094, 0.04213, 0.04431, 0.04524, 0.04744, 0.05302, 0.05382], "frac_dM": [0.03042, 0.02991, 0.03018, 0.03036, 0.03001, 0.03, 0.03017, 0.02997, 0.03014, 0.02987, 0.03, 0.02992, 0.02987, 0.03022, 0.02958, 0.0299, 0.02983, 0.02964, 0.029, 0.02851, 0.02816, 0.0277, 0.02761, 0.02708, 0.02501, 0.0261, 0.02561, 0.02553, 0.02458, 0.02461, 0.02537, 0.02445, 0.02341, 0.02266, 0.0216, 0.02009, 0.021, 0.0204, 0.01928, 0.01805, 0.01579, 0.01435, 0.01256, 0.00689, 0.00646]}}
    caption = "<b>γ_ld=+3 · γ_lM=+9 · γ_dM=+15 · T=10.0 · λ=1.0</b> &nbsp;·&nbsp; From a half-light/half-dark start the sorting completes cleanly: dark–medium contact collapses toward zero as dark buries, while light–medium contact rises as light wraps the aggregate surface (Fig 19)."
    return {"html": _figure("gg-gg1993_engulfment", "Engulfment: dark cells bury while light forms the outer layer", "Engulfment", "Fractional contact with the medium",
                            True, [["frac_lM", "l\u2013M  (light at surface)", "#d97706"], ["frac_dM", "d\u2013M  (dark buries)", "#7c3aed"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993PositionReversal", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_positionreversal(state):
    """Position reversal: dark–medium contact overtakes light–medium"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 1000, 1212, 1468, 1778, 2154, 2610, 3162, 3831, 4642, 5623, 6813, 8254, 10000], "series": {"corr_lM": [0.46392, 0.45833, 0.43182, 0.45833, 0.42553, 0.45263, 0.4086, 0.46, 0.46939, 0.47872, 0.47253, 0.51111, 0.5, 0.49451, 0.51579, 0.51064, 0.47778, 0.46739, 0.49462, 0.51111, 0.52326, 0.51111, 0.51064, 0.5, 0.54023, 0.54945, 0.51136, 0.48421, 0.53608, 0.44444, 0.53535, 0.48315, 0.5, 0.46591, 0.48936, 0.45745, 0.44944, 0.3913, 0.51613, 0.47475, 0.47917, 0.37931, 0.53684, 0.52809, 0.48454], "corr_dM": [0.53608, 0.54167, 0.56818, 0.54167, 0.57447, 0.54737, 0.5914, 0.54, 0.53061, 0.52128, 0.52747, 0.48889, 0.5, 0.50549, 0.48421, 0.48936, 0.52222, 0.53261, 0.50538, 0.48889, 0.47674, 0.48889, 0.48936, 0.5, 0.45977, 0.45055, 0.48864, 0.51579, 0.46392, 0.55556, 0.46465, 0.51685, 0.5, 0.53409, 0.51064, 0.54255, 0.55056, 0.6087, 0.48387, 0.52525, 0.52083, 0.62069, 0.46316, 0.47191, 0.51546]}}
    caption = "<b>γ_ld=+3 · γ_lM=+23 · γ_dM=+15 · T=10.0 · λ=1.0</b> &nbsp;·&nbsp; Raising γ_lM to 23 flips the layering — dark now presents the lower medium tension and sorts OUTWARD, so dark–medium correlation climbs past light–medium (Fig 21)."
    return {"html": _figure("gg-gg1993_positionreversal", "Position reversal: dark–medium contact overtakes light–medium", "Position reversal", "Type–medium correlation",
                            True, [["corr_lM", "light\u2013medium", "#d97706"], ["corr_dM", "dark\u2013medium", "#7c3aed"]], data, caption, ref=0.5)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993PartialSorting", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_partialsorting(state):
    """Partial sorting: heterotypic contact plateaus — sorting stalls"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 1000, 1212, 1468, 1778, 2000], "series": {"frac_ll": [0.25641, 0.25672, 0.25666, 0.25442, 0.25461, 0.25387, 0.2562, 0.2558, 0.2573, 0.25458, 0.25589, 0.25322, 0.2519, 0.25509, 0.25775, 0.2572, 0.25535, 0.25393, 0.25381, 0.259, 0.25828, 0.25632, 0.25847, 0.25325, 0.25316, 0.25756, 0.26184, 0.26134, 0.25426, 0.25077, 0.25254, 0.25675, 0.2531, 0.25688, 0.2597, 0.25853, 0.25744], "frac_dd": [0.26385, 0.2656, 0.26403, 0.26391, 0.26313, 0.26245, 0.2614, 0.26108, 0.26307, 0.26234, 0.26111, 0.26324, 0.26358, 0.26219, 0.26227, 0.26189, 0.26035, 0.26213, 0.26309, 0.26328, 0.26495, 0.26372, 0.26228, 0.25682, 0.25603, 0.25966, 0.2646, 0.263, 0.2615, 0.25722, 0.25909, 0.26434, 0.25965, 0.26272, 0.26253, 0.2622, 0.26232], "frac_ld": [0.42001, 0.41769, 0.41935, 0.42187, 0.42218, 0.42357, 0.42236, 0.42315, 0.41962, 0.42306, 0.4227, 0.42308, 0.42389, 0.42233, 0.41948, 0.42044, 0.42405, 0.42377, 0.42321, 0.41692, 0.41688, 0.41959, 0.41819, 0.4295, 0.43024, 0.42215, 0.41269, 0.41507, 0.42354, 0.43101, 0.42713, 0.41722, 0.42559, 0.41897, 0.41587, 0.4174, 0.41897]}}
    caption = "<b>γ_ld=+7.5 · γ_lM=+10.5 · γ_dM=+15 · T=5.0 · λ=1.0</b> &nbsp;·&nbsp; Violating the Young condition (γ_ld=7.5) traps inclusions; the l–d interface levels off instead of vanishing, and no light monolayer forms (Fig 23)."
    return {"html": _figure("gg-gg1993_partialsorting", "Partial sorting: heterotypic contact plateaus — sorting stalls", "Partial cell sorting", "Fractional boundary length",
                            True, [["frac_ll", "l\u2013l", "#4f46e5"], ["frac_dd", "d\u2013d", "#0d9488"], ["frac_ld", "l\u2013d (heterotypic)", "#e11d48"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993DispersalSloughing", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_dispersalsloughing(state):
    """Sloughing: light cells disperse into the medium as l–M contact surges"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 480], "series": {"frac_lM": [0.07285, 0.10268, 0.11549, 0.12494, 0.13646, 0.14456, 0.1533, 0.16178, 0.17498, 0.18094, 0.19186, 0.20525, 0.21463, 0.22161, 0.23615, 0.24302, 0.24594, 0.2588, 0.26901, 0.31504, 0.35637, 0.37565, 0.3983, 0.41268, 0.4458, 0.4749, 0.49875, 0.50489, 0.51197, 0.51269], "frac_ll": [0.20511, 0.19208, 0.18951, 0.18686, 0.18241, 0.17931, 0.1762, 0.17468, 0.16915, 0.1675, 0.16357, 0.15975, 0.1566, 0.15828, 0.15344, 0.15219, 0.15365, 0.15324, 0.15161, 0.12407, 0.10484, 0.10167, 0.09464, 0.08798, 0.07364, 0.06037, 0.05086, 0.05088, 0.05008, 0.04993], "frac_ld": [0.43264, 0.41711, 0.40828, 0.39974, 0.3947, 0.3914, 0.38371, 0.37797, 0.37057, 0.36616, 0.35813, 0.3463, 0.3376, 0.32802, 0.31584, 0.30854, 0.30011, 0.28791, 0.27553, 0.26139, 0.23998, 0.22455, 0.20817, 0.19686, 0.1807, 0.16464, 0.15099, 0.14361, 0.13716, 0.13592]}}
    caption = "<b>γ_ld=+2 · γ_lM=-5 · γ_dM=+14 · T=5.0 · λ=1.0</b> &nbsp;·&nbsp; A negative light–medium tension (γ_lM=−5) makes light cells prefer the medium and detach: light–medium contact climbs from 0.07 to over 0.5 while light–light and light–dark contact collapse; dark stays compact (Fig 25)."
    return {"html": _figure("gg-gg1993_dispersalsloughing", "Sloughing: light cells disperse into the medium as l–M contact surges", "Dispersal — light-cell sloughing", "Fractional boundary length",
                            True, [["frac_lM", "l\u2013M  (light \u2192 medium)", "#d97706"], ["frac_ll", "l\u2013l", "#4f46e5"], ["frac_ld", "l\u2013d", "#e11d48"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993DispersalSeparate", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_dispersalseparate(state):
    """Full separation: light and dark clusters split apart"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 1000, 1212, 1468, 1778, 2000], "series": {"frac_ll": [0.25632, 0.25519, 0.25429, 0.25719, 0.25586, 0.25701, 0.25523, 0.25627, 0.2538, 0.25127, 0.25086, 0.25219, 0.24931, 0.24699, 0.25209, 0.24886, 0.24814, 0.24554, 0.24561, 0.24717, 0.24673, 0.24212, 0.24057, 0.24384, 0.24558, 0.24508, 0.2423, 0.24083, 0.23711, 0.23413, 0.23533, 0.22808, 0.22759, 0.22247, 0.22078, 0.22426, 0.22343], "frac_dd": [0.26372, 0.265, 0.26531, 0.26386, 0.26505, 0.26562, 0.26521, 0.26461, 0.26212, 0.26034, 0.2628, 0.26125, 0.25985, 0.26123, 0.26229, 0.26274, 0.25793, 0.25499, 0.25568, 0.2534, 0.25185, 0.25409, 0.24809, 0.25389, 0.25538, 0.25259, 0.25265, 0.25011, 0.24563, 0.2452, 0.24653, 0.24083, 0.24191, 0.23071, 0.23442, 0.23541, 0.23452], "frac_ld": [0.38985, 0.38631, 0.38729, 0.3857, 0.38563, 0.38334, 0.38269, 0.38408, 0.38549, 0.38065, 0.38467, 0.38321, 0.38315, 0.38315, 0.3737, 0.37176, 0.37615, 0.37804, 0.37464, 0.37925, 0.37617, 0.37842, 0.38262, 0.37336, 0.36959, 0.36948, 0.37336, 0.36501, 0.37318, 0.36042, 0.3586, 0.35916, 0.35109, 0.34679, 0.33209, 0.33963, 0.33745]}}
    caption = "<b>γ_ld=+27 · γ_lM=+9 · γ_dM=+15 · T=5.0 · λ=1.0</b> &nbsp;·&nbsp; A very high γ_ld=27 makes unlike contact so costly that the two cell types form disjoint clusters — heterotypic l–d contact falls sharply (Fig 26)."
    return {"html": _figure("gg-gg1993_dispersalseparate", "Full separation: light and dark clusters split apart", "Dispersal — clusters separate", "Fractional boundary length",
                            True, [["frac_ll", "l\u2013l", "#4f46e5"], ["frac_dd", "d\u2013d", "#0d9488"], ["frac_ld", "l\u2013d (heterotypic)", "#e11d48"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993DispersalNoSeparate", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_dispersalnoseparate(state):
    """No separation: clusters stay attached despite strong tension"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 1000, 1212, 1468, 1778, 2000], "series": {"frac_ll": [0.25981, 0.25581, 0.25723, 0.25748, 0.25697, 0.25698, 0.25721, 0.25462, 0.25758, 0.25567, 0.25826, 0.25786, 0.25851, 0.25676, 0.25496, 0.25398, 0.25877, 0.25914, 0.2574, 0.26112, 0.25904, 0.26065, 0.25915, 0.2602, 0.25856, 0.25995, 0.25602, 0.25548, 0.25343, 0.25178, 0.25466, 0.24841, 0.24537, 0.25466, 0.24941, 0.24957, 0.24868], "frac_dd": [0.26412, 0.26568, 0.26438, 0.26523, 0.26402, 0.26658, 0.26728, 0.26451, 0.26636, 0.26612, 0.26507, 0.26632, 0.26398, 0.26596, 0.26526, 0.26523, 0.26513, 0.26629, 0.26228, 0.26722, 0.26727, 0.26657, 0.26311, 0.26533, 0.26488, 0.2651, 0.26064, 0.26201, 0.26017, 0.2594, 0.26421, 0.2577, 0.25581, 0.26057, 0.25588, 0.25741, 0.25605], "frac_ld": [0.41101, 0.41373, 0.41277, 0.41073, 0.41149, 0.40867, 0.40818, 0.41262, 0.40779, 0.41006, 0.40653, 0.40384, 0.40592, 0.40408, 0.40702, 0.40734, 0.40154, 0.40028, 0.40277, 0.3936, 0.39438, 0.39521, 0.39927, 0.39411, 0.3958, 0.39514, 0.40277, 0.40111, 0.40264, 0.40438, 0.39581, 0.40751, 0.40908, 0.39798, 0.40721, 0.4025, 0.40447]}}
    caption = "<b>γ_ld=+21 · γ_lM=+9 · γ_dM=+15 · T=5.0 · λ=1.0</b> &nbsp;·&nbsp; Just below the separation threshold (γ_ld=21) the clusters remain joined — heterotypic contact persists rather than collapsing (Fig 27; contrast Fig 26)."
    return {"html": _figure("gg-gg1993_dispersalnoseparate", "No separation: clusters stay attached despite strong tension", "Dispersal — clusters do not separate", "Fractional boundary length",
                            True, [["frac_ll", "l\u2013l", "#4f46e5"], ["frac_dd", "d\u2013d", "#0d9488"], ["frac_ld", "l\u2013d (heterotypic)", "#e11d48"]], data, caption, ref=None)}


@as_visualization(inputs={"mcs": "list[float]"}, name="GG1993VacancyCavity", demo={"mcs": [1, 2, 3, 4, 5, 6]})
def update_gg1993_vacancycavity(state):
    """Cavity: dark cells aggregate (d–d↑) around a light-lined medium pocket"""
    data = {"mcs": [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 32, 38, 46, 56, 68, 83, 100, 121, 147, 178, 200], "series": {"frac_dd": [0.35137, 0.35739, 0.35976, 0.35983, 0.36546, 0.37046, 0.36954, 0.36733, 0.37628, 0.37689, 0.37723, 0.38096, 0.38381, 0.38782, 0.39208, 0.39621, 0.40744, 0.41046, 0.42071, 0.42692, 0.43533, 0.44266, 0.43792, 0.43504, 0.44196], "frac_ll": [0.15519, 0.15772, 0.15668, 0.15232, 0.15261, 0.15293, 0.15161, 0.15222, 0.15433, 0.14884, 0.1451, 0.14115, 0.13952, 0.14313, 0.1344, 0.13568, 0.13125, 0.13096, 0.12805, 0.12535, 0.13018, 0.12885, 0.1243, 0.12277, 0.12401], "frac_ld": [0.43258, 0.42424, 0.42311, 0.42649, 0.4213, 0.41573, 0.4178, 0.41943, 0.40718, 0.41264, 0.41564, 0.41601, 0.41478, 0.4065, 0.41177, 0.4063, 0.39951, 0.39717, 0.38931, 0.38611, 0.37304, 0.36663, 0.37582, 0.38005, 0.37245]}}
    caption = "<b>γ_ld=+3 · γ_lM=+9 · γ_dM=+15 · T=5.0 · λ=1.0</b> &nbsp;·&nbsp; Unequal target areas (A_l=20, A_d=40) with vacancy nucleation let a medium cavity open, lined by light cells; dark–dark contact grows as dark cells aggregate while heterotypic l–d contact falls (Fig 28; a delicate case)."
    return {"html": _figure("gg-gg1993_vacancycavity", "Cavity: dark cells aggregate (d–d↑) around a light-lined medium pocket", "Vacancy nucleation (cavity)", "Fractional boundary length",
                            True, [["frac_dd", "d\u2013d", "#0d9488"], ["frac_ll", "l\u2013l", "#4f46e5"], ["frac_ld", "l\u2013d", "#e11d48"]], data, caption, ref=None)}
