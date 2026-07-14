"""Build self-contained HTML galleries from the reproduced figures:

* per-study ``workspace/studies/<slug>/viz/figures.html`` (referenced by each
  study's ``visualizations`` entry, served by the workbench), and
* the per-investigation report
  ``workspace/reports/glazier-graner-1993/index.html`` — a full gallery of every
  study and figure with the paper's exact parameters.

Figures are base64-embedded so the pages are fully portable.
"""

from __future__ import annotations

import base64
import glob
import os

import yaml

from . import run as runmod
from .params import STUDIES, ORDER

WS = runmod.WS_ROOT
FIG_ROOT = os.path.join(WS, "workspace", "gg1993_figures")
STU_DIR = os.path.join(WS, "workspace", "studies")
INV = "glazier-graner-1993"
REPORT_DIR = os.path.join(WS, "workspace", "reports", INV)

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 6px} h2{font-size:20px;border-bottom:1px solid #2a2f3a;padding-bottom:6px;margin-top:38px}
h3{font-size:15px;color:#9ecbff;margin:22px 0 8px}
.sub{color:#9aa4b2;font-size:14px;margin:0 0 18px}
.params{background:#161a22;border:1px solid #232a36;border-radius:8px;padding:10px 14px;font:13px/1.5 ui-monospace,Menlo,monospace;color:#c7d0dc;overflow-x:auto}
.claim{background:#12261a;border-left:3px solid #3fb950;padding:8px 12px;border-radius:4px;margin:12px 0;font-size:14px}
.fig{margin:16px 0 26px}
.fig img{max-width:100%;background:#fff;border-radius:6px;border:1px solid #232a36}
.cap{color:#9aa4b2;font-size:13px;margin-top:6px}
.toc{columns:2;font-size:14px} .toc a{color:#9ecbff;text-decoration:none;display:block;padding:2px 0}
.tag{display:inline-block;background:#1f2530;color:#9ecbff;border-radius:10px;padding:1px 9px;font-size:12px;margin-left:8px}
a.top{color:#5a6472;font-size:12px;margin-left:10px;text-decoration:none}
"""

_CAPTIONS = {
    "fig2_annealing_topology": "Fig 2 — bulk/total ⟨n⟩ and moments μ₂,μ₃,μ₄ vs MCS during T=0 annealing.",
    "fig3_annealing_walls": "Fig 3 — cell-wall detail: unannealed, 2 MCS, 20 MCS of T=0 annealing.",
    "fig4_equilibration_walls": "Fig 4 — rectangular tiling (0 MCS) rounding to a disk (400 MCS).",
    "fig5_equilibration_stats": "Fig 5 — total boundary length, moments, light-Medium fraction vs MCS.",
    "fig7_patterns": "Fig 7 — checkerboard pattern at 10/100/1000/2000 MCS.",
    "fig8_checkerboard_stats": "Fig 8 — total & fractional lengths, ⟨n⟩, moments.",
    "fig9_checkerboard_temperature": "Fig 9 — l-l, l-d, l-M interfaces vs MCS across T.",
    "table1_checkerboard_moments": "Table I — bulk moments vs temperature.",
    "fig12_patterns": "Fig 12 — cell sorting at 0→13500 MCS.",
    "fig13_sorting_lengths": "Fig 13 — total, medium-contact, cell-cell fractions, correlations.",
    "fig14_sorting_topology": "Fig 14 — ⟨n⟩ and moments.",
    "fig15_sorting_temperature": "Fig 15 — l-d, l-M, total length vs MCS across T.",
    "fig16_sorting_lambda": "Fig 16 — total, l-d, d-M vs MCS across area constraint λ.",
    "table2_sorting_moments_T": "Table II — bulk moments vs temperature.",
    "table3_sorting_moments_lambda": "Table III — bulk moments vs λ.",
    "fig18_patterns": "Fig 18 — engulfment at 0/1000/5000/10000 MCS.",
    "fig19_engulfment_lengths": "Fig 19 — homotypic and heterotypic fractional lengths.",
    "fig20_patterns": "Fig 20 — position reversal at 0/50/5000 MCS.",
    "fig21_reversal_lengths": "Fig 21 — fractional lengths and medium correlation.",
    "fig22_patterns": "Fig 22 — partial cell sorting at 10/100/1000/2000 MCS.",
    "fig23_partial_lengths": "Fig 23 — cell-cell and medium fractional lengths.",
    "fig24_partial_vs_normal": "Fig 24 — partial vs normal sorting comparison.",
    "fig25_pattern": "Fig 25 — light-cell sloughing/dispersal at 480 MCS.",
    "fig26_pattern": "Fig 26 — dispersal: clusters separate (2000 MCS).",
    "fig27_pattern": "Fig 27 — dispersal: clusters do not separate (2000 MCS).",
    "fig28_pattern": "Fig 28 — vacancy nucleation / cavity (200 MCS).",
}


def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _fig_order(slug):
    """Ordered list of figure PNGs for a study (paper order)."""
    files = sorted(glob.glob(os.path.join(FIG_ROOT, slug, "*.png")))
    # sort by figure number embedded in filename
    def key(p):
        base = os.path.basename(p)
        import re
        m = re.search(r"(fig|table)(\d+)", base)
        return (0 if base.startswith("fig") else 1, int(m.group(2)) if m else 99, base)
    return sorted(files, key=key)


def _params_block(slug):
    s = STUDIES[slug]; J = s["J"]; g = s["gamma"]
    return (f"J_ll={J['J_ll']}  J_dd={J['J_dd']}  J_ld={J['J_ld']}  "
            f"J_lM={J['J_lM']}  J_dM={J['J_dM']}\n"
            f"γ_ld={g['gamma_ld']:+g}  γ_lM={g['gamma_lM']:+g}  γ_dM={g['gamma_dM']:+g}\n"
            f"T={s['temperature']}   λ={s['lam']}   "
            f"target: A_light={s['target'][2]}  A_dark={s['target'][1]}\n"
            f"initial condition: {s['ic']}")


def _study_meta(slug):
    p = os.path.join(STU_DIR, slug, "study.yaml")
    if os.path.exists(p):
        return yaml.safe_load(open(p))
    return {}


def _study_section(slug, embed=True):
    meta = _study_meta(slug)
    title = meta.get("title", slug)
    figs = _fig_order(slug)
    parts = [f'<div class="params">{_params_block(slug)}</div>']
    if meta.get("claim"):
        parts.append(f'<div class="claim"><b>Finding:</b> {meta["claim"]}</div>')
    if not figs:
        parts.append('<p class="cap">(figures pending)</p>')
    for f in figs:
        name = os.path.splitext(os.path.basename(f))[0]
        cap = _CAPTIONS.get(name, name)
        src = ("data:image/png;base64," + _b64(f)) if embed else os.path.relpath(f, os.path.dirname(f))
        parts.append(f'<div class="fig"><img src="{src}"><div class="cap">{cap}</div></div>')
    return title, "\n".join(parts)


def build_study_page(slug):
    title, body = _study_section(slug)
    html = (f"<!doctype html><meta charset=utf-8><title>{title}</title>"
            f"<style>{_CSS}</style><div class=wrap><h1>{title}</h1>"
            f'<p class="sub">Glazier &amp; Graner 1993 reproduction — pbg-cpm</p>{body}</div>')
    out = os.path.join(STU_DIR, slug, "viz", "figures.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(html)
    return out


def build_investigation_report():
    secs, toc = [], []
    for slug in ORDER:
        title, body = _study_section(slug)
        secs.append(f'<h2 id="{slug}">{title}<span class="tag">{slug}</span>'
                    f'<a class="top" href="#top">↑ top</a></h2>{body}')
        toc.append(f'<a href="#{slug}">{title}</a>')
    head = (f'<h1 id="top">Glazier &amp; Graner 1993 — full reproduction</h1>'
            f'<p class="sub">Simulation of differential-adhesion-driven cell rearrangement '
            f'(Phys. Rev. E 47, 2128). 11 studies · pbg-cpm Cellular Potts engine · '
            f'exact energies, initial conditions and MCS convention.</p>'
            f'<div class="toc">{"".join(toc)}</div>')
    html = (f"<!doctype html><meta charset=utf-8><title>GG1993 — pbg-cpm reproduction</title>"
            f"<style>{_CSS}</style><div class=wrap>{head}{''.join(secs)}</div>")
    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, "index.html")
    open(out, "w").write(html)
    return out


def build_all():
    pages = [build_study_page(s) for s in ORDER]
    report = build_investigation_report()
    print(f"{len(pages)} study pages + investigation report -> {report}")
    return report


if __name__ == "__main__":
    build_all()
