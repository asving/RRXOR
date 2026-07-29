import json, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = os.path.expanduser("~/compositionality/figs")
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "lines.linewidth": 1.6})
C = ["#2166ac", "#b2182b", "#e08214", "#1b7837", "#762a83"]

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{FIG}/{name}.pdf"); plt.close(fig); print(name)

# ---- B1: transience ----
def comp_traj(path):
    steps, comp = [], []
    for l in open(path):
        m = re.search(r"step\s+(\d+) .*model_ho ([\d.]+) oracle_ho\* ([\d.]+) restr ([\d.]+)", l)
        if m:
            s, mh, oh, rs = float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))
            steps.append(s); comp.append((rs - mh) / (rs - oh))
    return steps, comp
fig, ax = plt.subplots(figsize=(3.6, 2.4))
for i, (eps, lab) in enumerate((("0.8", "approximate coupling ($\\epsilon$=0.8)"), ("1.0", "exact coupling ($\\epsilon$=1.0)"))):
    s, c = comp_traj(os.path.expanduser(f"~/comp_icl/runs/traj/eps{eps}.log"))
    ax.plot(s, c, color=C[i], label=lab)
ax.axhline(0, color="#999", lw=0.7)
ax.set_xlabel("training step"); ax.set_ylabel("composition score")
ax.legend(fontsize=7, frameon=False)
save(fig, "b1_transience")

# ---- B2: conflict ----
rows = []
for l in open(os.path.expanduser("~/group_comp/logs/conflict_lam_20260627_004051.log")):
    m = re.match(r"\s+([\d.]+) \|\s+([+-][\d.]+)±[\d.]+ \|\s+([\d.]+)±[\d.]+ \|\s+[\d.]+ \|\s+([+-][\d.]+)±([\d.]+)", l)
    if m: rows.append([float(m.group(k)) for k in (1, 2, 3, 4)])
rows = np.array(rows)
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.plot(rows[:, 0], rows[:, 1], "o-", color=C[0], label="gradient alignment of the two tasks")
ax.plot(rows[:, 0], rows[:, 3], "s-", color=C[1], label="unit overlap in the free net (Jaccard)")
ax.plot(rows[:, 0], rows[:, 2], "^-", color=C[2], label="error of a forced shared unit (MSE)")
ax.set_xlabel("conflict dial $\\lambda$ (0 = same use, 1 = opposite sign)")
ax.set_ylabel("per-series value (cosine / Jaccard / norm. MSE)", fontsize=7.5)
ax.legend(fontsize=6.5, frameon=False); ax.axhline(0, color="#999", lw=0.7)
save(fig, "b2_conflict")

# ---- B3: drive law + reuse selection ----
taus = {}
for l in open(os.path.expanduser("~/group_comp/logs/gate2dfreq_grid_20260627_032330.log")):
    m = re.match(r"\s+([\d.]+)\s+([\d.]+)\s+(\d) \|\s+(\d+)\s+(\d+)", l)
    if m:
        pC, pD, sd, tC, tD = [float(m.group(k)) for k in range(1, 6)]
        if pC == 1.0: taus.setdefault(pD, []).append(tD)
tables, cur = [], None
for l in open(os.path.expanduser("~/group_comp/logs/repgauge_20260627_001114.log")):
    if re.match(r"\s+T \|", l):
        cur = []; tables.append(cur); continue
    m = re.match(r"\s+(\d+) \|\s+[\d.]+ [\d.]+±[\d.]+ [\d.]+±[\d.]+ ([+-][\d.]+)±", l)
    if m and cur is not None:
        cur.append((int(m.group(1)), float(m.group(2))))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.3))
ph = sorted(taus)
a1.loglog(ph, [np.mean(taus[p]) for p in ph], "o-", color=C[0], label="measured (2-seed mean)")
xs = np.array(ph); a1.loglog(xs, 220 / xs, "--", color="#888", lw=1, label="$\\tau \\propto 1/\\varphi$")
a1.set_xlabel("use frequency $\\varphi$"); a1.set_ylabel("steps to circuit formation")
a1.legend(fontsize=7, frameon=False)
for tb, col, lab in ((tables[0], C[1], "main"), (tables[1] if len(tables) > 1 else [], "#999", "symmetric control")):
    if tb:
        a2.semilogx([t for t, _ in tb], [f for _, f in tb], "o-", color=col, base=2, label=lab)
a2.set_xlabel("tasks reusing the shared view, $T$")
a2.set_ylabel("reliance flip (shared $-$ private)")
a2.axhline(0, color="#999", lw=0.7); a2.legend(fontsize=7, frameon=False)
save(fig, "b3_drive")

# ---- degree ladder ----
tau = {1: [125, 150], 2: [650, 525], 3: [2250, 2250]}
snr = {1: 0.162, 2: 0.063, 3: 0.013}
fig, ax = plt.subplots(figsize=(3.3, 2.4))
x = [1 / snr[d] for d in (1, 2, 3)]
y = [np.mean(tau[d]) for d in (1, 2, 3)]
ax.loglog(x, y, "o", color=C[0], ms=7)
for d in (1, 2, 3):
    ax.loglog([1/snr[d]]*2, tau[d], "o", color=C[0], ms=3, mfc="none")
for d, xi, yi in zip((1, 2, 3), x, y):
    ax.annotate(f"degree {d}", (xi, yi), textcoords="offset points", xytext=(8, -3), fontsize=7.5)
xs = np.logspace(0.6, 2.1, 10)
ax.loglog(xs, 24 * xs, "--", color="#888", lw=1, label="$\\tau = c/\\mathrm{SNR}$")
ax.set_xlabel("1 / SNR of the teaching signal at init"); ax.set_ylabel("$\\tau_{95}$ (steps)")
ax.legend(fontsize=7.5, frameon=False)
save(fig, "degree_ladder")

# ---- rrxor formation (ce + pxor) ----
rows = json.load(open("rrxor_pxor.json"))
rows = [r for r in rows if r["step"] <= 4000]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.6, 3.2), sharex=True)
lab = {0: "free bit 1", 1: "free bit 2", 2: "xor position"}
for cl in (0, 1, 2):
    a1.plot([r["step"] for r in rows], [r[f"ce{cl}"] for r in rows], color=C[cl], label=lab[cl])
    a2.plot([r["step"] for r in rows], [r[f"px{cl}"] for r in rows], color=C[cl])
a1.axhline(np.log(2), color="#999", lw=0.7, ls=":")
a1.text(3800, np.log(2)+0.02, "$\\ln 2$", fontsize=6.5, color="#777", ha="right")
a1.set_ylabel("CE (nats)")
a1.legend(fontsize=7, frameon=False)
a2.axhline(0.5, color="#999", lw=0.7, ls=":")
a2.text(3800, 0.52, "chance", fontsize=6.5, color="#777", ha="right")
a2.set_ylabel("P(output = xor of prev two)"); a2.set_xlabel("training step")
save(fig, "rrxor_formation")

# ---- m27 cascade ----
rows = [json.loads(l) for l in open("runs/m27_nl6_seed1/eval.jsonl")]
keys = rows[0].keys()
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.9, 3.4), sharex=True)
CM = ["#1b7837", "#762a83", "#8c510a"]
for i, cl in enumerate(("xor", "xnor", "flip")):
    if f"ce_{cl}" in keys:
        a1.semilogx([r["step"] + 1 for r in rows], [r[f"ce_{cl}"] for r in rows], color=CM[i], label=cl)
        a2.semilogx([r["step"] + 1 for r in rows], [r[f"pxor_{cl}"] for r in rows], color=CM[i])
a1.set_ylabel("CE (nats)"); a1.legend(fontsize=7.5, frameon=False)
a2.axhline(0.5, color="#999", lw=0.7, ls=":")
a2.text(1.5, 0.52, "chance", fontsize=6.5, color="#777")
a2.set_ylabel("P(output = xor)"); a2.set_xlabel("training step (log; step 0 plotted at 1)")
save(fig, "m27_cascade")

# ---- race ----
fig, ax = plt.subplots(figsize=(3.6, 2.4))
lab = {"A": "full 36-state simplex", "B1": "xor circuit", "B2": "phase circuit"}
for i, r in enumerate(("B1", "B2", "A")):
    for s in (0, 1):
        rows = [json.loads(l) for l in open(f"runs/route_rrxor_{r}_s{s}/eval.jsonl")]
        ax.plot([x["step"] for x in rows], [x["acc"] for x in rows], color=C[i],
                alpha=1.0 if s == 0 else 0.45, label=lab[r] if s == 0 else None)
ax.axhline(0.95, color="#999", lw=0.7, ls=":")
ax.set_xlim(0, 4000); ax.set_xlabel("training step"); ax.set_ylabel("accuracy on route targets")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
save(fig, "race")

# ---- demand/supply ----
rows = json.load(open("m27_parts.json"))
fig, ax = plt.subplots(figsize=(3.9, 2.5))
st = [r["step"] + 1 for r in rows]
ax.loglog(st, [max(r["dem_u_L4"], 1e-4) for r in rows], "o-", color=C[0], ms=3, label="demand for $u$ (xor value)")
ax.loglog(st, [max(r["dem_phase9_L4"], 1e-4) for r in rows], "s-", color=C[1], ms=3, label="demand for block index (mod 9)")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.semilogx(st, [r["sup_u"] for r in rows], "--", color=C[3], lw=1.2, label="supply of $u$ (probe $R^2$)")
ax.set_xlabel("training step (log; step 0 plotted at 1)"); ax.set_ylabel("demand (normalized drift$^2$, log)")
ax2.set_ylabel("supply", color=C[3])
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="center left")
save(fig, "demand_supply")

# ---- decoherence ----
rows = []
for l in open(os.path.expanduser("~/comp_icl/runs/metrics/credit_traj_full.log")):
    m = re.match(r"\s+(\d+)\s+([+-][\d.]+)\s+([+-][\d.]+) \|\s+([+-][\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)", l)
    if m: rows.append([float(m.group(k)) for k in (1, 2, 7)])
rows = np.array([r for r in rows if r[0] >= 100])
fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax.plot(rows[:, 0], rows[:, 1], "o-", color=C[0], ms=3, label="held-out composition score")
ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
ax2.plot(rows[:, 0], rows[:, 2], "s-", color=C[1], ms=3, lw=1.2, label="binding-credit coherence")
ax.set_xlabel("training step"); ax.set_ylabel("composition", color=C[0]); ax2.set_ylabel("binding-credit coherence", color=C[1])
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False)
save(fig, "decoherence")

# ---- selection (crossover) ----
fig, ax = plt.subplots(figsize=(3.6, 2.4))
for s in (0, 1):
    rows = [json.loads(l) for l in open(f"runs/r2_lx_m1x2_s{s}_x1500/eval.jsonl")]
    ax.plot([r["step"] for r in rows], [r["pref1"] for r in rows], color=C[s], label=f"seed {s}")
ax.axvspan(0, 1500, color="#eee")
ax.text(700, 0.85, "lookup route\nmasked;\nxor owns the slot", fontsize=7, ha="center")
ax.text(2700, 0.1, "lookup enabled", fontsize=7)
ax.set_xlabel("training step"); ax.set_ylabel("P(shared answer follows lookup)")
ax.legend(fontsize=7.5, frameon=False, loc="center right")
save(fig, "selection")
