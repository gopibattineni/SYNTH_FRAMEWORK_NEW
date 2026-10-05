#!/usr/bin/env python3
"""Final results: Greedy vs Hungarian Mahalanobis (9 classification + 6 regression).

Writes 5 focused figures + 3 tables under:
  mapping study graphs/final results/

Uses only paired generator×dataset units (both Greedy and Hungarian Mahalanobis present).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy import stats

BASE = Path(__file__).resolve().parent
SRC = BASE / "extracted_results" / "hungarian_vs_greedy_summary.csv"
OUT = BASE / "final results"

# ---------------------------------------------------------------------------
# Dataset taxonomy + characteristics (working-set sizes used in this study)
# ---------------------------------------------------------------------------
CLASSIFICATION = [
    "Cancer",
    "Alzheimers",
    "Adult",
    "ForestCover",
    "Bank",
    "Wine",
    "CDC",
    "Mushroom",
    "MAGIC",
]
REGRESSION = [
    "Metro",
    "OnlineShop",
    "AirQuality",
    "Concrete",
    "Energy",
    "RealEstate",
]
DS_ORDER = CLASSIFICATION + REGRESSION

DS_META = {
    # Dataset: label, type, samples, features, classes (None for regression)
    "Cancer": ("Cancer", "Classification", 569, 30, 2),
    "Alzheimers": ("Alzheimer's", "Classification", 373, 10, 2),
    "Adult": ("Adult", "Classification", 1000, 14, 2),
    "ForestCover": ("Forest Cover", "Classification", 1000, 10, 7),
    "Bank": ("Bank Marketing", "Classification", 10000, 13, 2),
    "Wine": ("Wine Quality", "Classification", 1000, 11, 6),
    "CDC": ("CDC Diabetes", "Classification", 1000, 21, 2),
    "Mushroom": ("Mushroom", "Classification", 1000, 20, 2),
    "MAGIC": ("MAGIC Gamma", "Classification", 19020, 10, 2),
    "Metro": ("Metro Interstate", "Regression", 1000, 12, None),
    "OnlineShop": ("Online Shopping", "Regression", 1000, 12, None),
    "AirQuality": ("Air Quality", "Regression", 1000, 12, None),
    "Concrete": ("Concrete", "Regression", 1000, 8, None),
    "Energy": ("Energy Efficiency", "Regression", 768, 8, None),
    "RealEstate": ("Real Estate", "Regression", 414, 5, None),
}

SHORT_LABEL = {
    "Cancer": "Cancer",
    "Alzheimers": "Alzh.",
    "Adult": "Adult",
    "ForestCover": "Forest",
    "Bank": "Bank",
    "Wine": "Wine",
    "CDC": "CDC",
    "Mushroom": "Mush.",
    "MAGIC": "MAGIC",
    "Metro": "Metro",
    "OnlineShop": "Shop",
    "AirQuality": "AirQ.",
    "Concrete": "Concrete",
    "Energy": "Energy",
    "RealEstate": "RE",
}

COLOR_GREEDY = "#2563eb"
COLOR_HUNG = "#c2410c"
COLOR_POS = "#1b7f5a"  # Hungarian better
COLOR_NEG = "#b23a48"  # Greedy better
COLOR_CLS = "#1d4ed8"
COLOR_REG = "#b45309"


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_paired() -> pd.DataFrame:
    raw = pd.read_csv(SRC)
    need = ["Dataset", "Generator", "Greedy_Mahalanobis", "Hungarian_Mahalanobis"]
    df = raw.dropna(subset=["Greedy_Mahalanobis", "Hungarian_Mahalanobis"]).copy()
    df = df[need].drop_duplicates(["Dataset", "Generator"], keep="first")
    df["Type"] = df["Dataset"].map(
        lambda d: "Classification" if d in CLASSIFICATION else "Regression"
    )
    df = df[df["Dataset"].isin(DS_ORDER)].copy()
    df["Greedy"] = df["Greedy_Mahalanobis"].astype(float)
    df["Hungarian"] = df["Hungarian_Mahalanobis"].astype(float)
    # Improvement %: positive ⇒ Hungarian better (lower distance)
    denom = df["Greedy"].abs()
    df["Improvement_Pct"] = np.where(
        denom > 1e-12, 100.0 * (df["Greedy"] - df["Hungarian"]) / denom, np.nan
    )
    return df


def dataset_summary(unit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in DS_ORDER:
        sub = unit[unit["Dataset"] == ds]
        if len(sub) == 0:
            continue
        label, typ, n, p, k = DS_META[ds]
        g = float(sub["Greedy"].mean())
        h = float(sub["Hungarian"].mean())
        imp = 100.0 * (g - h) / g if abs(g) > 1e-12 else np.nan
        better = (
            "Hungarian"
            if imp > 0.05
            else ("Greedy" if imp < -0.05 else "Tie")
        )
        rows.append(
            {
                "Dataset": ds,
                "Dataset_Label": label,
                "Short": SHORT_LABEL[ds],
                "Type": typ,
                "Samples": n,
                "Features": p,
                "Classes": k if k is not None else "–",
                "N_Generators": int(len(sub)),
                "Greedy": g,
                "Hungarian": h,
                "Improvement_Pct": imp,
                "Better_Method": better,
            }
        )
    return pd.DataFrame(rows)


def wilcoxon_block(a: np.ndarray, b: np.ndarray, name: str) -> dict:
    """Paired Wilcoxon: H1 that Greedy > Hungarian (Hungarian better)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    n = int(len(a))
    diff = a - b  # positive ⇒ Hungarian lower
    mean_diff = float(np.mean(diff)) if n else np.nan
    if n < 3 or np.allclose(a, b):
        return {
            "Comparison": name,
            "N": n,
            "Mean_Difference_Greedy_minus_Hungarian": mean_diff,
            "Effect_RankBiserial": np.nan,
            "Test": "Wilcoxon signed-rank",
            "p_value": np.nan,
            "Conclusion": "Insufficient variation / sample size",
        }
    try:
        res = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
        # Rank-biserial from Wilcoxon statistic
        # r = 2R / (n(n+1)) - 1 with R = statistic under greater
        s_max = n * (n + 1) / 2.0
        r_rb = (2.0 * float(res.statistic)) / s_max - 1.0 if s_max > 0 else np.nan
        p = float(res.pvalue)
    except ValueError:
        p, r_rb = np.nan, np.nan
    if np.isnan(p):
        concl = "Test not applicable"
    elif p < 0.05 and mean_diff > 0:
        concl = "Hungarian significantly better (lower distance)"
    elif p < 0.05 and mean_diff < 0:
        concl = "Greedy significantly better"
    else:
        concl = "No significant difference at α=0.05"
    return {
        "Comparison": name,
        "N": n,
        "Mean_Difference_Greedy_minus_Hungarian": mean_diff,
        "Effect_RankBiserial": r_rb,
        "Test": "Wilcoxon signed-rank (paired; H1: Greedy > Hungarian)",
        "p_value": p,
        "Conclusion": concl,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig1_overall(ds: pd.DataFrame) -> None:
    """Grouped bars: Greedy vs Hungarian, panels = Classification / Regression.

    Uses geometric mean across datasets (distances span orders of magnitude;
    arithmetic mean would be dominated by Energy / Real Estate / Cancer).
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=False)
    for ax, typ in zip(axes, ["Classification", "Regression"]):
        sub = ds[ds["Type"] == typ]
        gvals = sub["Greedy"].to_numpy(float)
        hvals = sub["Hungarian"].to_numpy(float)
        g_mean = float(np.exp(np.mean(np.log(gvals))))
        h_mean = float(np.exp(np.mean(np.log(hvals))))
        # Geometric SEM ≈ exp(mean_log) * SEM(log)
        g_sem = g_mean * float(np.std(np.log(gvals), ddof=1) / np.sqrt(len(gvals)))
        h_sem = h_mean * float(np.std(np.log(hvals), ddof=1) / np.sqrt(len(hvals)))
        means = [g_mean, h_mean]
        sems = [g_sem, h_sem]
        x = np.arange(2)
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            color=[COLOR_GREEDY, COLOR_HUNG],
            width=0.62,
            capsize=4,
            ecolor="#475569",
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Greedy", "Hungarian"])
        ax.set_title(f"{typ} (n={len(sub)} datasets)")
        ax.set_ylabel("Geometric mean Mahalanobis distance" if ax is axes[0] else "")
        ax.set_yscale("log")
        for b, v in zip(bars, means):
            ax.annotate(
                f"{v:.3g}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=8.5,
                xytext=(0, 3),
                textcoords="offset points",
            )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_xlim(-0.5, 1.5)

    fig.suptitle(
        "Figure 1. Overall Mahalanobis matching cost by method\n"
        "(geometric mean ± SEM across datasets; log scale)",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, OUT / "figures" / "Fig01_overall_comparison")


def fig2_heatmap(ds: pd.DataFrame) -> None:
    """Dataset × {Greedy, Hungarian} absolute mean distances."""
    mat = ds.set_index("Short")[["Greedy", "Hungarian"]]
    # Preserve class then regression order
    order = [SHORT_LABEL[d] for d in DS_ORDER if d in set(ds["Dataset"])]
    mat = mat.reindex(order)

    fig, ax = plt.subplots(figsize=(6.2, 8.0))
    vals = mat.to_numpy(dtype=float)
    vmin = max(np.nanmin(vals[vals > 0]), 1e-3)
    vmax = np.nanmax(vals)
    sns.heatmap(
        mat,
        ax=ax,
        annot=True,
        fmt=".3g",
        cmap="YlOrRd",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Mean Mahalanobis distance (log scale)"},
        annot_kws={"size": 8},
    )
    # Separator between classification and regression
    n_cls = sum(1 for d in DS_ORDER if d in CLASSIFICATION and d in set(ds["Dataset"]))
    ax.axhline(n_cls, color="#0f172a", lw=1.6)
    ax.annotate(
        "Classification",
        xy=(1.0, 1.0 - (n_cls / 2) / len(mat)),
        xycoords="axes fraction",
        xytext=(8, 0),
        textcoords="offset points",
        rotation=270,
        va="center",
        ha="left",
        fontsize=9,
        color=COLOR_CLS,
        annotation_clip=False,
    )
    ax.annotate(
        "Regression",
        xy=(1.0, 1.0 - (n_cls + (len(mat) - n_cls) / 2) / len(mat)),
        xycoords="axes fraction",
        xytext=(8, 0),
        textcoords="offset points",
        rotation=270,
        va="center",
        ha="left",
        fontsize=9,
        color=COLOR_REG,
        annotation_clip=False,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "Figure 2. Dataset-level Mahalanobis distance\n"
        "Greedy vs Hungarian (paired generators only)"
    )
    save_fig(fig, OUT / "figures" / "Fig02_dataset_heatmap")


def fig3_improvement(ds: pd.DataFrame) -> None:
    """Diverging bars of Hungarian improvement % per dataset."""
    plot = ds.sort_values("Improvement_Pct")
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    y = np.arange(len(plot))
    colors = [
        COLOR_POS if v > 0 else COLOR_NEG if v < 0 else "#94a3b8"
        for v in plot["Improvement_Pct"]
    ]
    ax.barh(y, plot["Improvement_Pct"], color=colors, height=0.72, edgecolor="white")
    ax.axvline(0, color="black", lw=1.0)
    labels = [
        f"{r.Short}  ({'C' if r.Type == 'Classification' else 'R'})"
        for r in plot.itertuples()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        "Improvement (%) = (Greedy − Hungarian) / Greedy × 100\n"
        "positive = Hungarian better (lower distance)"
    )
    ax.set_title("Figure 3. Hungarian improvement over Greedy by dataset")
    ax.grid(axis="x", alpha=0.3)
    for yi, v in zip(y, plot["Improvement_Pct"]):
        ha = "left" if v >= 0 else "right"
        offset = 0.4 if v >= 0 else -0.4
        ax.text(v + offset, yi, f"{v:.2f}%", va="center", ha=ha, fontsize=8)
    # Legend proxy
    ax.plot([], [], color=COLOR_POS, lw=6, label="Hungarian better")
    ax.plot([], [], color=COLOR_NEG, lw=6, label="Greedy better")
    ax.legend(loc="lower right", frameon=True)
    save_fig(fig, OUT / "figures" / "Fig03_hungarian_improvement")


def fig4_distributions(unit: pd.DataFrame) -> None:
    """Box + strip of log10 Mahalanobis by method × task type."""
    long = unit.melt(
        id_vars=["Dataset", "Type", "Generator"],
        value_vars=["Greedy", "Hungarian"],
        var_name="Method",
        value_name="Distance",
    )
    long = long[long["Distance"] > 0].copy()
    long["log10_Distance"] = np.log10(long["Distance"])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    for ax, typ in zip(axes, ["Classification", "Regression"]):
        sub = long[long["Type"] == typ]
        sns.violinplot(
            data=sub,
            x="Method",
            y="log10_Distance",
            hue="Method",
            order=["Greedy", "Hungarian"],
            hue_order=["Greedy", "Hungarian"],
            palette={"Greedy": COLOR_GREEDY, "Hungarian": COLOR_HUNG},
            inner=None,
            cut=0,
            ax=ax,
            alpha=0.35,
            legend=False,
        )
        sns.boxplot(
            data=sub,
            x="Method",
            y="log10_Distance",
            order=["Greedy", "Hungarian"],
            width=0.28,
            showfliers=False,
            boxprops={"facecolor": "white", "alpha": 0.9},
            ax=ax,
        )
        sns.stripplot(
            data=sub,
            x="Method",
            y="log10_Distance",
            order=["Greedy", "Hungarian"],
            color="#334155",
            size=3.0,
            alpha=0.45,
            jitter=0.12,
            ax=ax,
        )
        ax.set_title(f"{typ} (n={len(sub)} units)")
        ax.set_xlabel("")
        ax.set_ylabel("log₁₀ Mahalanobis distance" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Figure 4. Distribution of Mahalanobis distances across dataset×generator units",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    save_fig(fig, OUT / "figures" / "Fig04_distance_distributions")


def fig5_characteristics(ds: pd.DataFrame) -> None:
    """Scatter: features vs Hungarian improvement %; size ~ samples."""
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for typ, color in [("Classification", COLOR_CLS), ("Regression", COLOR_REG)]:
        sub = ds[ds["Type"] == typ]
        sizes = 40 + 180 * (np.log10(sub["Samples"]) - np.log10(sub["Samples"].min())) / max(
            np.log10(sub["Samples"].max()) - np.log10(sub["Samples"].min()), 1e-9
        )
        ax.scatter(
            sub["Features"],
            sub["Improvement_Pct"],
            s=sizes,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
            label=typ,
            zorder=3,
        )
        for r in sub.itertuples():
            ax.annotate(
                r.Short,
                (r.Features, r.Improvement_Pct),
                textcoords="offset points",
                xytext=(5, 4),
                fontsize=8,
                color="#334155",
            )

    # Simple trend (all datasets)
    x = ds["Features"].to_numpy(float)
    y = ds["Improvement_Pct"].to_numpy(float)
    if len(ds) >= 3 and np.nanstd(x) > 0:
        slope, intercept, r, p, _ = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xx,
            intercept + slope * xx,
            color="#64748b",
            ls="--",
            lw=1.2,
            label=f"Linear trend (r={r:.2f}, p={p:.3f})",
            zorder=2,
        )

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Number of features (excluding target)")
    ax.set_ylabel("Hungarian improvement over Greedy (%)")
    ax.set_title(
        "Figure 5. When does Hungarian help most?\n"
        "Improvement vs dataset dimensionality (marker size ∝ sample size)"
    )
    ax.legend(frameon=True, loc="best")
    ax.grid(alpha=0.3)
    save_fig(fig, OUT / "figures" / "Fig05_improvement_vs_features")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_tables(ds: pd.DataFrame, unit: pd.DataFrame) -> None:
    tdir = OUT / "tables"
    tdir.mkdir(parents=True, exist_ok=True)

    # Table 1 — characteristics
    t1 = pd.DataFrame(
        [
            {
                "Dataset": DS_META[d][0],
                "Type": DS_META[d][1],
                "Samples": DS_META[d][2],
                "Features": DS_META[d][3],
                "Classes": DS_META[d][4] if DS_META[d][4] is not None else "–",
                "Task": DS_META[d][1],
            }
            for d in DS_ORDER
        ]
    )
    t1.to_csv(tdir / "Table1_dataset_characteristics.csv", index=False)

    # Table 2 — main quantitative results
    t2 = ds[
        [
            "Dataset_Label",
            "Type",
            "N_Generators",
            "Greedy",
            "Hungarian",
            "Improvement_Pct",
            "Better_Method",
        ]
    ].rename(
        columns={
            "Dataset_Label": "Dataset",
            "N_Generators": "N_paired_generators",
            "Greedy": "Greedy_Mahalanobis",
            "Hungarian": "Hungarian_Mahalanobis",
        }
    )
    t2.to_csv(tdir / "Table2_main_quantitative_results.csv", index=False)

    # Table 3 — statistical tests on dataset-level paired means
    blocks = []
    for typ in ["Classification", "Regression"]:
        sub = ds[ds["Type"] == typ]
        blocks.append(wilcoxon_block(sub["Greedy"].to_numpy(), sub["Hungarian"].to_numpy(), f"Hungarian vs Greedy – {typ}"))
    blocks.append(
        wilcoxon_block(ds["Greedy"].to_numpy(), ds["Hungarian"].to_numpy(), "Overall (15 datasets)")
    )
    # Also unit-level overall for power
    blocks.append(
        wilcoxon_block(
            unit["Greedy"].to_numpy(),
            unit["Hungarian"].to_numpy(),
            "Overall (dataset×generator units)",
        )
    )
    t3 = pd.DataFrame(blocks)
    t3.to_csv(tdir / "Table3_statistical_significance.csv", index=False)

    # Excel workbook
    with pd.ExcelWriter(tdir / "final_results_tables.xlsx") as writer:
        t1.to_excel(writer, sheet_name="Table1_characteristics", index=False)
        t2.to_excel(writer, sheet_name="Table2_main_results", index=False)
        t3.to_excel(writer, sheet_name="Table3_statistics", index=False)
        ds.to_excel(writer, sheet_name="dataset_level_detail", index=False)
        unit.to_excel(writer, sheet_name="unit_level_paired", index=False)

    # Pretty markdown
    md = []
    md.append("# Final results — Greedy vs Hungarian (Mahalanobis)\n")
    md.append("Paired units only (both methods present). Improvement % > 0 ⇒ Hungarian better.\n")
    md.append("## Table 1. Dataset characteristics\n")
    md.append(t1.to_markdown(index=False))
    md.append("\n## Table 2. Main quantitative results\n")
    show2 = t2.copy()
    show2["Greedy_Mahalanobis"] = show2["Greedy_Mahalanobis"].map(lambda v: f"{v:.4g}")
    show2["Hungarian_Mahalanobis"] = show2["Hungarian_Mahalanobis"].map(lambda v: f"{v:.4g}")
    show2["Improvement_Pct"] = show2["Improvement_Pct"].map(lambda v: f"{v:.2f}")
    md.append(show2.to_markdown(index=False))
    md.append("\n## Table 3. Statistical significance\n")
    show3 = t3.copy()
    show3["Mean_Difference_Greedy_minus_Hungarian"] = show3[
        "Mean_Difference_Greedy_minus_Hungarian"
    ].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    show3["Effect_RankBiserial"] = show3["Effect_RankBiserial"].map(
        lambda v: f"{v:.3f}" if pd.notna(v) else ""
    )
    show3["p_value"] = show3["p_value"].map(lambda v: f"{v:.3e}" if pd.notna(v) else "")
    md.append(show3.to_markdown(index=False))
    (tdir / "README_tables.md").write_text("\n".join(md), encoding="utf-8")


def write_readme(ds: pd.DataFrame, unit: pd.DataFrame) -> None:
    n_cls = int((ds["Type"] == "Classification").sum())
    n_reg = int((ds["Type"] == "Regression").sum())
    pct_h = 100.0 * (ds["Improvement_Pct"] > 0).mean()
    text = f"""# Final results — Greedy vs Hungarian Mahalanobis mapping

Focused publication set: **5 figures + 3 tables**.

## Scope
- **9 classification** + **6 regression** datasets ({n_cls}+{n_reg} with paired data)
- Metric: mean matched-pair **Mahalanobis distance** (lower is better)
- Methods: **One-to-One Greedy** vs **Hungarian**
- Only generator×dataset units where **both** methods exist (n={len(unit)} units)
- Dataset-level means average over paired generators for that dataset

## Improvement definition
```
Improvement(%) = (D_Greedy − D_Hungarian) / D_Greedy × 100
  > 0  → Hungarian better
  < 0  → Greedy better
```

## Figures
| File | Description |
|------|-------------|
| `figures/Fig01_overall_comparison` | Grouped bars, Classification vs Regression panels (log y) |
| `figures/Fig02_dataset_heatmap` | Dataset × {{Greedy, Hungarian}} absolute distances (log colour) |
| `figures/Fig03_hungarian_improvement` | Diverging improvement (%) for all 15 datasets |
| `figures/Fig04_distance_distributions` | Violin/box/strip of log₁₀ distances by method × task |
| `figures/Fig05_improvement_vs_features` | Improvement vs #features (size ∝ samples) |

## Tables
| File | Description |
|------|-------------|
| `tables/Table1_dataset_characteristics.csv` | Samples, features, classes, task |
| `tables/Table2_main_quantitative_results.csv` | Greedy, Hungarian, Improvement %, better method |
| `tables/Table3_statistical_significance.csv` | Wilcoxon signed-rank + rank-biserial |
| `tables/final_results_tables.xlsx` | All tables + detail sheets |

## Headline
- Datasets where Hungarian improves mean distance: **{pct_h:.0f}%** ({int((ds['Improvement_Pct']>0).sum())}/{len(ds)})
- Source summary: `{SRC}`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "tables").mkdir(exist_ok=True)

    unit = load_paired()
    ds = dataset_summary(unit)
    print(f"Paired units: {len(unit)}")
    print(f"Datasets: {len(ds)} (classification={sum(ds.Type=='Classification')}, regression={sum(ds.Type=='Regression')})")

    fig1_overall(ds)
    fig2_heatmap(ds)
    fig3_improvement(ds)
    fig4_distributions(unit)
    fig5_characteristics(ds)
    write_tables(ds, unit)
    write_readme(ds, unit)

    print(f"Wrote figures + tables under: {OUT}")


if __name__ == "__main__":
    main()
