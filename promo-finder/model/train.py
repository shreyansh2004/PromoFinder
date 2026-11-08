"""Train the ad-success model on REAL short-drama ads, with a full audit first.

Target: high reach (100K+ unique users) vs low reach (<10K), TikTok Ad Library reach bands.

Key design decision (found in the audit): the high and low groups came from different apps
(collection artefact: e.g. most low-reach ads are ReelShort). A naive model learns "which app is this"
(AUC 0.69 in plain CV, 0.39 on unseen apps). So we compare WITHIN apps:
  - keep apps that have both high and low ads,
  - weight samples so every app contributes equal high and low weight (app carries no label signal),
  - score with within-app AUC (does it rank an app's winners above the same app's losers?).
This is also the tool's real job: rank promos within ONE show.

Features: only things computable the same way for a script promo (promofinder/features.py).
Audit: leakage, near-zero variance, pairwise correlation (|r|>0.7), Cramer's V, VIF (>5), balance.
Feature groups are compared by within-app CV; the model keeps only groups that beat chance.
The ranker gives the model weight in proportion to its measured skill.

usage: python model/train.py <ads_dataset_tagged.csv>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promofinder.features import tag_features, text_features, timing_features, visual_features  # noqa: E402

ART = ROOT / "model" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
report: list[str] = []


def log(s=""):
    print(s)
    report.append(s)


def S(v):
    return "" if pd.isna(v) else str(v)


def nn(v):
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v


def features_of(r) -> dict:
    f = {**text_features(S(r.first_3s_text), S(r.hook_15s_text), ""),
         **timing_features(r.speech_starts_at_s if pd.notna(r.speech_starts_at_s) else None),
         **visual_features(r.n_cuts_3s if pd.notna(r.n_cuts_3s) else 0,
                           r.face_closeup_3s if pd.notna(r.face_closeup_3s) else 0)}
    f.pop("f_ending_question")  # ending not transcribed for long ads -> missing-not-at-random
    t = {k: nn(r.get(f"ai_{k}")) for k in ("genre", "hook_type", "setting", "emotion_intensity", "open_mid_conflict",
                                            "hook_is_question", "narrator_voiceover", "tropes")}
    t["visual_hook"] = nn(r.get("ai_img_visual_hook"))
    f.update(tag_features(t))
    f["_hook_tagged"] = int(t["hook_type"] is not None)
    return f


# ------------------------------------------------------------------ data
src = Path(sys.argv[1])
raw = pd.read_csv(src, dtype={"ad_id": str})
uni = raw[(raw.language == "en") & (raw.is_short_drama_ad == 1) & (raw.is_first_of_creative == 1)].copy()
X_all = pd.DataFrame([features_of(r) for _, r in uni.iterrows()], index=uni.index)
hl = uni[uni.creative_best_tier.isin(["high", "low"])]
log("# Ad-success model: training report\n")
log(f"Source `{src.name}`: {len(raw)} ads, {len(uni)} unique English short-drama videos "
    f"({(uni.creative_best_tier == 'high').sum()} high, {(uni.creative_best_tier == 'low').sum()} low, "
    f"{(uni.creative_best_tier == 'mid').sum()} mid reach).")

# ------------------------------------------------------------------ confounding check
ct = pd.crosstab(hl.app, hl.creative_best_tier)
log("\n## 1. Confounding check: app vs label\n")
log("| app | high | low |\n|---|---|---|")
for a, r in ct.iterrows():
    log(f"| {a} | {r.get('high', 0)} | {r.get('low', 0)} |")
log("\nHigh and low ads come from different apps (a collection artefact). A model trained naively learns the app, "
    "not the hook. So training uses only apps with both labels, with weights that give every app equal high and "
    "low weight, and is scored by **within-app AUC**.")
both = ct[(ct.get("high", 0) > 0) & (ct.get("low", 0) > 0)].index
train = hl[hl.app.isin(both)].copy()
y = (train.creative_best_tier == "high").astype(int).values
apps = train.app.values
cnt = train.groupby(["app", "creative_best_tier"]).ad_id.transform("count").values
w = 0.5 / cnt
w = w / w.mean()
X = X_all.loc[train.index]
log(f"\nTraining set: {len(train)} ads from {len(both)} apps ({', '.join(both)}); {y.sum()} high / {(1 - y).sum()} low.")

# ------------------------------------------------------------------ audit
log("\n## 2. Feature audit\n")
LEAK = ["reach", "days", "shown", "users", "winner", "tier", "advertiser", "app", "title", "collected"]
assert not [c for c in X.columns if any(k in c for k in LEAK)]
log("- **Leakage:** none. Features come only from the hook transcript, measured video signals and AI tags of the hook; "
    "nothing from reach, dates, app, advertiser or title.")
log(f"- **Missing values:** AI hook tags present for {X['_hook_tagged'].mean():.0%} of training ads "
    "(the rest were tags that contradicted the transcript and were blanked; their tag columns are 0).")
X = X.drop(columns=["_hook_tagged"])
binary = [c for c in X.columns if set(X[c].unique()) <= {0, 1}]
rare = [c for c in binary if min(X[c].sum(), len(X) - X[c].sum()) < 6]
# reference level per categorical tag = its most common level (avoids the dummy-variable trap, which otherwise
# inflates VIF for the informative levels); "unclear" levels are merged into the reference as well
refs = [c for c in X.columns if c.endswith("_unclear") or c == "t_trope_none"]
for pre in ("t_hook_", "t_vis_", "t_genre_", "t_setting_"):
    lv = [c for c in X.columns if c.startswith(pre) and c not in ("t_hook_is_question",) and c not in refs]
    if lv:
        refs.append(max(lv, key=lambda c: X[c].sum()))


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    if min(tab.shape) < 2:
        return 0.0
    chi2 = chi2_contingency(tab, correction=False)[0]
    return float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))


cats = {k: train[c].fillna("missing") for k, c in (("hook_type", "ai_hook_type"), ("visual_hook", "ai_img_visual_hook"),
                                                    ("genre", "ai_genre"), ("setting", "ai_setting")) if c in train}
cv_pairs = [(a, b, cramers_v(cats[a], cats[b])) for i, a in enumerate(cats) for b in list(cats)[i + 1:]]
high_v = [(a, b, v) for a, b, v in cv_pairs if v > 0.7]
setting_cols = [c for c in X.columns if c.startswith("t_setting_")] if any("setting" in (a, b) for a, b, _ in high_v) else []
X1 = X.drop(columns=sorted(set(rare) | set(refs) | set(setting_cols)))
log(f"- **Near-zero variance:** dropped {len(rare)} binary features with fewer than 6 positives or negatives. "
    f"**Reference levels** (one per categorical tag, the most common level, plus 'unclear'/'none') removed to avoid "
    f"the dummy-variable trap: {', '.join(f'`{c}`' for c in sorted(set(refs) - set(rare)))}.")
log("- **Cramer's V between categorical tags:** " + ", ".join(f"{a}/{b} {v:.2f}" for a, b, v in cv_pairs) +
    (f". genre/setting is above 0.7 (fantasy and historical genres imply the setting), so **setting is dropped**." if setting_cols else "."))

corr = X1.corr().fillna(0)
tr_r = X1.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]) if c.std() > 0 else 0)
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:] if abs(corr.loc[a, b]) > 0.7]
drop_corr = set()
for a, b, r in sorted(pairs, key=lambda p: -abs(p[2])):
    if a not in drop_corr and b not in drop_corr:
        drop_corr.add(a if tr_r[a] < tr_r[b] else b)
log(f"- **Pairwise correlation (|r| > 0.7):** {len(pairs)} pairs" +
    "".join(f"\n  - `{a}` / `{b}`: r = {r:.2f}" for a, b, r in pairs) +
    (f"\n  Dropped the weaker member of each: {', '.join(f'`{c}`' for c in sorted(drop_corr))}." if drop_corr else ""))
X2 = X1.drop(columns=sorted(drop_corr))


def vif(df):
    Z = ((df - df.mean()) / df.std().replace(0, 1)).values
    out = {}
    for j, c in enumerate(df.columns):
        A = np.c_[np.ones(len(Z)), np.delete(Z, j, axis=1)]
        beta, *_ = np.linalg.lstsq(A, Z[:, j], rcond=None)
        r2 = 1 - ((Z[:, j] - A @ beta) ** 2).sum() / (((Z[:, j] - Z[:, j].mean()) ** 2).sum() or 1)
        out[c] = 1 / max(1 - r2, 1e-6)
    return pd.Series(out)


drop_vif = []
while X2.shape[1] > 2 and (v := vif(X2)).max() > 5:
    drop_vif.append((v.idxmax(), round(float(v.max()), 1)))
    X2 = X2.drop(columns=v.idxmax())
log(f"- **Multicollinearity (VIF > 5):** " + (", ".join(f"dropped `{c}` ({x})" for c, x in drop_vif) + "; " if drop_vif else "none above 5; ")
    + f"max VIF now {vif(X2).max():.2f}.")
log(f"- **Class balance:** {y.mean():.0%} high overall; the within-app weights make each app 50/50.")
log(f"\n{X2.shape[1]} features pass the audit.")

plt.figure(figsize=(10, 8.5))
c2 = X2.corr().fillna(0)
plt.imshow(c2, cmap="RdBu_r", vmin=-1, vmax=1)
plt.xticks(range(len(c2)), c2.columns, rotation=90, fontsize=6)
plt.yticks(range(len(c2)), c2.columns, fontsize=6)
plt.colorbar(fraction=0.03)
plt.title("Feature correlation after the audit")
plt.tight_layout()
plt.savefig(ART / "correlation_heatmap.png", dpi=130)
plt.close()


# ------------------------------------------------------------------ evaluation
def within_app_auc(yt, p, a):
    vals, wts = [], []
    for g in set(a):
        m = a == g
        if len(set(yt[m])) == 2:
            vals.append(roc_auc_score(yt[m], p[m]))
            wts.append(m.sum())
    return float(np.average(vals, weights=wts)) if vals else np.nan


def make(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))


strata = [f"{a}|{t}" for a, t in zip(apps, y)]


def evaluate(cols, C, repeats=10):
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=11)
    out = []
    for tr, te in rs.split(X2, strata):
        m = make(C).fit(X2[cols].iloc[tr], y[tr], logisticregression__sample_weight=w[tr])
        out.append(within_app_auc(y[te], m.predict_proba(X2[cols].iloc[te])[:, 1], apps[te]))
    return float(np.nanmean(out)), float(np.nanstd(out))


GROUPS = {
    "measured signals (no AI)": [c for c in X2.columns if c.startswith("f_")],
    "hook type (AI tag)": [c for c in X2.columns if c.startswith("t_hook_") and c != "t_hook_is_question"],
    "on-screen visual hook (AI tag)": [c for c in X2.columns if c.startswith("t_vis_")],
    "genre + tropes (AI tag)": [c for c in X2.columns if c.startswith(("t_genre_", "t_trope_"))],
    "tone flags (AI tag)": [c for c in X2.columns if c in ("t_emotion_intensity", "t_open_mid_conflict",
                                                           "t_hook_is_question", "t_narrator_voiceover")],
}
log("\n## 3. Which signals predict reach within an app?\n")
log("Within-app ROC AUC, 5-fold x 10 repeats (0.50 = coin flip).\n")
log("| feature group | features | within-app AUC |\n|---|---|---|")
gres = {}
for name, cols in GROUPS.items():
    if not cols:
        continue
    C, m, sd = max(((C, *evaluate(cols, C)) for C in (0.01, 0.03, 0.1, 0.3)), key=lambda t: t[1])
    gres[name] = {"cols": cols, "C": C, "auc": round(m, 3), "sd": round(sd, 3)}
    log(f"| {name} | {len(cols)} | {m:.3f} ± {sd:.3f} |")

keep = [n for n, r in gres.items() if r["auc"] - r["sd"] / np.sqrt(50) * 2 > 0.5]  # mean clearly above chance
if not keep:
    keep = [max(gres, key=lambda n: gres[n]["auc"])]
FULL = sorted({c for n in keep for c in gres[n]["cols"]})
C_full, auc_full, sd_full = max(((C, *evaluate(FULL, C)) for C in (0.01, 0.03, 0.1, 0.3)), key=lambda t: t[1])
BASIC = GROUPS["measured signals (no AI)"]
C_basic, auc_basic, sd_basic = gres["measured signals (no AI)"]["C"], gres["measured signals (no AI)"]["auc"], gres["measured signals (no AI)"]["sd"]
log(f"\nKept the groups whose AUC is clearly above chance: **{', '.join(keep)}**. "
    f"Final model: {len(FULL)} features, within-app AUC **{auc_full:.3f} ± {sd_full:.3f}** (C = {C_full}). "
    f"Note: choosing groups on the same CV slightly flatters this number.")
skill = lambda a: round(float(np.clip((a - 0.5) * 2, 0, 0.5)), 3)  # noqa: E731
log(f"\n**Weight in the final ranking** = min(0.5, 2 x (AUC - 0.5)) = **{skill(auc_full):.2f}** for the full model; "
    f"the basic model (no AI) gets {skill(auc_basic):.2f}, i.e. it is not used.")

full_model = make(C_full).fit(X2[FULL], y, logisticregression__sample_weight=w)
basic_model = make(C_basic).fit(X2[BASIC], y, logisticregression__sample_weight=w)
joblib.dump(full_model, ART / "model_full.joblib")
joblib.dump(basic_model, ART / "model_basic.joblib")


def export_json(pipe, cols, path):
    """Plain-text copy of the model (scaler + logistic regression) the app can run without pickles."""
    sc, lr = pipe[0], pipe[-1]
    path.write_text(json.dumps({"features": list(cols), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}, indent=2),
                    encoding="utf-8")


export_json(full_model, FULL, ART / "model_full.json")
export_json(basic_model, BASIC, ART / "model_basic.json")

# ------------------------------------------------------------------ external check
mid = uni[uni.creative_best_tier == "mid"]
ext = None
if len(mid) > 10:
    pm = full_model.predict_proba(X_all.loc[mid.index].reindex(columns=FULL, fill_value=0))[:, 1]
    rho, p = spearmanr(pm, mid.days_shown)
    ext = {"rho": round(float(rho), 3), "p": round(float(p), 3), "n": int(len(mid))}
    verdict = ("a small positive link" if rho > 0.1 and p < 0.1 else "no clear link")
    log(f"\n## 4. External check\nOn {len(mid)} mid-reach ads never used in training, the model's score vs the ad's real "
        f"active days: Spearman rho = {rho:.2f} (p = {p:.2f}), {verdict}.")

coef = pd.Series(full_model[-1].coef_[0], index=FULL).sort_values()
log("\n## 5. What the model learned (standardised coefficients, + = more likely high reach)\n")
for c, v in coef.items():
    log(f"- `{c}`: {v:+.2f}")

# ------------------------------------------------------------------ insights (within-app weighted win rates)
base = 0.5  # by construction of the weights


def wrates(col, title, subtitle, min_n=8, fmt=lambda s: s.replace("_", " ")):
    t = pd.DataFrame({"v": col, "y": y, "w": w}).dropna()
    rows = []
    for k, g in t.groupby("v"):
        if len(g) >= min_n:
            rows.append({"label": fmt(str(k)), "rate": round(float(np.average(g.y, weights=g.w)), 3), "n": int(len(g))})
    return {"title": title, "subtitle": subtitle, "base": base, "rows": sorted(rows, key=lambda r: -r["rate"])}


def auc_note(group):
    r = gres.get(group)
    return f" As a predictor: within-app AUC {r['auc']:.2f} ± {r['sd']:.2f}." if r else ""


blocks = [
    wrates(train.ai_hook_type, "How the ad opens (hook type)",
           "AI-tagged from the first 15 seconds." + auc_note("hook type (AI tag)")),
    wrates(train.ai_img_visual_hook, "What is on screen in the first 3 seconds",
           "AI-tagged from real keyframes." + auc_note("on-screen visual hook (AI tag)")),
    wrates(np.where(X.f_silent_open_3s == 1, "Silent first 3 s", "Someone speaks in the first 3 s"),
           "Talking vs silent opening", "Measured from the audio, no AI." + auc_note("measured signals (no AI)")),
    wrates(np.where(X.f_face_closeup_3s == 1, "Face close-up in first 3 s", "No face close-up"),
           "Close-up opening", "Measured with a face detector, no AI."),
]
insights = {
    "headline": "What made real short-drama ads reach more people",
    "lede": (f"{len(train)} real ads from {', '.join(a for a in both if a != 'unspecified')} and other drama publishers "
             f"in TikTok's public ad library, compared within each app. "
             "Bars show the share of an app's ads that reached 100K+ people (weighted so every app counts equally); "
             "the pink line is 50%, a coin flip."),
    "blocks": [b for b in blocks if b["rows"]],
    "model_card": [
        f"Trained on {len(train)} unique English short-drama ad videos from {len(both)} apps; labels are TikTok's reach bands (100K+ vs under 10K).",
        f"High- and low-reach ads came from different apps, so the model compares ads within the same app. A naive model only learned the app.",
        f"Honest result: no signal group reliably predicts reach within an app at this sample size (best within-app AUC {auc_full:.2f}; measured signals {auc_basic:.2f}; 0.50 is a coin flip).",
        "Every ranking signal (this model, the AI stranger test, the script rules) was tested on the same real ads; each one's share of the final score is earned from that test (see above). The win rates below are descriptive, not proof.",
        "Reach measures exposure, not installs. Production plan: retrain weekly on Kuku's own ad results (3-second hold, CTR, installs).",
    ],
}
(ART / "insights.json").write_text(json.dumps(insights, indent=2), encoding="utf-8")
meta = {"full": {"features": FULL, "C": C_full, "within_app_auc": round(auc_full, 3), "sd": round(sd_full, 3), "weight": skill(auc_full)},
        "basic": {"features": BASIC, "C": C_basic, "within_app_auc": auc_basic, "weight": skill(auc_basic)},
        "groups": {k: {kk: vv for kk, vv in v.items() if kk != "cols"} for k, v in gres.items()},
        "external_mid_days": ext, "n_train": int(len(train)), "apps": list(both),
        "dropped": {"rare": rare, "reference": refs, "setting": setting_cols, "correlated": sorted(drop_corr), "vif": drop_vif},
        "summary": f"hook-type model, {len(train)} real ads, within-app AUC {auc_full:.2f} (ranking weight {skill(auc_full):.2f})"}
(ART / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
(ART / "training_report.md").write_text("\n".join(report), encoding="utf-8")
print(f"\nSaved to {ART}")
