"""Exploration: is there any within-app signal once app/collection confounding is removed?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promofinder.features import tag_features, text_features, timing_features, visual_features  # noqa

d = pd.read_csv(sys.argv[1], dtype={"ad_id": str})
d = d[(d.language == "en") & (d.is_short_drama_ad == 1) & (d.is_first_of_creative == 1) & d.creative_best_tier.isin(["high", "low"])]
both = d.groupby("app").creative_best_tier.nunique().loc[lambda s: s == 2].index
d = d[d.app.isin(both)].copy()
y = (d.creative_best_tier == "high").astype(int).values
cnt = d.groupby(["app", "creative_best_tier"]).ad_id.transform("count")
w = (0.5 / cnt).values
w = w / w.mean()
print("apps:", dict(d.app.value_counts()), "n", len(d))


def S(v):
    return "" if pd.isna(v) else str(v)


def feats(r):
    f = {**text_features(S(r.first_3s_text), S(r.hook_15s_text), ""),
         **timing_features(r.speech_starts_at_s if pd.notna(r.speech_starts_at_s) else None),
         **visual_features(r.n_cuts_3s if pd.notna(r.n_cuts_3s) else 0, r.face_closeup_3s if pd.notna(r.face_closeup_3s) else 0)}
    t = {k: (r.get(f"ai_{k}") if pd.notna(r.get(f"ai_{k}")) else None) for k in
         ("genre", "hook_type", "emotion_intensity", "open_mid_conflict", "hook_is_question", "narrator_voiceover", "tropes")}
    t["visual_hook"] = r.get("ai_img_visual_hook") if pd.notna(r.get("ai_img_visual_hook")) else None
    f.update(tag_features(t))
    return f


X = pd.DataFrame([feats(r) for _, r in d.iterrows()], index=d.index)
X = X.loc[:, [c for c in X.columns if X[c].std() > 0 and min(X[c].sum(), len(X) - X[c].sum()) >= 6 or not set(X[c].unique()) <= {0, 1}]]
X = X.drop(columns=[c for c in X.columns if c.endswith("_unclear") or c == "t_trope_none" or c.startswith("t_setting")], errors="ignore")
apps = d.app.values
strata = [f"{a}|{t}" for a, t in zip(apps, y)]


def within_app_auc(yt, p, a):
    out, wts = [], []
    for g in set(a):
        m = a == g
        if len(set(yt[m])) == 2:
            out.append(roc_auc_score(yt[m], p[m]))
            wts.append(m.sum())
    return np.average(out, weights=wts) if out else np.nan


def evaluate(cols, C):
    rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=8, random_state=3)
    res = []
    for tr, te in rs.split(X, strata):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(X[cols].iloc[tr], y[tr], logisticregression__sample_weight=w[tr])
        res.append(within_app_auc(y[te], m.predict_proba(X[cols].iloc[te])[:, 1], apps[te]))
    return np.nanmean(res), np.nanstd(res)


groups = {
    "measured (no AI)": [c for c in X.columns if c.startswith("f_")],
    "visual hook tags": [c for c in X.columns if c.startswith("t_vis_")],
    "hook type tags": [c for c in X.columns if c.startswith("t_hook_")],
    "genre+trope tags": [c for c in X.columns if c.startswith(("t_genre_", "t_trope_"))],
    "other tags": [c for c in X.columns if c in ("t_emotion_intensity", "t_open_mid_conflict", "t_hook_is_question", "t_narrator_voiceover")],
}
groups["all"] = list(X.columns)
for name, cols in groups.items():
    if not cols:
        continue
    best = max(((C, *evaluate(cols, C)) for C in (0.01, 0.03, 0.1, 0.3)), key=lambda t: t[1])
    print(f"{name:20s} n_feat={len(cols):3d}  best C={best[0]:<5} within-app AUC {best[1]:.3f} ± {best[2]:.3f}")

# raw within-app win rates per visual hook (weighted)
t = pd.DataFrame({"vis": d.ai_img_visual_hook.fillna("missing"), "y": y, "w": w})
g = t.groupby("vis").apply(lambda x: pd.Series({"wrate": np.average(x.y, weights=x.w), "n": len(x)}))
print(g.sort_values("wrate", ascending=False).round(2))
t = pd.DataFrame({"h": d.ai_hook_type.fillna("missing"), "y": y, "w": w})
print(t.groupby("h").apply(lambda x: pd.Series({"wrate": np.average(x.y, weights=x.w), "n": len(x)})).sort_values("wrate", ascending=False).round(2))
