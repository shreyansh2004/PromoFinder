"""Experiment: AI tags only -> success, for two labels (reach band, active days), three model types,
three evaluations (plain CV, unseen source, within app). Answers: does dropping other features or
switching label/model improve AUC?"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from promofinder.features import tag_features  # noqa: E402

raw = pd.read_csv(sys.argv[1], dtype={"ad_id": str})
u = raw[(raw.language == "en") & (raw.is_short_drama_ad == 1) & (raw.is_first_of_creative == 1)].copy()
u["source"] = np.where(u.app == "unspecified", "search", "named")


def nn(v):
    return None if (isinstance(v, float) and np.isnan(v)) else v


def tags(r):
    t = {k: nn(r.get(f"ai_{k}")) for k in ("genre", "hook_type", "setting", "emotion_intensity", "open_mid_conflict",
                                            "hook_is_question", "narrator_voiceover", "tropes")}
    t["visual_hook"] = nn(r.get("ai_img_visual_hook"))
    return tag_features(t)


X_all = pd.DataFrame([tags(r) for _, r in u.iterrows()], index=u.index)
X_all = X_all.loc[:, X_all.std() > 0]
X_all = X_all.loc[:, [c for c in X_all if X_all[c].nunique() > 2 or min(X_all[c].sum(), len(X_all) - X_all[c].sum()) >= 6]]

labels = {
    "reach (100K+ vs <10K)": u.creative_best_tier.map({"high": 1, "low": 0}),
    "active days (>=7 vs <=2)": np.where(u.days_shown >= 7, 1, np.where(u.days_shown <= 2, 0, np.nan)),
}
models = {
    "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
    "random forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
    "gradient boosting": lambda: GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0),
}


def within_app(y, p, apps):
    v, w = [], []
    for a in set(apps):
        m = apps == a
        if len(set(y[m])) == 2:
            v.append(roc_auc_score(y[m], p[m]))
            w.append(m.sum())
    return np.average(v, weights=w) if v else np.nan


print(f"{len(u)} unique English drama ads; {X_all.shape[1]} AI-tag features\n")
for lname, lab in labels.items():
    lab = pd.Series(lab, index=u.index)
    m = lab.notna()
    X, y, src, apps = X_all[m], lab[m].astype(int).values, u.source[m].values, u.app[m].values
    print(f"=== label: {lname}  n={len(y)}  positives={y.mean():.0%}")
    print(f"    label rate by source: search {y[src == 'search'].mean():.0%}, named apps {y[src == 'named'].mean():.0%}")
    for mname, mk in models.items():
        rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
        plain, wa = [], []
        for tr, te in rs.split(X, y):
            mod = mk().fit(X.iloc[tr], y[tr])
            p = mod.predict_proba(X.iloc[te])[:, 1]
            plain.append(roc_auc_score(y[te], p))
            wa.append(within_app(y[te], p, apps[te]))
        # unseen source: train on one source, test on the other (both directions)
        lso = []
        for s in ("search", "named"):
            tr, te = src != s, src == s
            if len(set(y[te])) == 2 and len(set(y[tr])) == 2:
                mod = mk().fit(X[tr], y[tr])
                lso.append(roc_auc_score(y[te], mod.predict_proba(X[te])[:, 1]))
        print(f"    {mname:20s} plain CV {np.mean(plain):.3f} | unseen source {np.mean(lso):.3f} | within app {np.nanmean(wa):.3f}")
    print()
