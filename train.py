"""Train, calibrate, evaluate and save the loan default model.

Usage:
    python src/train.py --data data/cs-training.csv
    python src/train.py --data data/cs-training.csv --tune 30
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import TARGET, clean, engineer, load_raw

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

BASE_PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31,
                   min_child_samples=50, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, reg_lambda=1.0,
                   random_state=SEED, n_jobs=-1, verbose=-1)


def cv_scores(model, X, y):
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    r = cross_validate(model, X, y, cv=cv,
                       scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"})
    return {k: (float(r[f"test_{k}"].mean()), float(r[f"test_{k}"].std()))
            for k in ("roc_auc", "pr_auc")}


def tune(X, y, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    def objective(trial):
        p = dict(BASE_PARAMS,
                 n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 8, 64),
                 min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True))
        r = cross_validate(LGBMClassifier(**p), X, y, cv=cv, scoring="roc_auc")
        return r["test_score"].mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    print(f"Best CV ROC-AUC {study.best_value:.4f} with {study.best_params}")
    return dict(BASE_PARAMS, **study.best_params)


def pick_threshold(y, p, cost_fn, cost_fp):
    """Threshold minimising expected cost per applicant."""
    grid = np.linspace(0.01, 0.60, 120)
    costs = [(((p < t) & (y == 1)).sum() * cost_fn +
              ((p >= t) & (y == 0)).sum() * cost_fp) / len(y) for t in grid]
    return float(grid[int(np.argmin(costs))])


def decision_metrics(y, p, t, cost_fn, cost_fp):
    pred = p >= t
    tp = int(((pred) & (y == 1)).sum())
    fp = int(((pred) & (y == 0)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    return dict(threshold=t,
                precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1),
                reject_rate=float(pred.mean()),
                cost_per_applicant=(fn * cost_fn + fp * cost_fp) / len(y))


def score_block(y, p):
    return dict(roc_auc=float(roc_auc_score(y, p)),
                pr_auc=float(average_precision_score(y, p)),
                brier=float(brier_score_loss(y, p)))


def main(a):
    reports = ROOT / "reports"
    (ROOT / "models").mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    df = clean(load_raw(a.data))
    y = df[TARGET].astype(int).to_numpy()
    X = engineer(df)
    features = list(X.columns)
    print(f"{len(X):,} rows, default rate {y.mean():.2%}, {len(features)} features")

    # 70% train / 15% calibration + threshold selection / 15% untouched test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                stratify=y, random_state=SEED)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                stratify=y_tmp, random_state=SEED)

    baseline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=1000, class_weight="balanced"))
    params = tune(X_tr, y_tr, a.tune) if a.tune else BASE_PARAMS

    print("Cross-validating on the training split...")
    cv = {"logistic_regression": cv_scores(baseline, X_tr, y_tr),
          "lightgbm": cv_scores(LGBMClassifier(**params), X_tr, y_tr)}
    for name, s in cv.items():
        print(f"  {name:20s} ROC-AUC {s['roc_auc'][0]:.4f} ± {s['roc_auc'][1]:.4f}"
              f" | PR-AUC {s['pr_auc'][0]:.4f}")

    baseline.fit(X_tr, y_tr)
    model = LGBMClassifier(**params).fit(X_tr, y_tr)

    # Isotonic calibration so the score can be read as a real probability
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    calibrator.fit(model.predict_proba(X_cal)[:, 1], y_cal)
    p_cal = calibrator.predict(model.predict_proba(X_cal)[:, 1])
    threshold = pick_threshold(y_cal, p_cal, a.cost_fn, a.cost_fp)

    p_lr = baseline.predict_proba(X_te)[:, 1]
    p_raw = model.predict_proba(X_te)[:, 1]
    p_te = calibrator.predict(p_raw)

    metrics = {
        "n_rows": int(len(X)), "default_rate": float(y.mean()),
        "cost_assumption": {"false_negative": a.cost_fn, "false_positive": a.cost_fp},
        "cv": cv,
        "test": {"logistic_regression": score_block(y_te, p_lr),
                 "lightgbm_raw": score_block(y_te, p_raw),
                 "lightgbm_calibrated": score_block(y_te, p_te)},
        "decision": decision_metrics(y_te, p_te, threshold, a.cost_fn, a.cost_fp),
        "decision_reject_all_baseline_cost": float(y_te.mean() * a.cost_fn),
        "params": {k: v for k, v in params.items() if k not in ("n_jobs", "verbose")},
    }
    print(json.dumps(metrics["test"], indent=2))
    print(json.dumps(metrics["decision"], indent=2))

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, p in [("Logistic regression", p_lr), ("LightGBM", p_te)]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[1].plot(rc, pr, label=f"{name} (AP {average_precision_score(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for x in ax:
        x.legend()
    fig.tight_layout()
    fig.savefig(reports / "roc_pr.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in [("Raw LightGBM", p_raw), ("Calibrated", p_te)]:
        frac, mean = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=name)
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=0.8)
    ax.set(xlabel="Predicted probability", ylabel="Observed default rate",
           title="Calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "calibration.png", dpi=150)

    sample = X_te.sample(min(2000, len(X_te)), random_state=SEED)
    sv = shap.TreeExplainer(model)(sample)
    if sv.values.ndim == 3:
        sv = sv[:, :, 1]
    plt.figure()
    shap.plots.beeswarm(sv, show=False, max_display=12)
    plt.tight_layout()
    plt.savefig(reports / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    joblib.dump({"model": model, "calibrator": calibrator, "threshold": threshold,
                 "features": features, "metrics": metrics}, ROOT / "models" / "model.joblib")
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/model.joblib and reports/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "cs-training.csv"))
    ap.add_argument("--tune", type=int, default=0, help="Optuna trials (0 = skip)")
    ap.add_argument("--cost-fn", type=float, default=5.0,
                    help="Cost of approving a loan that defaults")
    ap.add_argument("--cost-fp", type=float, default=1.0,
                    help="Cost of rejecting a good applicant")
    main(ap.parse_args())
