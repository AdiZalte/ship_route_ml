"""
train_risk.py
-------------
Weather-risk classifier.

Label: a test window is "deviation" when the course-speed baseline's
25-minute (final) displacement error > 90th percentile of TRAIN errors.

Features:
  Base : last lat, lon, sog, mean sog (window), sin_cog, cos_cog
  +Weather: wave_height, wind_speed at last step

Threshold chosen on validation set to maximise F2 (missed risk costs more
than a false alarm).

Usage:
    python -m src.models.train_risk
"""

import json
import logging
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, fbeta_score)
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

from src.data.weather_join import add_weather, set_train_medians

DATA    = "data/processed/trajectory_sequences.npz"
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)

SEED     = 42
OUTPUT_STEPS = 5

# -- load sequences ------------------------------------------------------------
data    = np.load(DATA)
X_train = data["X_train"];  y_train = data["y_train"];  T_train = data["T_train"]
X_val   = data["X_val"];    y_val   = data["y_val"];    T_val   = data["T_val"]
X_test  = data["X_test"];   y_test  = data["y_test"];   T_test  = data["T_test"]


# -- course-speed baseline displacement error ----------------------------------
def course_speed_final_err(X, Y):
    """Haversine error (km) at the final (25-min) output step."""
    sog    = X[:, -1, 2] * 1.852 / 60        # km/min
    course = np.arctan2(X[:, -1, 3], X[:, -1, 4])
    dist   = sog * (OUTPUT_STEPS * 5)         # 25 min -> km
    lat1   = np.radians(X[:, -1, 0])
    lon1   = np.radians(X[:, -1, 1])
    ad     = dist / 6371.0
    lat2   = np.arcsin(np.sin(lat1) * np.cos(ad)
                       + np.cos(lat1) * np.sin(ad) * np.cos(course))
    lon2   = lon1 + np.arctan2(
        np.sin(course) * np.sin(ad) * np.cos(lat1),
        np.cos(ad) - np.sin(lat1) * np.sin(lat2),
    )
    pred   = np.column_stack([np.degrees(lat2),
                              (np.degrees(lon2) + 540) % 360 - 180])
    actual = Y[:, -1, :]      # (N, 2)
    a, b = np.radians(actual), np.radians(pred)
    h = (np.sin((b[:, 0] - a[:, 0]) / 2) ** 2
         + np.cos(a[:, 0]) * np.cos(b[:, 0])
         * np.sin((b[:, 1] - a[:, 1]) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


print("Computing course-speed final errors ...")
err_train = course_speed_final_err(X_train, y_train)
err_val   = course_speed_final_err(X_val,   y_val)
err_test  = course_speed_final_err(X_test,  y_test)

threshold_90 = float(np.percentile(err_train, 90))
print(f"  Train 90th-pctile error: {threshold_90:.3f} km")

y_tr  = (err_train > threshold_90).astype(int)
y_va  = (err_val   > threshold_90).astype(int)
y_te  = (err_test  > threshold_90).astype(int)

print(f"  Train deviation rate: {y_tr.mean():.2%}")
print(f"  Val   deviation rate: {y_va.mean():.2%}")
print(f"  Test  deviation rate: {y_te.mean():.2%}")


# -- features ------------------------------------------------------------------
def base_features(X):
    """Return (N, 6) array: last lat, lon, sog, mean_sog, sin_cog, cos_cog."""
    return np.column_stack([
        X[:, -1, 0],        # last lat
        X[:, -1, 1],        # last lon
        X[:, -1, 2],        # last sog
        X[:, :, 2].mean(1), # mean sog over window
        X[:, -1, 3],        # sin_cog
        X[:, -1, 4],        # cos_cog
    ]).astype(np.float32)


FEAT_NAMES_BASE = ["lat", "lon", "sog", "mean_sog", "sin_cog", "cos_cog"]
FEAT_NAMES_W    = FEAT_NAMES_BASE + ["wave_height", "wind_speed"]

F_tr_base = base_features(X_train)
F_va_base = base_features(X_val)
F_te_base = base_features(X_test)

# -- weather lookup ------------------------------------------------------------
print("Looking up weather ...")
wh_tr, ws_tr = add_weather(X_train[:, -1, 0], X_train[:, -1, 1], T_train)
set_train_medians(float(np.nanmedian(wh_tr)), float(np.nanmedian(ws_tr)))
# re-lookup with medians in place
wh_tr, ws_tr = add_weather(X_train[:, -1, 0], X_train[:, -1, 1], T_train)
wh_va, ws_va = add_weather(X_val[:, -1, 0],   X_val[:, -1, 1],   T_val)
wh_te, ws_te = add_weather(X_test[:, -1, 0],  X_test[:, -1, 1],  T_test)

F_tr_w = np.column_stack([F_tr_base, wh_tr, ws_tr]).astype(np.float32)
F_va_w = np.column_stack([F_va_base, wh_va, ws_va]).astype(np.float32)
F_te_w = np.column_stack([F_te_base, wh_te, ws_te]).astype(np.float32)

scale_pos = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
print(f"  scale_pos_weight: {scale_pos:.2f}")


# -- train classifiers ---------------------------------------------------------
def train_clf(F_tr, y_tr, F_va, y_va, label: str) -> XGBClassifier:
    clf = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        early_stopping_rounds=20,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )
    clf.fit(F_tr, y_tr, eval_set=[(F_va, y_va)], verbose=False)
    print(f"  [{label}] best iter: {clf.best_iteration}")
    return clf


print("\nTraining base classifier ...")
clf_base = train_clf(F_tr_base, y_tr, F_va_base, y_va, "base")
print("Training weather classifier ...")
clf_w    = train_clf(F_tr_w,    y_tr, F_va_w,    y_va, "weather")


# -- threshold selection (maximise F2 on validation) ---------------------------
def best_threshold(clf, F_va, y_va):
    proba = clf.predict_proba(F_va)[:, 1]
    best_f2, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        pred = (proba >= thr).astype(int)
        f2   = fbeta_score(y_va, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr = f2, thr
    return best_thr, best_f2


thr_base, f2_base = best_threshold(clf_base, F_va_base, y_va)
thr_w,    f2_w    = best_threshold(clf_w,    F_va_w,    y_va)
print(f"\n  Base threshold (F2 opt): {thr_base:.2f}  (val F2={f2_base:.3f})")
print(f"  Weathr threshold(F2 opt): {thr_w:.2f}  (val F2={f2_w:.3f})")


# -- test evaluation -----------------------------------------------------------
def evaluate(clf, F_te, y_te, thr, feat_names, label):
    proba = clf.predict_proba(F_te)[:, 1]
    pred  = (proba >= thr).astype(int)
    prec  = precision_score(y_te, pred, zero_division=0)
    rec   = recall_score(y_te, pred, zero_division=0)
    f1    = f1_score(y_te, pred, zero_division=0)
    f2    = fbeta_score(y_te, pred, beta=2, zero_division=0)
    cm    = confusion_matrix(y_te, pred)

    print(f"\n-- {label} --")
    print(f"  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  F2: {f2:.3f}")
    print(f"  Confusion matrix (rows=actual, cols=pred):")
    print(f"    TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"    FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
    print("  Feature importance:")
    for feat, imp in sorted(zip(feat_names, clf.feature_importances_),
                            key=lambda x: -x[1]):
        print(f"    {feat:<20} {imp:.4f}")
    return proba


print("\n" + "="*60)
print("Risk Classifier -- Test Results")
print("="*60)
prob_base = evaluate(clf_base, F_te_base, y_te, thr_base,
                     FEAT_NAMES_BASE, "Without weather")
prob_w    = evaluate(clf_w,    F_te_w,    y_te, thr_w,
                     FEAT_NAMES_W,    "With weather")
print("="*60)


# -- save ----------------------------------------------------------------------
joblib.dump(clf_w, OUT_DIR / "risk_xgb.joblib")
with open(OUT_DIR / "risk_threshold.json", "w") as f:
    json.dump({
        "threshold":        thr_w,
        "threshold_no_weather": thr_base,
        "error_threshold_km": threshold_90,
    }, f, indent=2)

print(f"\nSaved: {OUT_DIR / 'risk_xgb.joblib'}")
print(f"Saved: {OUT_DIR / 'risk_threshold.json'}")
