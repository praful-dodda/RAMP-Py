# =============================================================
# ozone_stacked_ensemble.py
#
# Train a stacked ensemble that blends LightGBM, Random Forest
# and a lightweight 1‑D CNN to predict daily‑max 8‑hour (MDA8)
# ozone at monitoring stations, using collocated model outputs
# and optional auxiliary covariates.
#
# Assumptions
# -----------
# • You have already created a collocated dataframe where each
#   row represents one monitor‑day and contains:
#       - observation column    ->  TARGET_COL (default: "OBS")
#       - model predictions     -> columns starting with "mdl_"
#         (e.g. mdl_AM4, mdl_CAMS, …)
#       - optional covariates   -> any other numeric columns
#       - grouping keys         -> STATION_ID_COL, YEAR_COL
# • Functions for loading / tagging data (get_ozone_file, …)
#   live elsewhere in the codebase. Here we focus on modelling.
# -------------------------------------------------------------

from __future__ import annotations

import os
import re
import joblib
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Union

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMRegressor

# Keras wrapper for scikit‑learn
from scikeras.wrappers import KerasRegressor
from tensorflow.keras import layers, models, callbacks

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------
# Configuration constants                                              
# --------------------------------------------------------------------
TARGET_COL: str = "OBS"          # name of observed MDA8 column
STATION_ID_COL: str = "station_id"
DATE_COL: str = "date"          # pandas.Timestamp or YYYY‑MM‑DD str
YEAR_COL: str = "year"          # convenience integer column

MODEL_PREFIX = "mdl_"            # prefix for input‑model predictors
N_SPLITS = 10                     # GroupKFold splits (station × year)

# --------------------------------------------------------------------
# Helper utilities                                                     
# --------------------------------------------------------------------

def _infer_model_cols(df: pd.DataFrame, prefix: str = MODEL_PREFIX) -> List[str]:
    """Return model‑prediction columns (input features)."""
    return [c for c in df.columns if c.startswith(prefix)]


def _define_cnn(input_dim: int) -> models.Model:
    """Build a lightweight 1‑D CNN for tabular features.

    The architecture treats the feature vector as a 1‑D sequence
    (length = n_features, channels = 1). This simple conv net has
    proven competitive versus MLPs while keeping ≪100k params.
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(32, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.Conv1D(16, kernel_size=3, strides=1, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, name="out")
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


# --------------------------------------------------------------------
# Core modelling function                                              
# --------------------------------------------------------------------

def train_stacked_ensemble(
    df: pd.DataFrame,
    feature_columns: Union[List[str], None] = None,
    aux_numeric_cols: Union[List[str], None] = None,
    output_dir: str = "models",
    model_label: str = "stacked_LGBM_RF_CNN",
    debug_rows: Union[int, None] = None,
):
    """Train a stacked ensemble and save artefacts.

    Parameters
    ----------
    df : pd.DataFrame
        Collocated training dataframe.
    feature_columns : list[str] | None
        List of model‑prediction feature columns. If None, inferred
        with `MODEL_PREFIX`.
    aux_numeric_cols : list[str] | None
        Extra numeric covariates to concatenate; scaled jointly with
        model features. E.g. ERA5_T2M, elevation, NDVI …
    output_dir : str
        Directory where the fitted stacker (.joblib) and scalers are
        saved. Created if missing.
    model_label : str
        Base filename for artefacts (without extension).
    debug_rows : int | None
        If set, subsample rows for quick debugging.
    """

    os.makedirs(output_dir, exist_ok=True)

    if debug_rows is not None:
        df = df.sample(n=debug_rows, random_state=SEED).reset_index(drop=True)

    # --------------------------------------------------------------
    # 1. Feature selection & preprocessing
    # --------------------------------------------------------------
    model_cols = feature_columns or _infer_model_cols(df)
    if aux_numeric_cols is None:
        aux_numeric_cols = []

    X = df[model_cols + aux_numeric_cols]
    y = df[TARGET_COL].values

    # Scale every feature (tree ensembles don't depend on scaling but
    # the CNN does). We keep a copy of the scaler for inference.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, f"{model_label}_scaler.joblib"))

    # --------------------------------------------------------------
    # 2. Define base learners
    # --------------------------------------------------------------
    lgbm_params = dict(
        objective="regression",
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=255,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
    )
    lgbm = LGBMRegressor(**lgbm_params)

    rf = RandomForestRegressor(
        n_estimators=600,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=SEED,
    )

    # CNN wrapped for scikit‑learn (uses standardised features)
    input_dim = X_scaled.shape[1]
    cnn = KerasRegressor(
        model=_define_cnn,
        model__input_dim=input_dim,
        optimizer="adam",
        epochs=40,
        batch_size=128,
        verbose=0,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        random_state=SEED,
    )

    base_learners = [
        ("lgbm", lgbm),
        ("rf", rf),
        ("cnn", cnn),
    ]

    # --------------------------------------------------------------
    # 3. Meta‑learner (LightGBM) inside scikit‑learn StackingRegressor
    # --------------------------------------------------------------
    meta_learner = LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        random_state=SEED,
        n_jobs=-1,
    )

    stacker = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        passthrough=True,
        n_jobs=-1,
    )

    # --------------------------------------------------------------
    # 4. Cross‑validated training using GroupKFold (station × year)
    # --------------------------------------------------------------
    groups = (
        df[STATION_ID_COL].astype(str) + "_" + df[YEAR_COL].astype(str)
    )
    cv = GroupKFold(n_splits=N_SPLITS)

    # We assemble a pipeline: scaling → stacker. Although the tree
    # learners ignore scaling, this ensures the CNN sees scaled data.
    pipeline = Pipeline([
        ("scaler", scaler),  # passthrough obj remains fitted
        ("stack", stacker),
    ])

    # Fit (GroupKFold validation can be inspected externally if needed)
    pipeline.fit(X, y, stack__cv=cv, stack__groups=groups)

    # --------------------------------------------------------------
    # 5. Persist artefact
    # --------------------------------------------------------------
    artefact_path = os.path.join(output_dir, f"{model_label}.joblib")
    joblib.dump(pipeline, artefact_path)
    print(f"Saved stacked ensemble to {artefact_path}")

    return pipeline, artefact_path


# --------------------------------------------------------------------
# Prediction helper                                                    
# --------------------------------------------------------------------

def predict_with_ensemble(
    model_path: str,
    df_new: pd.DataFrame,
    feature_columns: Union[List[str], None] = None,
    aux_numeric_cols: Union[List[str], None] = None,
) -> np.ndarray:
    """Load fitted ensemble and return predictions for new data."""

    pipeline = joblib.load(model_path)
    if feature_columns is None:
        feature_columns = _infer_model_cols(df_new)
    if aux_numeric_cols is None:
        aux_numeric_cols = []

    X_new = df_new[feature_columns + aux_numeric_cols]
    preds = pipeline.predict(X_new)
    return preds


# --------------------------------------------------------------------
# Command‑line usage                                                   
# --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Train stacked ozone ensemble")
    ap.add_argument("--train_csv", required=True, help="Path to collocated training CSV")
    ap.add_argument("--output_dir", default="models", help="Dir to save model artefacts")
    ap.add_argument("--debug_rows", type=int, default=None, help="Subsample rows for quick tests")
    args = ap.parse_args()

    df_train = pd.read_csv(args.train_csv, parse_dates=[DATE_COL])

    # Ensure YEAR_COL exists (use date column if necessary)
    if YEAR_COL not in df_train.columns:
        df_train[YEAR_COL] = pd.to_datetime(df_train[DATE_COL]).dt.year

    train_stacked_ensemble(
        df=df_train,
        aux_numeric_cols=[],  # add extra covariates here
        output_dir=args.output_dir,
        debug_rows=args.debug_rows,
    )
