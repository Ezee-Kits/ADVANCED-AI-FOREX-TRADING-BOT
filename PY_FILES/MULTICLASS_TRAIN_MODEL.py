# ==========================================================
# TRAINNING THE MODEL [ MAX EXCURSION --> 5 CLASSES TARGET ]
# ==========================================================

import os
import ta
import shap
import json
import optuna
import joblib
import random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import TimeSeriesSplit
from func import apply_features, build_5class_target
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)


SYMBOL = "EURUSD"
CD_TIME = "10M"
print(f"CURRENTLY RUNNING SYMBOL IS {SYMBOL}")

MODEL_FOLDER = "/content/drive/MyDrive/FOREX TRADING/ALL_MODELS"
CSV_FOLDER = "/content/drive/MyDrive/FOREX TRADING/CSV_FILES"
IMPORTANCE_FOLDER = "/content/drive/MyDrive/FOREX TRADING/SHAP RESULT"

# MODEL_FOLDER = 'ALL_MODELS'
# CSV_FOLDER = 'CSV_FILES'
# IMPORTANCE_FOLDER = 'SHAP RESULTS'


POINTS_FILE = os.path.join(CSV_FOLDER, "symbol_points.json")
MT5_SYMBOL = SYMBOL + "c"

print(f"CURRENTLY RUNNING SYMBOL: {SYMBOL}")
print(f"MT5 SYMBOL: {MT5_SYMBOL}")

with open(POINTS_FILE, "r", encoding="utf-8") as file:
    SYMBOL_POINTS = json.load(file)

if MT5_SYMBOL not in SYMBOL_POINTS:
    raise ValueError(f"Point information not found for {MT5_SYMBOL} in {POINTS_FILE}")


POINT = SYMBOL_POINTS[MT5_SYMBOL]["point"]
print(f"POINT: {POINT}")


# ==========================================================
# LOAD DATA
# ==========================================================
data = pd.read_csv(f"{CSV_FOLDER}/MT5_{CD_TIME}_{SYMBOL}_Exchange_Rate_Dataset.csv")
if len(data) < 150000:
    raise ValueError(f"---->> NUMBER OF TRAINING DATASET IS TOO LOW ")


df = apply_features(df=data, point=POINT)

df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(columns=["Date"], inplace=True)

print("DF INFO:", df.shape)

# ==========================================================
# PARAMETERS
# ==========================================================
TARGET_HORIZON = 18  # 18 × 10M = 3 HOURS
TARGET_NAME = "TARGET_5CLASS"

df, target_settings = build_5class_target(
    df=df, target_horizon=TARGET_HORIZON, target_name=TARGET_NAME
)
print(target_settings)

# ==========================================================
# STEP 5: LABEL DISTRIBUTION
# ==========================================================
print("\nLabel Distribution:")
print(df[TARGET_NAME].value_counts(normalize=True).sort_index())

print("\nLabel Counts:")

print(df[TARGET_NAME].value_counts().sort_index())

print("\nTarget Mapping:")
print("0 = Strong DOWN")
print("1 = Moderate DOWN")
print("2 = Small / NEUTRAL")
print("3 = Moderate UP")
print("4 = Strong UP")
print(f"Horizon = {TARGET_HORIZON} candles = {TARGET_HORIZON * 10 / 60:.1f} hours")


# ==========================================================
# SETTINGS
# ==========================================================
N_SPLITS = 5
N_TRIALS = 30
CORR_THRESHOLD = 0.98

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================


def remove_correlated_features(X, threshold=0.98):

    split_idx = int(len(X) * 0.7)
    X_train = X.iloc[:split_idx]

    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    print(f" \n 🧹 Dropping {len(to_drop)} correlated features")
    print(f"Features Before Corr Filter: {X.shape[1]}")
    print(f"Features After Corr Filter: {X.shape[1] - len(to_drop)} \n")
    print(" TO DROP FEATURES : \n", to_drop)
    return X.drop(columns=to_drop)


# ==========================================================
# EVALUATION FUNCTION (TIME SERIES SAFE - 5 CLASS)
# ==========================================================
def evaluate_model(X, y, params):

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []

    NUM_CLASSES = 5

    for train_idx, val_idx in tscv.split(X):

        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

        # --------------------------------------------------
        # Class Weights
        # --------------------------------------------------
        classes = np.unique(y_train)

        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )

        class_weight_dict = {
            int(classes[i]): float(class_weights[i]) for i in range(len(classes))
        }

        # --------------------------------------------------
        # Local Parameters
        # --------------------------------------------------
        params_local = params.copy()

        params_local["objective"] = "multiclass"
        params_local["num_class"] = NUM_CLASSES
        params_local["metric"] = "multi_logloss"
        params_local["class_weight"] = class_weight_dict

        # --------------------------------------------------
        # Train Model
        # --------------------------------------------------
        model = LGBMClassifier(**params_local)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)],
        )

        # --------------------------------------------------
        # Predict Probability For ALL 5 Classes
        # --------------------------------------------------
        best_iteration = model.best_iteration_

        if best_iteration is None:
            best_iteration = model.n_estimators

        proba = model.predict_proba(X_valid, num_iteration=best_iteration)

        # --------------------------------------------------
        # Safety Check
        # --------------------------------------------------
        if proba.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} probability columns, "
                f"but model returned {proba.shape[1]}."
            )

        auc = roc_auc_score(
            y_valid,
            proba,
            multi_class="ovr",
            average="macro",
            labels=np.arange(NUM_CLASSES),
        )

        scores.append(auc)

    # ------------------------------------------------------
    # Mean CV AUC
    # ------------------------------------------------------
    return np.mean(scores)


# ==========================================================
# SHAP FEATURE IMPORTANCE
# TIME SERIES SAFE - 5 CLASS MULTICLASS
# ==========================================================
def get_fold_shap_importance(X, y, params):

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    NUM_CLASSES = 5

    feature_importance = pd.DataFrame(index=X.columns)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):

        print(f"\n🔍 SHAP ANALYSIS - FOLD {fold + 1}/{N_SPLITS}")

        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[val_idx]

        # --------------------------------------------------
        # Class Weights
        # --------------------------------------------------
        classes = np.unique(y_train)

        class_weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=y_train
        )

        class_weight_dict = {
            int(classes[i]): float(class_weights[i]) for i in range(len(classes))
        }

        # --------------------------------------------------
        # Local Parameters
        # --------------------------------------------------
        params_local = params.copy()

        params_local["objective"] = "multiclass"
        params_local["num_class"] = NUM_CLASSES
        params_local["metric"] = "multi_logloss"
        params_local["class_weight"] = class_weight_dict

        # --------------------------------------------------
        # Train Model
        # --------------------------------------------------
        model = LGBMClassifier(**params_local)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)],
        )

        # --------------------------------------------------
        # Sample Training Data For SHAP
        # --------------------------------------------------
        X_sample = X_train.sample(min(1000, len(X_train)), random_state=42)

        # --------------------------------------------------
        # SHAP Tree Explainer
        # --------------------------------------------------
        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            if len(shap_values) != NUM_CLASSES:
                raise ValueError(
                    f"Expected SHAP values for {NUM_CLASSES} classes, "
                    f"but received {len(shap_values)}."
                )

            shap_array = np.stack(shap_values, axis=-1)

        elif isinstance(shap_values, np.ndarray):
            shap_array = shap_values

            if shap_array.ndim != 3:
                raise ValueError(
                    f"Expected 3D multiclass SHAP array, "
                    f"but received shape {shap_array.shape}."
                )

            # Detect and correct this automatically.
            if shap_array.shape[2] == NUM_CLASSES:
                pass

            elif shap_array.shape[1] == NUM_CLASSES:
                shap_array = np.transpose(shap_array, (0, 2, 1))

            else:
                raise ValueError(
                    f"Could not identify class dimension "
                    f"in SHAP array with shape {shap_array.shape}."
                )

        else:
            raise TypeError(f"Unsupported SHAP output type: " f"{type(shap_values)}")

        # --------------------------------------------------
        # Final SHAP Shape Check
        # --------------------------------------------------
        expected_shape = (len(X_sample), X_sample.shape[1], NUM_CLASSES)

        if shap_array.shape != expected_shape:
            raise ValueError(
                f"Unexpected SHAP shape: {shap_array.shape}. "
                f"Expected: {expected_shape}."
            )

        importance = np.abs(shap_array).mean(axis=0).mean(axis=1)

        # --------------------------------------------------
        # Safety Check
        # --------------------------------------------------
        if len(importance) != X.shape[1]:

            raise ValueError(
                f"SHAP importance length ({len(importance)}) "
                f"does not match number of features ({X.shape[1]})."
            )

        # --------------------------------------------------
        # Store Fold Importance
        # --------------------------------------------------
        feature_importance[f"fold_{fold}"] = importance

    # ======================================================
    # Aggregate Across Folds
    # ======================================================

    fold_columns = [
        col for col in feature_importance.columns if col.startswith("fold_")
    ]

    # Mean SHAP importance across folds
    feature_importance["mean_shap"] = feature_importance[fold_columns].mean(axis=1)

    # Standard deviation across folds
    feature_importance["std_shap"] = feature_importance[fold_columns].std(axis=1)

    # Stability:
    # Higher = important AND consistent across folds
    feature_importance["stability"] = feature_importance["mean_shap"] / (
        feature_importance["std_shap"] + 1e-9
    )

    return feature_importance


# ==========================================================
# TRAIN LOOP — 5-CLASS DIRECTION + MAGNITUDE MODEL
# ==========================================================

NUM_CLASSES = 5

for target in [TARGET_NAME]:

    print(f"\n==============================")
    print(f"🚀 FULL FEATURE TRAINING: {target}")
    print(f"==============================")

    # ------------------------------------------------------
    # Prevent target leakage
    # ------------------------------------------------------
    drop_targets = [t for t in [TARGET_NAME] if t != target and t in df.columns]

    df_train_target = df.drop(columns=drop_targets, errors="ignore").copy()

    df_train_target = df_train_target[df_train_target[target].notna()].copy()

    df_train_target.reset_index(drop=True, inplace=True)

    # ------------------------------------------------------
    # X / y
    # ------------------------------------------------------
    X_full = df_train_target.drop(columns=[target])

    y = df_train_target[target].astype(int)

    # Safety check
    unique_classes = np.sort(y.unique())

    expected_classes = np.arange(NUM_CLASSES)

    if not np.array_equal(unique_classes, expected_classes):
        raise ValueError(
            f"Expected target classes "
            f"{expected_classes.tolist()}, "
            f"but found "
            f"{unique_classes.tolist()}"
        )

    print(f"\nNumber of classes: {NUM_CLASSES}")

    print("Classes:", unique_classes.tolist())

    print("\nClass Distribution:")
    print(y.value_counts(normalize=True).sort_index())

    print("\nClass Counts:")
    print(y.value_counts().sort_index())

    print("\nLEN X AFTER PREPROCESSING:", len(X_full))

    # ------------------------------------------------------
    # Handle class imbalance
    # ------------------------------------------------------
    classes = np.unique(y)

    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)

    class_weight_dict = {
        int(classes[i]): float(class_weights[i]) for i in range(len(classes))
    }

    print("\nClass Weights:", class_weight_dict)

    # ------------------------------------------------------
    # Step 1: Correlation Filtering
    # ------------------------------------------------------
    X_filtered = remove_correlated_features(X_full, CORR_THRESHOLD)

    # ======================================================
    # STEP 2: INITIAL OPTUNA TUNING
    # ======================================================

    def objective_initial(trial):

        params = {
            "objective": "multiclass",
            "num_class": NUM_CLASSES,
            "metric": "multi_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": class_weight_dict,
        }

        # IMPORTANT:
        # evaluate_model() must now support multiclass
        # and return a HIGHER-IS-BETTER multiclass score.
        score = evaluate_model(X_filtered, y, params)
        return score

    sampler_initial = optuna.samplers.TPESampler(seed=42)
    study_initial = optuna.create_study(direction="maximize", sampler=sampler_initial)
    study_initial.optimize(objective_initial, n_trials=N_TRIALS)

    best_params_initial = study_initial.best_params
    print("\nInitial Best Multiclass AUC:", study_initial.best_value)
    print("Initial Best Params:", best_params_initial)

    # ======================================================
    # STEP 3: SHAP FEATURE SELECTION
    # ======================================================

    importance_df = get_fold_shap_importance(X_filtered, y, best_params_initial)
    importance_df = importance_df.sort_values("mean_shap", ascending=False)
    importance_df["pct"] = importance_df["mean_shap"] / importance_df["mean_shap"].sum()

    importance_df["cum_pct"] = importance_df["pct"].cumsum()
    selected_features = importance_df[importance_df["cum_pct"] <= 0.95].index.tolist()

    if len(selected_features) == 0:
        selected_features = [importance_df.index[0]]

    print(f"\nFeatures Before: " f"{X_filtered.shape[1]}")

    print(f"Features After SHAP: " f"{len(selected_features)}")

    importance_df.to_csv(
        os.path.join(IMPORTANCE_FOLDER, f"SHAP_IMPORTANCE_{target}_{SYMBOL}.csv")
    )

    # ======================================================
    # STEP 4: FINAL OPTUNA TUNING
    # ======================================================

    X_final = X_filtered[selected_features]

    def objective_final(trial):

        params = {
            "objective": "multiclass",
            "num_class": NUM_CLASSES,
            "metric": "multi_logloss",
            "n_estimators": trial.suggest_int("n_estimators", 100, 900),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "class_weight": class_weight_dict,
        }

        return evaluate_model(X_final, y, params)

    sampler_final = optuna.samplers.TPESampler(seed=42)
    study_final = optuna.create_study(direction="maximize", sampler=sampler_final)
    study_final.optimize(objective_final, n_trials=20)

    best_params_final = study_final.best_params

    print("\nFinal Best Multiclass AUC:", study_final.best_value)
    print("Final Best Params:", best_params_final)

    # ======================================================
    # STEP 5: TRAIN FINAL 5-CLASS MODEL
    # ======================================================

    final_model = LGBMClassifier(
        objective="multiclass",
        num_class=NUM_CLASSES,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight_dict,
        **best_params_final,
    )

    # ------------------------------------------------------
    # Chronological validation split
    # ------------------------------------------------------
    split = int(len(X_final) * 0.9)

    X_train = X_final.iloc[:split]
    X_val = X_final.iloc[split:]

    y_train = y.iloc[:split]
    y_val = y.iloc[split:]

    print("\nFinal Training Rows:", len(X_train))
    print("Final Validation Rows:", len(X_val))

    # ------------------------------------------------------
    # Train
    # ------------------------------------------------------
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[early_stopping(50), log_evaluation(0)],
    )

    # ------------------------------------------------------
    # Safe best iteration
    # ------------------------------------------------------
    best_iter = final_model.best_iteration_

    if best_iter is None:
        best_iter = final_model.n_estimators

    # ======================================================
    # STEP 6: VALIDATION PREDICTIONS
    # ======================================================

    val_proba = final_model.predict_proba(X_val, num_iteration=best_iter)

    # Predicted class
    val_pred_class = np.argmax(val_proba, axis=1)

    # ------------------------------------------------------
    # Multiclass ROC-AUC
    # ------------------------------------------------------
    val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr", average="macro")

    # ------------------------------------------------------
    # Validation accuracy
    # ------------------------------------------------------
    val_accuracy = accuracy_score(y_val, val_pred_class)

    # ------------------------------------------------------
    # Validation log loss
    # ------------------------------------------------------
    val_logloss = log_loss(y_val, val_proba, labels=np.arange(NUM_CLASSES))

    print("\n==============================")
    print("FINAL VALIDATION RESULTS")
    print("==============================")

    print(f"Multiclass ROC-AUC : {val_auc:.5f}")

    print(f"Accuracy           : " f"{val_accuracy:.5f}")

    print(f"Log Loss           : " f"{val_logloss:.5f}")

    print(f"Best Iteration     : " f"{best_iter}")

    # ======================================================
    # STEP 7: SAVE COMPLETE MODEL ARTIFACT
    # ======================================================

    model_artifact = {
        # ----------------------------------------------
        # Trained model
        # ----------------------------------------------
        "model": final_model,
        # ----------------------------------------------
        # Exact features used by the model
        # ----------------------------------------------
        "features": selected_features,
        # ----------------------------------------------
        # Target name
        # ----------------------------------------------
        "target": target,
        # ----------------------------------------------
        # COMPLETE TARGET + MOVEMENT INFORMATION
        "target_settings": target_settings,
        # ----------------------------------------------
        # Hyperparameters
        # ----------------------------------------------
        "params": best_params_final,
        # ----------------------------------------------
        # Best iteration
        # ----------------------------------------------
        "best_iteration": best_iter,
        # ----------------------------------------------
        # Validation metrics
        # ----------------------------------------------
        "val_auc": val_auc,
        "val_accuracy": val_accuracy,
        "val_logloss": val_logloss,
        # ----------------------------------------------
        # Feature importance
        # ----------------------------------------------
        "feature_importance": (final_model.feature_importances_.tolist()),
    }

    model_path = os.path.join(
        MODEL_FOLDER, f"HL_MAIN_{target}_{CD_TIME}_{SYMBOL}_model.pkl"
    )

    joblib.dump(model_artifact, model_path)

    print(f"\n💾 Model saved: {target} | Multiclass AUC: {val_auc:.5f}")
