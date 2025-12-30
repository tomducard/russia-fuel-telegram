"""Baseline modeling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

@dataclass
class ModelResult:
    model: Pipeline
    feature_cols: List[str]
    train_mse: float = 0.0
    test_mse: float = 0.0
    accuracy: float = 0.0
    report: str = ""
    roc_auc: float = 0.0
    f1_score: float = 0.0


def _time_split(df: pd.DataFrame, train_ratio: float = 0.8) -> int:
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to perform a time-based split.")
    split_idx = max(1, int(len(df) * train_ratio))
    return min(split_idx, len(df) - 1)


def get_feature_selection(df: pd.DataFrame, feature_cols: Optional[Sequence[str]] = None) -> List[str]:
    # Dynamic feature selection: Use all available signal columns
    if feature_cols:
        return list(feature_cols)
    
    # Default strategy: grab all relevant numeric signals
    base_candidates = {
        "total_messages", "unique_messages", "keyword_mentions", 
        "price_mentions", "avg_price", "fuel_stress_index"
    }
    all_cols = set(df.columns)
    
    # Add dynamic group columns (share_*, count_*, sentiment_*, price_*)
    dynamic_features = {
        c for c in all_cols 
        if c.startswith(("share_", "count_", "sentiment_", "price_"))
    }
    
    # Combine valid base features + dynamic features
    return list((base_candidates & all_cols) | dynamic_features)


def train_baseline(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    train_ratio: float = 0.8,
) -> ModelResult:
    """Train a Ridge regression on time-ordered data."""
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in DataFrame.")

    data = df.dropna(subset=[target_col]).copy()
    if "date" in data.columns:
        data = data.sort_values("date")

    features = get_feature_selection(data, feature_cols)
        
    if not features:
        raise ValueError("No feature columns available for training.")

    split_idx = _time_split(data, train_ratio=train_ratio)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    X_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col].astype(float)
    X_test = test_df[features].fillna(0.0)
    y_test = test_df[target_col].astype(float)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)
    train_mse = float(mean_squared_error(y_train, train_pred))
    test_mse = float(mean_squared_error(y_test, test_pred))

    return ModelResult(
        model=pipeline,
        feature_cols=features,
        train_mse=train_mse,
        test_mse=test_mse,
    )


from xgboost import XGBClassifier

def train_classifier(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    train_ratio: float = 0.8,
    model_type: str = "rf",
) -> ModelResult:
    """Train a Classifier (Random Forest, XGBoost, MLP, or Gradient Boosting) on time-ordered data."""
    if target_col not in df.columns:
        raise ValueError(f"target column '{target_col}' not found in DataFrame.")

    data = df.dropna(subset=[target_col]).copy()
    if "date" in data.columns:
        data = data.sort_values("date")

    features = get_feature_selection(data, feature_cols)
        
    if not features:
        raise ValueError("No feature columns available for training.")

    split_idx = _time_split(data, train_ratio=train_ratio)
    train_df = data.iloc[:split_idx]
    test_df = data.iloc[split_idx:]

    X_train = train_df[features].fillna(0.0)
    y_train = train_df[target_col].astype(int)
    X_test = test_df[features].fillna(0.0)
    y_test = test_df[target_col].astype(int)

    if model_type == "xgb":
        # XGBoost configuration with Hyperparameter Tuning
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from scipy.stats import uniform, randint
        
        print("Running Randomized Search for XGBoost hyperparameters...")
        
        base_clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        
        # Hyperparameter Grid
        param_dist = {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 10),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4), # 0.6 to 1.0
            "colsample_bytree": uniform(0.6, 0.4),
            "scale_pos_weight": [1, 5, 9, 12, 15], # Test different imbalance weights
            "gamma": uniform(0, 0.5)
        }
        
        search = RandomizedSearchCV(
            estimator=base_clf,
            param_distributions=param_dist,
            n_iter=20, # Test 20 random combinations
            scoring="f1", # Optimize for F1 Score directly
            cv=StratifiedKFold(n_splits=3), # Robust cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1 # Use all cores
        )
        
        search.fit(X_train, y_train)
        print(f"Best Parameters: {search.best_params_}")
        clf = search.best_estimator_
    elif model_type == "rf":
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    elif model_type == "gb":
        clf = GradientBoostingClassifier(random_state=42)
    elif model_type == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=42,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", clf),
        ]
    )
    pipeline.fit(X_train, y_train)

    from sklearn.metrics import roc_auc_score, f1_score

    # --- Threshold Tuning ---
    # Standard .predict() uses 0.5. For imbalanced data, we optimize this.
    train_proba = pipeline.predict_proba(X_train)[:, 1]
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Grid search for best threshold on TRAINING data
    thresholds = np.arange(0.1, 0.9, 0.05)
    for thresh in thresholds:
        train_pred_t = (train_proba >= thresh).astype(int)
        f1_t = f1_score(y_train, train_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
            
    print(f"Optimized Decision Threshold: {best_thresh:.2f} (Max Train F1: {best_f1:.2f})")

    # Apply optimal threshold to TEST predictions
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_thresh).astype(int)
    
    accuracy = float(accuracy_score(y_test, test_pred))
    
    # New Coherent Metrics
    from sklearn.metrics import roc_auc_score
    try:
        roc_auc = float(roc_auc_score(y_test, test_proba))
    except ValueError:
        roc_auc = 0.5 # Default if only one class in test set
        
    f1 = float(f1_score(y_test, test_pred, zero_division=0))
    
    report = classification_report(y_test, test_pred, zero_division=0)

    return ModelResult(
        model=pipeline,
        feature_cols=features,
        accuracy=accuracy,
        report=report,
        roc_auc=roc_auc,
        f1_score=f1
    )


# --- LSTM Implementation ---

def create_sequences(X, y, time_steps=30):
    """Convert 2D array into 3D sequences (Samples, TimeSteps, Features)."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    train_ratio: float = 0.8,
    time_steps: int = 30
) -> ModelResult:
    """Train an LSTM model for sequence classification."""
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # Prepare Data
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} missing.")
    
    data = df.dropna(subset=[target_col]).sort_values("date")
    features = get_feature_selection(data, feature_cols)
    print(f"LSTM Features ({len(features)}): {features}")
    
    # Scale Data (Crucial for LSTM)
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
    target = data[target_col].astype(int) # Fix for bincount inputs
    
    # Create Sequences
    X_seq, y_seq = create_sequences(scaled_features, target, time_steps)
    
    # Split Time-Based
    split_idx = int(len(X_seq) * train_ratio)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"LSTM Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
    
    # Compute Class Weights usually
    neg, pos = np.bincount(y_train)
    total = neg + pos
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)
    class_weight = {0: weight_for_0, 1: weight_for_1}
    print(f"Class Weights: {class_weight}")

    # Build Model
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=50, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='binary_crossentropy', 
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        validation_split=0.2, # Validation on end of TRAIN set
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )
    
    # Evaluate on Test Set
    test_proba = model.predict(X_test).flatten()
    
    # Dynamic Threshold Tuning (Reuse logic)
    # We predict on train again to find best threshold
    train_proba = model.predict(X_train).flatten()
    best_thresh = 0.5
    best_f1 = 0.0
    
    # Import locally to avoid top-level dependency if not installed
    from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
    
    for thresh in np.arange(0.1, 0.9, 0.05):
        train_pred_t = (train_proba >= thresh).astype(int)
        f1_t = f1_score(y_train, train_pred_t, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = thresh
            
    print(f"LSTM Best Threshold: {best_thresh:.2f} (Train F1: {best_f1:.2f})")
    
    test_pred = (test_proba >= best_thresh).astype(int)
    
    accuracy = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, test_proba)
    except ValueError:
        roc_auc = 0.5
        
    report = classification_report(y_test, test_pred, zero_division=0)
    
    return ModelResult(
        model=model, # Returns Keras model
        feature_cols=features,
        accuracy=accuracy,
        report=report,
        roc_auc=roc_auc,
        f1_score=f1
    )
