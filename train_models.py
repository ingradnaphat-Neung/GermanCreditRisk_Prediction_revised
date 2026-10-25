"""Train all three German Credit models exactly with Final.ipynb's CV grids.

The hold-out test set is never used during hyperparameter or threshold selection.
Public dataset: OpenML credit-g, data_id=31. No private inputs are uploaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_utils import (FEATURES, NUMERICAL, CATEGORICAL,
                         get_categories, optimal_cost_threshold, group_fairness_metrics)

BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE / 'credit_models.joblib'
MODEL_NAMES = ('Logistic Regression', 'Random Forest', 'Gradient Boosting')
# Hyperparameter grids copied from the supplied notebook (Final.ipynb).
PARAM_GRIDS = {
    'Logistic Regression': {'classifier__C': [0.01, 0.1, 1.0, 10.0], 'classifier__penalty': ['l2']},
    'Random Forest': {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_split': [2, 5],
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [100, 150],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__max_depth': [3, 5],
    },
}


def make_pipeline(name: str) -> Pipeline:
    """Each model gets its own independent preprocessor (no cross-fit sharing)."""
    classifiers = {
        'Logistic Regression': lambda: LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': lambda: RandomForestClassifier(random_state=42, n_jobs=1),
        'Gradient Boosting': lambda: GradientBoostingClassifier(random_state=42),
    }
    if name not in classifiers:
        raise ValueError(f'Unknown model: {name}')
    preprocessing = ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]), NUMERICAL),
        ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), CATEGORICAL),
    ])
    return Pipeline([('preprocessor', preprocessing), ('classifier', classifiers[name]())])


def classification_report(y_true, probability: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probability = np.asarray(probability).astype(float)
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'cost': int(5 * fn + fp),
    }


def compute_rf_shap(fitted_pipeline: Pipeline, X_test: pd.DataFrame) -> dict:
    """Optional SHAP summary of class bad=1; explain a test-set subset.

    SHAP arrays vary across versions: legacy [class0, class1] and current
    (samples, transformed_features, classes) are both supported.
    """
    try:
        import shap
        preprocessor = fitted_pipeline.named_steps['preprocessor']
        classifier = fitted_pipeline.named_steps['classifier']
        # Deterministically sample from held-out test observations only.
        sample = X_test.sample(n=min(80, len(X_test)), random_state=42)
        transformed = preprocessor.transform(sample)
        if hasattr(transformed, 'toarray'):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=float)
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            class_bad_values = np.asarray(values[1], dtype=float)
        else:
            arr = np.asarray(values, dtype=float)
            class_bad_values = arr[:, :, 1] if arr.ndim == 3 else arr
        if class_bad_values.shape != transformed.shape:
            raise ValueError(f'Unexpected SHAP shape {class_bad_values.shape} vs {transformed.shape}')
        # Group OneHotEncoder outputs back into the original 20 features.
        # ColumnTransformer output is numerical columns first, then categorical.
        encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        raw_means = np.mean(np.abs(class_bad_values), axis=0)
        summaries = []
        cursor = 0
        for feature in NUMERICAL:
            summaries.append({'feature': feature, 'mean_abs_shap': float(raw_means[cursor])})
            cursor += 1
        for feature, categories in zip(CATEGORICAL, encoder.categories_):
            width = len(categories)
            summaries.append({
                'feature': feature,
                'mean_abs_shap': float(raw_means[cursor:cursor + width].sum()),
            })
            cursor += width
        if cursor != len(raw_means):
            raise ValueError('Unexpected encoded-column count for SHAP aggregation')
        summaries.sort(key=lambda row: row['mean_abs_shap'], reverse=True)
        return {'status': 'computed', 'n_test_sampled': int(len(sample)),
                'algorithm': 'TreeExplainer', 'class': 'bad',
                'unit': 'SHAP output units of the Random Forest bad-class probability',
                'feature_importance': summaries}
    except (ImportError, RuntimeError, ValueError, TypeError) as exc:
        # Permutation Importance is still available; report exact optional failure.
        return {'status': 'unavailable', 'reason': str(exc)}


def train_models(dataset_frame: pd.DataFrame | None = None,
                 param_grids: dict | None = None,
                 folds: int = 5, with_shap: bool = False) -> dict:
    """Train/compare LR, RF and GB on the same stratified 80/20 split.

    dataset_frame is injectable for unit testing and offline/local datasets.
    The full production run uses the public OpenML dataset and 5-fold CV.
    """
    public_fetch = dataset_frame is None
    if dataset_frame is None:
        dataset_frame = fetch_openml(data_id=31, as_frame=True, parser='auto').frame
    expected = set(FEATURES) | {'class'}
    if not expected.issubset(dataset_frame.columns):
        raise ValueError(f'Missing dataset columns: {sorted(expected - set(dataset_frame.columns))}')
    if dataset_frame[FEATURES + ['class']].isna().any().any():
        raise ValueError('Input dataset contains missing values; notebook assumes none')
    classes = set(dataset_frame['class'].astype(str).unique())
    if classes != {'good', 'bad'}:
        raise ValueError(f'Expected good/bad labels, got {classes}')

    # Keep the same data and same test split across every model.
    X = dataset_frame[FEATURES].copy()
    y = (dataset_frame['class'].astype(str) == 'bad').astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    result = {
        'artifact_version': 3,
        'source': 'OpenML credit-g, data_id=31' if public_fetch else 'Supplied local dataframe (offline/test)' ,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sklearn_version': sklearn.__version__,
        'feature_columns': FEATURES,
        'numerical_cols': NUMERICAL,
        'categorical_cols': CATEGORICAL,
        'dataset_samples': int(len(X)),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'test_bad_count': int(y_test.sum()),
        'cv_folds': int(folds),
        'split_random_state': 42,
        'positive_class': 'bad',
        'explainability': 'Permutation Importance on untouched test set; ROC-AUC; 10 repeats; random_state=42',
        'fairness_definition': 'sex proxy from personal_status; age <=25 vs >25; positive=bad (flagged); test set only',
        'cost_fn': 5, 'cost_fp': 1,
        'models': {},
    }
    grids = param_grids if param_grids is not None else PARAM_GRIDS
    for name in MODEL_NAMES:
        print(f'[{name}] GridSearchCV {folds}-fold...', flush=True)
        search = GridSearchCV(
            make_pipeline(name), grids[name], scoring='roc_auc', cv=cv,
            n_jobs=1, refit=True, error_score='raise',
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        # Model-specific threshold fitted on OOF TRAIN predictions only.
        oof = cross_val_predict(
            best, X_train, y_train, cv=cv, method='predict_proba', n_jobs=1
        )[:, 1]
        cost_threshold, oof_cost, cost_curve = optimal_cost_threshold(y_train, oof)
        held_out_prob = best.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, held_out_prob, pos_label=1)

        # Explainability is computed only AFTER model & threshold selection.
        # On the original feature columns, pipeline permutation importance
        # automatically re-applies the fitted ColumnTransformer on each call.
        importances = permutation_importance(
            best, X_test, y_test, n_repeats=10, random_state=42,
            scoring='roc_auc', n_jobs=1,
        )
        importance_rows = sorted((
            {'feature': feat, 'importance_mean': float(mu),
             'importance_std': float(sd)}
            for feat, mu, sd in zip(
                FEATURES, importances.importances_mean, importances.importances_std
            )
        ), key=lambda r: r['importance_mean'], reverse=True)

        # Fairness is an audit, NEVER used for choosing the model/threshold.
        fairness = {}
        for threshold_name, threshold_value in (
            ('cost', cost_threshold), ('0.5', 0.5),
        ):
            fairness[threshold_name] = {}
            for dimension in ('sex', 'age_group'):
                table, gaps = group_fairness_metrics(
                    X_test, y_test, held_out_prob, threshold_value, dimension,
                )
                fairness[threshold_name][dimension] = {
                    'groups': table.to_dict(orient='records'), 'gaps': gaps,
                }

        result['models'][name] = {
            'model': best,
            'cv_roc_auc': float(search.best_score_),
            'test_roc_auc': float(roc_auc_score(y_test, held_out_prob)),
            'threshold_cost': float(cost_threshold),
            'oof_cost': int(oof_cost),
            'metrics_05': classification_report(y_test, held_out_prob, 0.5),
            'metrics_cost': classification_report(y_test, held_out_prob, cost_threshold),
            'best_params': search.best_params_,
            'permutation_importance': importance_rows,
            'fairness': fairness,
            'shap': {'status': 'not_computed', 'reason': 'Optional: use --with-shap to compute SHAP on Random Forest test observations'},
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'cost_curve': cost_curve.to_dict(orient='records'),
            'categories': get_categories(best, CATEGORICAL),
        }
        if with_shap and name == 'Random Forest':
            result['models'][name]['shap'] = compute_rf_shap(best, X_test)
        print(f'  CV AUC={search.best_score_:.4f}; Test AUC={result["models"][name]["test_roc_auc"]:.4f}; threshold={cost_threshold:.2f}', flush=True)
    result['selected_cv_model'] = max(MODEL_NAMES, key=lambda n: result['models'][n]['cv_roc_auc'])
    return result


def train_and_save(output_path: Path = DEFAULT_OUTPUT, dataset_frame=None,
                   param_grids=None, folds=5, with_shap=False) -> dict:
    artifact = train_models(dataset_frame=dataset_frame, param_grids=param_grids, folds=folds, with_shap=with_shap)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    return artifact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train all models in Final.ipynb')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--data', type=Path, default=None, help='Optional local CSV with OpenML credit-g schema')
    parser.add_argument('--with-shap', action='store_true', help='Optional SHAP TreeExplainer for Random Forest on held-out test observations; install requirements-shap.txt first')
    args = parser.parse_args()
    dataframe = pd.read_csv(args.data) if args.data is not None else None
    trained = train_and_save(args.output, dataset_frame=dataframe, with_shap=args.with_shap)
    print(f'Saved all models: {args.output.resolve()}')
