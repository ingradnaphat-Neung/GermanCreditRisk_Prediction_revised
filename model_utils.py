"""Shared German Credit (OpenML id=31) feature schema and threshold helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    'checking_status', 'duration', 'credit_history', 'purpose',
    'credit_amount', 'savings_status', 'employment',
    'installment_commitment', 'personal_status', 'other_parties',
    'residence_since', 'property_magnitude', 'age',
    'other_payment_plans', 'housing', 'existing_credits', 'job',
    'num_dependents', 'own_telephone', 'foreign_worker',
]

NUMERICAL = [
    'duration', 'credit_amount', 'installment_commitment',
    'residence_since', 'age', 'existing_credits', 'num_dependents',
]

CATEGORICAL = [x for x in FEATURES if x not in NUMERICAL]

LABELS = {
    'checking_status': 'สถานะบัญชีกระแสรายวัน (Checking status)',
    'duration': 'ระยะเวลาสินเชื่อ (เดือน)',
    'credit_history': 'ประวัติเครดิต (Credit history)',
    'purpose': 'วัตถุประสงค์สินเชื่อ (Purpose)',
    'credit_amount': 'วงเงินกู้ (Deutsche Mark: DM)',
    'savings_status': 'สถานะเงินออม (Savings status)',
    'employment': 'ระยะเวลาการทำงาน (Employment)',
    'installment_commitment': 'ระดับภาระผ่อนชำระ (1–4)',
    'personal_status': 'สถานะครอบครัวและเพศ (Personal status)',
    'other_parties': 'ผู้ร่วมกู้/ผู้ค้ำประกัน (Other parties)',
    'residence_since': 'ระดับระยะเวลาอยู่อาศัย (1–4)',
    'property_magnitude': 'ทรัพย์สิน (Property)',
    'age': 'อายุ (ปี)',
    'other_payment_plans': 'แผนชำระหนี้อื่น',
    'housing': 'ลักษณะที่พักอาศัย (Housing)',
    'existing_credits': 'จำนวนบัญชีเครดิตที่เปิดอยู่',
    'job': 'ประเภทอาชีพ (Job)',
    'num_dependents': 'จำนวนผู้ที่ต้องอุปการะ',
    'own_telephone': 'โทรศัพท์จดทะเบียนในชื่อตน',
    'foreign_worker': 'สถานะแรงงานต่างชาติ',
}

DEFAULTS = {
    'checking_status': 'no checking',
    'duration': 24,
    'credit_history': 'existing paid',
    'purpose': 'radio/tv',
    'credit_amount': 3000,
    'savings_status': '<100',
    'employment': '1<=X<4',
    'installment_commitment': 3,
    'personal_status': 'male single',
    'other_parties': 'none',
    'residence_since': 2,
    'property_magnitude': 'car',
    'age': 35,
    'other_payment_plans': 'none',
    'housing': 'own',
    'existing_credits': 1,
    'job': 'skilled',
    'num_dependents': 1,
    'own_telephone': 'none',
    'foreign_worker': 'yes',
}

NUMERIC_LIMITS = {
    'duration': (4, 72), 'credit_amount': (250, 18424),
    'installment_commitment': (1, 4), 'residence_since': (1, 4),
    'age': (19, 75), 'existing_credits': (1, 4),
    'num_dependents': (1, 2),
}

GROUPS = {
    'ข้อมูลสินเชื่อ': [
        'checking_status', 'duration', 'credit_history', 'purpose',
        'credit_amount', 'savings_status', 'installment_commitment',
        'other_payment_plans', 'existing_credits',
    ],
    'ข้อมูลการทำงานและผู้สมัคร': [
        'employment', 'personal_status', 'age', 'job',
        'num_dependents', 'foreign_worker',
    ],
    'ข้อมูลที่พักอาศัยและอื่น ๆ': [
        'other_parties', 'residence_since', 'property_magnitude',
        'housing', 'own_telephone',
    ],
}


def optimal_cost_threshold(y_true, bad_prob, cost_fn=5, cost_fp=1):
    """Mirror the notebook's OOF grid search and FN:FP = 5:1 business costs."""
    y = np.asarray(y_true, dtype=int)
    proba = np.asarray(bad_prob, dtype=float)
    if y.ndim != 1 or proba.ndim != 1 or len(y) != len(proba) or len(y) == 0:
        raise ValueError('y_true and bad_prob must be nonempty, equally sized 1-D arrays')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(proba).all():
        raise ValueError('Invalid target or probability')
    records = []
    for th in np.linspace(0, 1, 101):
        pred = proba >= th
        fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum())
        records.append((round(float(th), 2), fn * cost_fn + fp * cost_fp))
    # Python min keeps the lowest threshold in a tie, matching the notebook.
    best_th, best_cost = min(records, key=lambda x: x[1])
    return best_th, int(best_cost), pd.DataFrame(records, columns=['threshold','total_cost'])


def build_input_dataframe(row: dict) -> pd.DataFrame:
    missing = [name for name in FEATURES if name not in row]
    if missing:
        raise ValueError(f'Missing features: {missing}')
    row = {name: row[name] for name in FEATURES}
    frame = pd.DataFrame([row], columns=FEATURES)
    for name in NUMERICAL:
        frame[name] = pd.to_numeric(frame[name], errors='raise')
    for name in CATEGORICAL:
        frame[name] = frame[name].astype(str)
    return frame


def get_categories(fitted_pipeline, categorical_cols: list[str]) -> dict[str, list[str]]:
    preprocess = fitted_pipeline.named_steps['preprocessor']
    encoder = preprocess.named_transformers_['cat'].named_steps['onehot']
    return {
        name: [str(v) for v in values]
        for name, values in zip(categorical_cols, encoder.categories_)
    }

# Sensitive attributes below are used to *audit* test predictions, not to
# modify a prediction or choose a threshold for any demographic group.
# The trained models still use age/personal_status as predictors.  The sex
# attribute is an imperfect proxy extracted from a composite dataset field.
def derived_sex(personal_status: object) -> str:
    """Infer dataset-coded sex from the German Credit personal_status field.

    IMPORTANT: this is a proxy derived from a composite feature, not an
    independently verified demographic field and not a gender identity.
    """
    status = str(personal_status).strip().lower()
    if status.startswith('female'):
        return 'female'
    if status.startswith('male'):
        return 'male'
    return 'unknown'


def derived_age_group(age: object) -> str:
    """Reproduce Final.ipynb's age split (age <= 25 versus > 25)."""
    if pd.isna(age):
        return 'unknown'
    age_num = float(age)
    if not np.isfinite(age_num):
        return 'unknown'
    return 'Young (<=25)' if age_num <= 25 else 'Old (>25)'


def _safe_rate(numerator: int, denominator: int) -> float:
    """Undefined conditional rates remain NaN, never a misleading zero."""
    return float(numerator / denominator) if denominator else float('nan')


def group_fairness_metrics(
    X_test: pd.DataFrame, y_test, bad_probability, threshold: float,
    dimension: str,
) -> tuple[pd.DataFrame, dict]:
    """Fairness diagnostics for a *held-out* test set at a fixed threshold.

    Positive class is bad=1, so selection_rate means the fraction FLAGGED BAD,
    not the fraction approved. TPR is the fraction of actual bad cases flagged.
    Rates whose denominators are empty are undefined (NaN).
    """
    if dimension not in ('sex', 'age_group'):
        raise ValueError('dimension must be sex or age_group')
    if not isinstance(X_test, pd.DataFrame) or not {'personal_status', 'age'}.issubset(X_test.columns):
        raise ValueError('X_test needs personal_status and age columns')
    y = np.asarray(y_test, dtype=int)
    p = np.asarray(bad_probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(X_test) != len(y) or len(y) != len(p):
        raise ValueError('X_test, y_test and probabilities must have matching lengths')
    if not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise ValueError('Invalid y_test or probability values')
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between zero and one')

    # Turn X_test into positional series to avoid pandas index alignment gotchas.
    group_values = (
        X_test['personal_status'].map(derived_sex)
        if dimension == 'sex' else X_test['age'].map(derived_age_group)
    ).to_numpy()
    pred = (p >= threshold).astype(int)
    rows = []
    for group in sorted(set(group_values)):
        mask = group_values == group
        truth, guess = y[mask], pred[mask]
        tp = int(np.sum((truth == 1) & (guess == 1)))
        tn = int(np.sum((truth == 0) & (guess == 0)))
        fp = int(np.sum((truth == 0) & (guess == 1)))
        fn = int(np.sum((truth == 1) & (guess == 0)))
        count = int(mask.sum())
        rows.append({
            'group': group, 'n': count, 'actual_bad': int(np.sum(truth == 1)),
            'actual_good': int(np.sum(truth == 0)),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'selection_rate': _safe_rate(tp + fp, count),
            'tpr': _safe_rate(tp, tp + fn),
            'fpr': _safe_rate(fp, fp + tn),
            'bad_base_rate': _safe_rate(tp + fn, count),
        })
    table = pd.DataFrame(rows)
    gaps: dict = {'group_count': len(table)}
    for field in ('selection_rate', 'tpr', 'fpr'):
        values = table[field].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        # A gap across only one valid group cannot be estimated; return NaN.
        gaps[field + '_gap'] = float(valid.max() - valid.min()) if len(valid) >= 2 else float('nan')
    tpr_gap = gaps['tpr_gap']
    fpr_gap = gaps['fpr_gap']
    gaps['equalized_odds_gap'] = (
        float(max(tpr_gap, fpr_gap))
        if np.isfinite(tpr_gap) and np.isfinite(fpr_gap) else float('nan')
    )
    return table, gaps
