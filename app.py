"""Thai Streamlit dashboard: compare every algorithm in Final.ipynb.

Uses only public German Credit dataset for model training. User form inputs
stay in Streamlit session state; the application does not send them to APIs.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from model_utils import (CATEGORICAL, DEFAULTS, FEATURES, GROUPS, LABELS,
                         NUMERICAL, NUMERIC_LIMITS, build_input_dataframe)
from train_models import DEFAULT_OUTPUT, MODEL_NAMES

st.set_page_config(
    page_title='Credit Risk Lab | Model Comparison',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown('''
<style>
  .hero {background: linear-gradient(105deg,#173A73,#1566A0);padding:24px 28px;
      border-radius:16px;color:white;margin-bottom:16px}
  .hero h1 {color:white;font-size:2.05rem;font-weight:750;margin:0 0 8px}
  .hero p {color:#E9F3FD;margin:0;font-size:1.02rem}
  .section-kicker {color:#3867A0;text-transform:uppercase;letter-spacing:.08em;
      font-size:.78rem;font-weight:700;margin:8px 0}
  div[data-testid="stMetric"] {background:#F1F6FC;padding:12px 14px;border-radius:12px}
</style>
''', unsafe_allow_html=True)

# Human-readable labels always preserve the original training category value.
OPTION_LABELS = {
    'checking_status': {
        '<0': 'ยอดติดลบ (< 0 DM)', '0<=X<200': '0 ถึงต่ำกว่า 200 DM',
        '>=200': 'ตั้งแต่ 200 DM', 'no checking': 'ไม่มีบัญชีกระแสรายวัน',
    },
    'credit_history': {
        'critical/other existing credit': 'มีเครดิตค้างชำระ / ประวัติมีปัญหา',
        'existing paid': 'ชำระเครดิตเดิมตามปกติ',
        'delayed previously': 'เคยชำระล่าช้า',
        'no credits/all paid': 'ไม่มีเครดิต / ชำระแล้ว',
        'all paid': 'เครดิตทั้งหมดชำระครบแล้ว',
    },
    'purpose': {
        'radio/tv': 'วิทยุ / โทรทัศน์', 'education': 'การศึกษา',
        'furniture/equipment': 'เฟอร์นิเจอร์ / อุปกรณ์',
        'new car': 'รถยนต์ใหม่', 'used car': 'รถยนต์มือสอง',
        'business': 'ธุรกิจ', 'domestic appliance': 'เครื่องใช้ในบ้าน',
        'repairs': 'ซ่อมแซม', 'other': 'อื่น ๆ', 'retraining': 'ฝึกอบรมใหม่',
    },
    'savings_status': {
        '<100': 'ต่ำกว่า 100 DM', '100<=X<500': '100 ถึงต่ำกว่า 500 DM',
        '500<=X<1000': '500 ถึงต่ำกว่า 1,000 DM', '>=1000': 'ตั้งแต่ 1,000 DM',
        'no known savings': 'ไม่มีข้อมูลเงินออม',
    },
    'employment': {
        'unemployed': 'ว่างงาน', '<1': 'น้อยกว่า 1 ปี',
        '1<=X<4': '1 ถึงต่ำกว่า 4 ปี', '4<=X<7': '4 ถึงต่ำกว่า 7 ปี',
        '>=7': 'ตั้งแต่ 7 ปี',
    },
    'personal_status': {
        'male single': 'ชาย / โสด', 'male mar/wid': 'ชาย / สมรสหรือหม้าย',
        'male div/sep': 'ชาย / หย่าหรือแยกกันอยู่',
        'female div/dep/mar': 'หญิง / ตามหมวดหมู่ชุดข้อมูลต้นฉบับ',
    },
    'other_parties': {
        'none': 'ไม่มี', 'guarantor': 'ผู้ค้ำประกัน', 'co-applicant': 'ผู้ร่วมกู้',
    },
    'property_magnitude': {
        'real estate': 'อสังหาริมทรัพย์', 'car': 'รถยนต์',
        'life insurance': 'ประกันชีวิต', 'no known property': 'ไม่มีข้อมูลทรัพย์สิน',
    },
    'other_payment_plans': {
        'none': 'ไม่มี', 'bank': 'ธนาคาร', 'stores': 'ร้านค้า',
    },
    'housing': {'own': 'บ้านตนเอง', 'rent': 'เช่า', 'for free': 'อยู่ฟรี'},
    'job': {
        'skilled': 'พนักงานที่มีทักษะ',
        'unskilled resident': 'แรงงานทั่วไปที่มีถิ่นพำนัก',
        'high qualif/self emp/mgmt': 'ผู้เชี่ยวชาญ / ธุรกิจส่วนตัว / ผู้บริหาร',
        'unemployed/unskilled non-resident': 'ว่างงาน / แรงงานไม่มีถิ่นพำนัก',
    },
    'own_telephone': {'yes': 'มี', 'none': 'ไม่มี'},
    'foreign_worker': {'yes': 'ใช่', 'no': 'ไม่ใช่'},
}

QUICK_FIELDS = [
    'credit_amount', 'duration', 'checking_status', 'credit_history',
    'savings_status', 'employment', 'age',
]
ADVANCED_FIELDS = [f for f in FEATURES if f not in QUICK_FIELDS]
METRIC_LABELS = {
    'cv_roc_auc': 'Cross-validation ROC-AUC (Train)',
    'test_roc_auc': 'ROC-AUC (Test)',
    'accuracy': 'Accuracy (Test)',
    'precision': 'Precision ของ Bad (Test)',
    'recall': 'Recall ของ Bad (Test)',
    'f1': 'F1-score ของ Bad (Test)',
}


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[dict, str]:
    if Path(DEFAULT_OUTPUT).exists():
        artifact = joblib.load(DEFAULT_OUTPUT)  # only local/trusted project file
        source = 'loaded'
    else:
        from train_models import train_and_save
        artifact = train_and_save(DEFAULT_OUTPUT)
        source = 'trained'
    if artifact.get('artifact_version') != 3 or set(artifact.get('models', {})) != set(MODEL_NAMES):
        raise ValueError('Incompatible model artifact; run: python train_models.py')
    return artifact, source


def input_control(field: str, categories: dict) -> object:
    label = LABELS[field]
    if field in NUMERICAL:
        low, high = NUMERIC_LIMITS[field]
        return st.number_input(
            label, min_value=int(low), max_value=int(high),
            value=int(DEFAULTS[field]), step=1, key=f'credit_{field}',
        )
    options = categories.get(field, [])
    if not options:
        st.error(f'ไม่พบหมวดหมู่ของ {field} ใน Encoder')
        st.stop()
    preferred = DEFAULTS.get(field)
    index = options.index(preferred) if preferred in options else 0
    return st.selectbox(
        label, options, index=index, key=f'credit_{field}',
        format_func=lambda val, fld=field: f'{OPTION_LABELS.get(fld, {}).get(val, val)}',
    )


def metric_value(record: dict, field: str, key: str) -> float:
    if field in ('cv_roc_auc', 'test_roc_auc'):
        return float(record[field])
    return float(record[key][field])


def comparison_frame(artifact: dict, mode: str) -> pd.DataFrame:
    key = 'metrics_cost' if mode == 'cost' else 'metrics_05'
    rows = []
    for name in MODEL_NAMES:
        rec = artifact['models'][name]
        m = rec[key]
        rows.append({
            'โมเดล': name,
            'CV ROC-AUC': rec['cv_roc_auc'],
            'Test ROC-AUC': rec['test_roc_auc'],
            'Accuracy': m['accuracy'],
            'Precision (Bad)': m['precision'],
            'Recall (Bad)': m['recall'],
            'F1-score (Bad)': m['f1'],
            'Threshold': m['threshold'],
            'FN': m['fn'], 'FP': m['fp'], 'Cost (5×FN+FP)': m['cost'],
        })
    return pd.DataFrame(rows)


def draw_roc_chart(artifact: dict):
    # Interpolation aligns curves on one shared x-axis without requiring Plotly.
    common_fpr = np.linspace(0, 1, 201)
    roc_df = pd.DataFrame(index=common_fpr)
    for name in MODEL_NAMES:
        roc = artifact['models'][name]['roc']
        roc_df[name] = np.interp(common_fpr, roc['fpr'], roc['tpr'])
    roc_df['Random baseline'] = common_fpr
    roc_df.index.name = 'False Positive Rate'
    st.line_chart(roc_df, x_label='False Positive Rate (FPR)', y_label='True Positive Rate (TPR)')


st.markdown('''<div class="hero"><h1>📊 Credit Risk Lab</h1>
<p>ประเมินความเสี่ยงสินเชื่อ • เปรียบเทียบ 3 Machine Learning Models • อ้างอิง German Credit</p>
</div>''', unsafe_allow_html=True)
st.caption('โปรเจกต์เพื่อการศึกษา/วิจัย | German Credit (OpenML #31) | Target: Bad Credit = 1')

with st.spinner('กำลังเตรียมโมเดลทั้ง 3 ตัว (การเปิดครั้งแรกอาจต้องดาวน์โหลดและฝึกจากข้อมูลสาธารณะ)...'):
    try:
        artifact, source = load_models()
    except Exception as error:
        st.error('ไม่สามารถโหลดหรือฝึกโมเดลทั้ง 3 ได้')
        st.write('หากเป็น Streamlit Cloud โปรดตรวจสอบว่าเชื่อมต่อ OpenML ได้ หรือฝึกโมเดลบนเครื่องก่อน แล้วเพิ่มไฟล์ `credit_models.joblib` ใน Repository')
        with st.expander('รายละเอียดข้อผิดพลาด'):
            st.code(str(error))
        st.code('python train_models.py')
        st.stop()

models = artifact['models']
with st.sidebar:
    st.header('⚙️ การตั้งค่าการเปรียบเทียบ')
    threshold_mode = st.radio(
        'รูปแบบ Threshold',
        ['cost', '0.5'],
        format_func=lambda v: (
            'Cost-based: ปรับเกณฑ์แยกตามโมเดล' if v == 'cost'
            else 'Standard: ใช้ 50% เท่ากันทุกโมเดล'
        ),
        help='Cost-based หา threshold จาก Out-of-Fold เฉพาะชุด Train โดยให้ FN มีต้นทุน 5 เท่าของ FP',
    )
    st.info('คำว่า Bad หมายถึงกลุ่มความเสี่ยงตามป้ายกำกับในชุดข้อมูล ไม่ใช่การตัดสินสิทธิผู้กู้')
    st.divider()
    st.caption(f'ข้อมูลฝึก {artifact["train_samples"]:,} | ทดสอบ {artifact["test_samples"]:,} รายการ')
    st.caption(f'Models: {len(models)} | CV: {artifact["cv_folds"]} folds')
    st.caption('โหลดผลฝึกจากไฟล์' if source == 'loaded' else 'ฝึกใหม่และบันทึกผลสำเร็จ')
    st.caption('ไม่บันทึกฟอร์มผู้สมัครลงฐานข้อมูล และไม่มี API สำหรับส่งข้อมูลผู้ใช้')

k1, k2, k3, k4 = st.columns(4)
k1.metric('จำนวนชุดข้อมูล', f'{artifact["dataset_samples"]:,}')
k2.metric('โมเดล', '3 โมเดล')
k3.metric('ชุดทดสอบ', f'{artifact["test_samples"]:,} รายการ')
best_cv = artifact['selected_cv_model']
k4.metric('CV ROC-AUC สูงสุด', f'{models[best_cv]["cv_roc_auc"]:.3f}', help=best_cv)
st.caption(f'โมเดลที่ CV ROC-AUC สูงสุดจากการฝึกครั้งนี้: **{best_cv}** (ไม่ได้หมายถึงดีที่สุดในทุกเกณฑ์)')

screen_tab, compare_tab, fairness_tab, guide_tab = st.tabs([
    '🧾 ประเมินความเสี่ยง', '📊 เปรียบเทียบโมเดล',
    '🔎 การอธิบายและความเป็นธรรม', '📘 วิธีใช้และข้อควรระวัง',
])

with screen_tab:
    st.subheader('1. ใส่ข้อมูลตัวอย่างสินเชื่อ')
    st.write('กรอกข้อมูลสำคัญด้านล่าง และกดเปิด **ข้อมูลเพิ่มเติม** หากต้องการปรับตัวแปรอื่น ๆ ให้ครบทั้ง 20 ตัว')
    st.info('ระบบใส่ค่าเริ่มต้นเพื่อใช้ทดลองเท่านั้น ควรตรวจสอบข้อมูลทั้ง 20 ช่องก่อนประเมิน ไม่ใช่ข้อมูลของลูกค้าจริง')
    # Every model's OneHotEncoder was fitted on the same training split.
    categories = models['Random Forest']['categories']
    values = {}
    with st.form('credit_profile', clear_on_submit=False):
        st.markdown('#### ข้อมูลสำคัญ')
        lc, rc = st.columns(2, gap='large')
        for idx, name in enumerate(QUICK_FIELDS):
            with (lc if idx % 2 == 0 else rc):
                values[name] = input_control(name, categories)
        with st.expander('🧩 ข้อมูลเพิ่มเติม (อีก 13 ตัวแปร) — กดเพื่อแก้ไข'):
            cc1, cc2 = st.columns(2, gap='large')
            for idx, name in enumerate(ADVANCED_FIELDS):
                with (cc1 if idx % 2 == 0 else cc2):
                    values[name] = input_control(name, categories)
        submitted = st.form_submit_button('🔍 ประเมินด้วยทั้ง 3 โมเดล', type='primary', use_container_width=True)
    if submitted:
        try:
            frame = build_input_dataframe(values)
            probas = {name: float(models[name]['model'].predict_proba(frame)[0, 1]) for name in MODEL_NAMES}
            st.session_state['credit_prediction'] = {'values': values.copy(), 'probabilities': probas}
        except Exception as e:
            st.error(f'ไม่สามารถประเมินได้: {e}')
    if 'credit_prediction' not in st.session_state:
        st.caption('เมื่อกดประเมิน ระบบจะแสดงความน่าจะเป็นจากทั้ง 3 โมเดลและเปรียบเทียบผลในหน้านี้')
    else:
        predictions = st.session_state['credit_prediction']['probabilities']
        st.divider()
        st.subheader('2. ผลประเมินจากทั้ง 3 โมเดล')
        st.caption('คะแนนความเสี่ยงด้านเครดิต (%) คือค่าที่โมเดลประมาณโอกาสอยู่ในกลุ่ม Bad Credit ไม่ใช่ความน่าจะเป็นหนี้สูญ หรืออัตราผิดนัดจริงที่ผ่าน Calibration')
        result_rows = []
        alerts = 0
        for name in MODEL_NAMES:
            proba = predictions[name]
            threshold = models[name]['threshold_cost'] if threshold_mode == 'cost' else 0.5
            flagged = proba >= threshold
            alerts += int(flagged)
            result_rows.append({
                'โมเดล': name,
                'คะแนนความเสี่ยงด้านเครดิต': proba,
                'Threshold': threshold,
                'ผลคัดกรอง': 'ความเสี่ยงถึงเกณฑ์แจ้งเตือน' if flagged else 'ความเสี่ยงต่ำกว่าเกณฑ์',
            })
        result = pd.DataFrame(result_rows)
        if alerts == 3:
            st.error('ทั้ง 3 โมเดลแนะนำให้ **ตรวจสอบเพิ่มเติม** (เป็นผลการคัดกรองเชิงสถิติ)')
        elif alerts == 0:
            st.success('ทั้ง 3 โมเดลประเมินว่า **ความเสี่ยงต่ำกว่าเกณฑ์** ตาม Threshold ที่เลือก')
        else:
            st.warning(f'ผลจากโมเดลไม่ตรงกัน: {alerts} จาก 3 โมเดลแจ้งให้ตรวจสอบเพิ่มเติม')
        pcols = st.columns(3)
        for col, row in zip(pcols, result_rows):
            col.metric(row['โมเดล'],f'{row["คะแนนความเสี่ยงด้านเครดิต"]:.1%}')
            col.caption(f'เกณฑ์ {row["Threshold"]:.0%} · {row["ผลคัดกรอง"]}')
        st.bar_chart(result.set_index('โมเดล')[['คะแนนความเสี่ยงด้านเครดิต']], y_label='คะแนนความเสี่ยงด้านเครดิต (0–1)')
        st.dataframe(
            result.style.format({'คะแนนความเสี่ยงด้านเครดิต': '{:.1%}', 'Threshold': '{:.0%}'}),
            use_container_width=True, hide_index=True,
        )
        with st.expander('ดูข้อมูล 20 ตัวแปรที่ใช้ประเมิน'):
            st.dataframe(
                build_input_dataframe(st.session_state['credit_prediction']['values'])
                .rename(columns=LABELS).T.rename(columns={0: 'ค่าที่กรอก'}),
                use_container_width=True,
            )
        st.caption('ผลการคัดกรองไม่ใช่คำสั่งอนุมัติหรือปฏิเสธสินเชื่อ; แม้มีรายงาน Fairness เบื้องต้นแล้วก็ยังต้องตรวจสอบความเป็นธรรม การปรับเทียบความน่าจะเป็น และบริบทธุรกิจจริง')

with compare_tab:
    st.subheader('เปรียบเทียบประสิทธิภาพโมเดลบนข้อมูลชุดเดียวกัน')
    st.write('โมเดลทั้ง 3 ตัวใช้ Train/Test split 80/20 เดียวกัน และค้นหา Hyperparameter ด้วย Stratified 5-Fold CV บนชุด Train')
    metric_key = 'metrics_cost' if threshold_mode == 'cost' else 'metrics_05'
    comparison = comparison_frame(artifact, threshold_mode)
    metric = st.selectbox(
        'เลือกตัวชี้วัดสำหรับกราฟ', list(METRIC_LABELS),
        index=1, format_func=lambda x: METRIC_LABELS[x],
    )
    chart_data = pd.DataFrame({
        'Model': list(MODEL_NAMES),
        'Score': [metric_value(models[name], metric, metric_key) for name in MODEL_NAMES],
    }).set_index('Model')
    st.bar_chart(chart_data, y_label=METRIC_LABELS[metric])
    st.caption('ROC-AUC ค่ายิ่งสูงยิ่งดี; สำหรับ FN/FP และ Cost ค่ายิ่งต่ำยิ่งดี การเลือก Threshold มีผลต่อ Precision/Recall/F1/Accuracy/Cost แต่ไม่กระทบ ROC-AUC')
    st.markdown('#### ตารางคะแนนเปรียบเทียบ')
    st.dataframe(
        comparison.style.format({
            'CV ROC-AUC': '{:.3f}', 'Test ROC-AUC': '{:.3f}',
            'Accuracy': '{:.1%}', 'Precision (Bad)': '{:.1%}',
            'Recall (Bad)': '{:.1%}', 'F1-score (Bad)': '{:.1%}',
            'Threshold': '{:.0%}',
        }),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        '⬇️ ดาวน์โหลดตารางเปรียบเทียบ (CSV)',
        data=comparison.to_csv(index=False).encode('utf-8-sig'),
        file_name='credit_model_comparison.csv',
        mime='text/csv',
        help='ดาวน์โหลดเฉพาะตัวชี้วัดผลทดสอบ ไม่รวมข้อมูลในฟอร์ม',
    )
    st.markdown('#### ROC Curves บน Test Set')
    draw_roc_chart(artifact)
    st.caption('ROC Curve คำนวณจากชุด Test ที่กันไว้ ส่วนค่าพารามิเตอร์และ Threshold คัดเลือกด้วยชุด Train เท่านั้น')

    st.markdown('#### ดูผลเชิงลึกของแต่ละโมเดล')
    selected_name = st.selectbox('เลือกโมเดลสำหรับดูรายละเอียด', MODEL_NAMES, index=1)
    selected = models[selected_name]
    score = selected[metric_key]
    a, b, c = st.columns(3)
    a.metric('Recall (Bad)', f'{score["recall"]:.1%}')
    b.metric('Precision (Bad)', f'{score["precision"]:.1%}')
    c.metric('Cost = 5×FN + FP', str(score['cost']))
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown('**Confusion Matrix**')
        matrix = pd.DataFrame(
            [[score['tn'], score['fp']], [score['fn'], score['tp']]],
            index=['จริง: Good', 'จริง: Bad'],
            columns=['ทำนาย: Good', 'ทำนาย: Bad'],
        )
        st.dataframe(matrix, use_container_width=True)
        st.caption(f'FN = {score["fn"]} (Bad แต่ทำนาย Good) | FP = {score["fp"]} (Good แต่ทำนาย Bad)')
    with cc2:
        st.markdown('**Hyperparameters ที่ดีที่สุดบนชุด Train**')
        st.json(selected['best_params'])
        st.caption(f'Cost-based threshold: {selected["threshold_cost"]:.0%} · OOF Train Cost: {selected["oof_cost"]}')
    with st.expander('ดูกราฟต้นทุนตาม Threshold (จาก Out-of-Fold Train)'):
        curve = pd.DataFrame(selected['cost_curve']).set_index('threshold')
        st.line_chart(curve, x_label='Threshold', y_label='Cost (5×FN+FP)')


with fairness_tab:
    st.subheader('3. การอธิบายโมเดลและการตรวจสอบความเป็นธรรม')
    st.write(
        'ผลทั้งหมดคำนวณจาก **Hold-out Test Set** หลังจากฝึกและเลือก Threshold '
        'ด้วยชุด Train แล้ว — ไม่มีการนำข้อมูลจากฟอร์มทดลองมาคำนวณ Fairness'
    )
    audit_name = st.selectbox(
        'เลือกโมเดลสำหรับการวิเคราะห์', MODEL_NAMES, index=1, key='audit_model',
    )
    audit_model = models[audit_name]
    st.caption(
        f'โมเดล: {audit_name} | ชุด Test: {artifact["test_samples"]:,} รายการ | '
        f'Threshold: {audit_model["threshold_cost"]:.0%} (Cost-based)' if threshold_mode == 'cost'
        else f'โมเดล: {audit_name} | ชุด Test: {artifact["test_samples"]:,} รายการ | Threshold: 50%'
    )
    st.markdown('#### 3.1 Permutation Importance — บน Test Set')
    st.caption(
        'ค่าความสำคัญแสดงการลดลงของ Test ROC-AUC เมื่อสลับค่าตัวแปรหนึ่งตัว '
        '(10 ครั้ง, random_state=42) ค่ายิ่งมากหมายถึงตัวแปรนั้นมีผลต่อความสามารถในการจัดอันดับ; '
        'ค่าติดลบหมายถึงการสลับค่าอาจทำให้ ROC-AUC เพิ่มขึ้น ไม่ใช่ความเสี่ยงติดลบ'
    )
    importance = pd.DataFrame(audit_model['permutation_importance'])
    top_count = st.slider('จำนวนตัวแปรที่แสดง', min_value=5, max_value=20, value=10, key='top_importance')
    top_importance = importance.head(top_count).copy()
    top_importance['ชื่อไทย'] = top_importance['feature'].map(LABELS)
    st.bar_chart(
        top_importance.set_index('ชื่อไทย')[['importance_mean']],
        y_label='Test ROC-AUC reduction',
    )
    st.dataframe(
        importance.rename(columns={
            'feature': 'ตัวแปร', 'importance_mean': 'Mean ROC-AUC reduction',
            'importance_std': 'Std. dev.',
        }).style.format({'Mean ROC-AUC reduction': '{:.4f}', 'Std. dev.': '{:.4f}'}),
        hide_index=True, use_container_width=True,
    )
    st.download_button(
        '⬇️ ดาวน์โหลด Permutation Importance (CSV)',
        data=importance.to_csv(index=False).encode('utf-8-sig'),
        file_name=f'permutation_{audit_name.lower().replace(" ", "_")}.csv',
        mime='text/csv',
    )

    with st.expander('SHAP (ตัวเลือกเพิ่มเติมของ Random Forest)'):
        shap_record = audit_model.get('shap', {})
        if shap_record.get('status') == 'computed':
            st.caption(
                f'SHAP TreeExplainer ของ Bad Class บนตัวอย่าง {shap_record["n_test_sampled"]} รายการ'
                'ที่สุ่มจาก Test Set (ไม่ใช่ Test Set ทั้งหมด)'
            )
            shap_data = pd.DataFrame(shap_record['feature_importance'])
            st.bar_chart(shap_data.head(15).set_index('feature')[['mean_abs_shap']])
            st.dataframe(shap_data, hide_index=True, use_container_width=True)
        elif shap_record.get('status') == 'unavailable':
            st.warning(f'SHAP ไม่สามารถคำนวณได้: {shap_record.get("reason", "ไม่ทราบสาเหตุ")}')
        else:
            st.info(
                'Permutation Importance ด้านบนครบตามข้อ 3 แล้ว '
                'หากต้องการ SHAP ของ Random Forest เพิ่ม ให้ติดตั้ง requirements-shap.txt '
                'และสั่ง `python train_models.py --with-shap` แล้วโหลดไฟล์โมเดลใหม่'
            )

    st.divider()
    st.markdown('#### 4. Fairness — แยกตาม Sex และ Age Group')
    st.info(
        'Sex ใน German Credit อนุมานจาก personal_status (ข้อมูลเชิงตัวแทน ไม่ใช่การยืนยันเพศโดยตรง); '
        'Age Group แบ่ง Young (≤25) และ Old (>25) ตาม Notebook เดิม '
        'โดย **age และ personal_status เป็นตัวแปรที่โมเดลใช้ทำนายด้วย**'
    )
    st.caption(
        '**Selection Rate** = สัดส่วนที่ถูกทำนายว่า Bad (แจ้งเตือน) ' 
        'ไม่ใช่สัดส่วนที่ได้รับอนุมัติ | **TPR** = สัดส่วนของ Bad จริงที่ตรวจพบ | '
        '**FPR** = สัดส่วนของ Good จริงที่ถูกแจ้งเตือนผิด'
    )
    current_threshold = 'cost' if threshold_mode == 'cost' else '0.5'
    audit_fairness = audit_model['fairness'][current_threshold]
    for dimension, dimension_label in (
        ('sex', 'เพศที่อนุมานจาก personal_status (Sex proxy)'),
        ('age_group', 'กลุ่มอายุ (Age Group)'),
    ):
        st.markdown(f'**{dimension_label}**')
        report = audit_fairness[dimension]
        fairness_table = pd.DataFrame(report['groups'])
        gaps = report['gaps']
        showcols = ['group', 'n', 'actual_bad', 'actual_good',
                    'selection_rate', 'tpr', 'fpr', 'tp', 'fp', 'tn', 'fn']
        st.dataframe(
            fairness_table[showcols].rename(columns={
                'group': 'กลุ่ม', 'n': 'จำนวน', 'actual_bad': 'Bad จริง',
                'actual_good': 'Good จริง', 'selection_rate': 'Selection Rate (ทำนาย Bad)',
                'tpr': 'TPR (Bad)', 'fpr': 'FPR (Good)',
                'tp': 'TP', 'fp': 'FP', 'tn': 'TN', 'fn': 'FN',
            }).style.format({
                'Selection Rate (ทำนาย Bad)': '{:.1%}',
                'TPR (Bad)': '{:.1%}', 'FPR (Good)': '{:.1%}',
            }, na_rep='—'),
            hide_index=True, use_container_width=True,
        )
        gap_a, gap_b, gap_c = st.columns(3)
        fmt_gap = lambda value: f'{value:.1%}' if np.isfinite(value) else 'N/A'
        gap_a.metric('Selection Rate Gap', fmt_gap(gaps['selection_rate_gap']))
        gap_b.metric('TPR Gap', fmt_gap(gaps['tpr_gap']))
        gap_c.metric('FPR Gap', fmt_gap(gaps['fpr_gap']))
        if (fairness_table['n'] < 20).any():
            st.warning('มีบางกลุ่มที่มีตัวอย่างน้อยกว่า 20 รายการ ควรระวังความไม่แน่นอนของค่า Fairness')
        if fairness_table[['tpr', 'fpr']].isna().any().any():
            st.warning('มีค่า TPR/FPR ที่ไม่สามารถคำนวณได้ (กลุ่มนั้นไม่มี Bad หรือไม่มี Good จริง) จึงแสดงเป็น — แทน 0%')
        st.download_button(
            f'⬇️ ดาวน์โหลด Fairness ตาม {dimension} (CSV)',
            data=fairness_table.to_csv(index=False).encode('utf-8-sig'),
            file_name=f'fairness_{audit_name.lower().replace(" ", "_")}_{dimension}_{current_threshold.replace(".", "")}.csv',
            mime='text/csv', key=f'fairness_download_{dimension}',
        )
    st.caption(
        'Gaps เป็นผลต่างระหว่างค่าสูงสุด–ต่ำสุดของกลุ่มที่คำนวณได้ '
        'จึงเป็นเพียงตัวชี้วัดความแตกต่าง ไม่ใช่หลักฐานยืนยันว่าการตัดสินมีความเป็นธรรมหรือเลือกปฏิบัติ '
        'และการใช้ Test Set ในการรายงานนี้ไม่ใช่การปรับ Threshold ตามกลุ่ม'
    )

with guide_tab:
    st.subheader('การใช้งานและการอ่านผล')
    st.markdown('''
**ความครอบคลุมของข้อกำหนดงานวิจัย**

- **2.1** Target: bad=1, good=0; `train_test_split(test_size=0.2, stratify=y, random_state=42)`
- **2.2** 3 sklearn Pipelines: `ColumnTransformer(StandardScaler + OneHotEncoder)` และ Logistic Regression, Random Forest, Gradient Boosting
- **2.3** GridSearchCV / StratifiedKFold(n_splits=5, shuffle=True, random_state=42) เฉพาะ Training Set
- **2.4** OOF `cross_val_predict` บน Train; เลือก Threshold ด้วย Cost = `5 × FN + FP` แยกแต่ละโมเดล
- **3** Permutation Importance (10 repeats; ROC-AUC) บนชุด Test แยกโมเดล; SHAP เป็นตัวเลือกเสริม
- **4** Fairness: แยก Sex proxy และ Age Group บน Test; Selection Rate / TPR / FPR และ Gap พร้อมจำนวนตัวอย่าง

**ขั้นตอนใช้งาน**

1. เปิดแท็บ **ประเมินความเสี่ยง** และกรอกข้อมูลสำคัญ พร้อมตรวจสอบรายละเอียดเพิ่มเติมทั้ง 20 ตัวแปร
2. กด **ประเมินด้วยทั้ง 3 โมเดล** เพื่อดูคะแนนและการแจ้งเตือนของแต่ละโมเดล
3. เปิด **เปรียบเทียบโมเดล** เพื่อดูผลการทดสอบจริงและ ROC Curve
4. เปิด **การอธิบายและความเป็นธรรม** เพื่อดู Permutation Importance และกลุ่ม Sex / Age
5. เลือกเกณฑ์ด้านซ้ายเพื่อเปรียบเทียบผลที่ Threshold 50% กับเกณฑ์ Cost-based ของแต่ละโมเดล

**ความหมายของ Metrics**

- **ROC-AUC**: ความสามารถในการจัดอันดับความเสี่ยง แยกจาก Threshold
- **Precision (Bad)**: ในรายการที่ถูกคัดกรองว่า Bad มี Bad จริงเท่าใด
- **Recall (Bad)**: ในรายการที่เป็น Bad จริง โมเดลตรวจพบได้เท่าใด
- **F1-score**: ค่าเฉลี่ยฮาร์มอนิกของ Precision และ Recall
- **FN**: ผู้มีความเสี่ยง Bad แต่ถูกคัดกรองเป็น Good (กำหนดต้นทุน 5)
- **FP**: ผู้มีสถานะ Good แต่ถูกคัดกรองเป็น Bad (กำหนดต้นทุน 1)

**การตีความ Threshold**

- **50%** ใช้ค่าตัดสินเดียวกันทุกโมเดล เพื่อดูผลในเงื่อนไขเดียวกัน
- **Cost-based** ปรับ Threshold สำหรับแต่ละโมเดลด้วย Out-of-Fold predictions จากชุด Train (FN:FP = 5:1) ตามแนวคิดใน Notebook
- Cost เป็น *หน่วยสมมติ* ตามน้ำหนักของข้อผิดพลาด ไม่ใช่ต้นทุนทางการเงินจริง

**ข้อจำกัดทางวิจัยและความเป็นธรรม**

- ข้อมูล German Credit มี 1,000 ตัวอย่าง เป็นข้อมูลต่างประเทศในอดีตและใช้หน่วยเงิน DM
- โมเดลใช้ข้อมูล **age และ personal_status โดยตรง** ตลอดจน foreign_worker; การตรวจสอบ Fairness ที่แสดงยังไม่รับรองว่าโมเดลเป็นธรรม
- ผลความน่าจะเป็นอาจไม่ Calibrated และไม่ได้ยืนยันความแม่นยำกับประชากรไทย
- ห้ามใช้คะแนนนี้เป็นคำตัดสินสินเชื่ออัตโนมัติ
- โปรดใช้ข้อมูลจำลองบน Streamlit Community Cloud เพราะข้อมูลฟอร์มจะถูกประมวลผลบนเซิร์ฟเวอร์ของผู้ให้บริการ
- โค้ดไม่ได้เขียนบันทึกข้อมูลฟอร์มหรือส่งข้อมูลผู้สมัครออกไปยัง API แต่แพลตฟอร์ม Cloud เป็นระบบภายนอก
''')
    st.caption(f'แหล่งข้อมูล: {artifact["source"]} · โมเดลฝึกเมื่อ (UTC): {artifact["created_utc"]} · scikit-learn: {artifact["sklearn_version"]}')
