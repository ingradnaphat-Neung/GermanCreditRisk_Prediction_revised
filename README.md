# Credit Risk Lab — Streamlit เปรียบเทียบ 3 Machine Learning Models

Dashboard ภาษาไทย สร้างจากไฟล์ `Final.ipynb` (German Credit / OpenML credit-g, dataset ID `31`)

**สำหรับงานสาธิตและวิจัยเท่านั้น ไม่ใช่ระบบอนุมัติสินเชื่อ**

## สิ่งที่ได้จากเวอร์ชันนี้

- หน้าใช้งานง่ายขึ้น: ช่องข้อมูลสำคัญ 7 ช่อง และส่วนขยายข้อมูลอีก 13 ช่อง ครบ 20 ตัวแปร
- คำอธิบายตัวเลือกเป็นภาษาไทย ตัวแปรและ category ที่ส่งให้ Pipeline ยังเป็นรหัสต้นฉบับ
- ประเมินข้อมูลตัวอย่างเดียวกันผ่าน **Logistic Regression, Random Forest, Gradient Boosting** พร้อมกัน
- ตารางเปรียบเทียบ Cross-Validation ROC-AUC, Test ROC-AUC, Accuracy, Precision, Recall, F1-score และ Cost
- กราฟเปรียบเทียบคะแนน, ROC Curves, Confusion Matrix ของแต่ละโมเดล
- เปลี่ยนเกณฑ์ตัดสินระหว่าง **Threshold 0.50** และ **Cost-based Threshold แยกรายโมเดล**
- ดาวน์โหลดผลเปรียบเทียบเป็น CSV (เฉพาะ metric รวม ไม่มีข้อมูลผู้กรอก)
- ไม่มีฐานข้อมูลผู้สมัครหรือฟังก์ชันส่งข้อมูลผู้กรอกไปยัง API อื่น

## ผลการทดลองเดิมใน Final.ipynb (อ้างอิงเท่านั้น)

| Model | Best 5-Fold CV ROC-AUC จาก Notebook |
|:--|--:|
| Logistic Regression | 0.7838 |
| Random Forest | 0.7950 |
| Gradient Boosting | 0.7847 |

ใน Notebook เดิม Random Forest ถูกเลือกเป็นโมเดลหลัก และมีเกณฑ์ Cost-based threshold `0.21` (กำหนด `FN:FP = 5:1`) **เมื่อฝึกใหม่ ค่าผลประเมินและ Threshold อาจแตกต่างจาก Notebook** ตามเวอร์ชันของไลบรารีและข้อมูลที่ได้รับ แอปจะแสดง *ค่าจากการฝึกจริง* ไม่กรอกเลขจากตารางนี้แทน

## โครงสร้างไฟล์

```text
GermanCredit_Comparison_Streamlit/
├── app.py                  # Dashboard ภาษาไทย 4 แท็บ
├── train_models.py         # ฝึก + เปรียบเทียบ 3 โมเดล แล้วบันทึก
├── train_model.py          # ชื่อเดิมสำหรับ compatibility
├── model_utils.py          # Feature schema และ threshold helper
├── requirements.txt
├── requirements-shap.txt    # SHAP เป็นทางเลือกเสริม
├── .gitignore
├── .streamlit/
│   └── config.toml         # สีและธีม
└── tests/
    └── test_models.py      # Offline unit + integration tests (synthetic data)
```

ไฟล์ `credit_models.joblib` จะถูกสร้างเมื่อฝึกโมเดลสำเร็จ จึงไม่ได้แนบมาใน ZIP ต้นฉบับ (ระบบนี้ไม่เชื่อมต่อ OpenML จากสภาพแวดล้อมที่ใช้จัดทำไฟล์) ถ้าต้องการให้เปิดใน Streamlit Cloud ได้เร็วขึ้น ให้ฝึกบนเครื่องที่เข้าถึง OpenML ได้ แล้วเพิ่ม `credit_models.joblib` ลงใน GitHub Repository เองได้ **เพราะโมเดลนี้ฝึกด้วยชุดข้อมูลสาธารณะ** แต่ตรวจสอบขนาดไฟล์ก่อน Commit

## วิธีรันในเครื่อง

แนะนำ **Python 3.12** และใช้ virtual environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

`python train_models.py` ใช้ข้อมูล **OpenML credit-g (ID 31)** ซึ่งเป็นข้อมูลสาธารณะ หากต้องการฝึกด้วยข้อมูลที่มีอยู่แล้วในเครื่อง (รูปแบบ CSV มี 20 คอลัมน์ตาม Dataset และ `class` ที่เป็น `good`/`bad`) ใช้:

```bash
python train_models.py --data path/to/german_credit.csv
```

**ข้อควรระวัง:** การใช้ชุดข้อมูลอื่นควรทำกระบวนการตรวจสอบ schema/คุณภาพ, fairness และ leakage ก่อน ไม่ควรนำข้อมูลส่วนบุคคลไปทดลองบน GitHub สาธารณะหรือ Streamlit Cloud

ถ้ายังไม่ได้สร้างไฟล์ `credit_models.joblib` แอปจะพยายามโหลด OpenML และฝึกโมเดล 3 ตัวโดยอัตโนมัติในครั้งแรก ซึ่งอาจใช้ทรัพยากรของ Streamlit Cloud มากกว่าการแนบไฟล์โมเดลที่เตรียมไว้แล้ว

## วิธีส่งขึ้น GitHub และ Deploy บน Streamlit Community Cloud

1. สร้าง Repository **Private** (แนะนำก่อนสำหรับโค้ดที่ยังอยู่ระหว่างการพัฒนา)
2. อัปโหลดไฟล์จากโฟลเดอร์นี้ขึ้น GitHub ให้ `app.py` และ `requirements.txt` อยู่ที่ root
3. หากฝึกโมเดลด้วยข้อมูล OpenML **สาธารณะเท่านั้น** และต้องการให้เปิดเร็วขึ้น ให้ตรวจสอบไฟล์ก่อนแล้วเพิ่ม `credit_models.joblib` ที่ root โดยใช้ `git add -f credit_models.joblib` (เริ่มต้น `.gitignore` กันไฟล์โมเดลเพื่อความปลอดภัย)
4. ไปที่ <https://share.streamlit.io/> เลือก **Create app** → **From existing repo**
5. กำหนด Repository, Branch `main`, Main file path `app.py` และ Python 3.12 (หากมีตัวเลือก)
6. Deploy และทดสอบด้วยข้อมูล **สมมติ** เท่านั้น

หรืออัปโหลดผ่านคำสั่ง Git บนเครื่อง:

```bash
git init
git add .
git commit -m "Thai credit-risk model comparison dashboard"
git branch -M main
git remote add origin https://github.com/YOUR-USER/YOUR-REPOSITORY.git
git push -u origin main
```

ให้สร้าง Repo ที่ GitHub ก่อน และแทน `YOUR-USER` / `YOUR-REPOSITORY` ด้วยชื่อที่มีจริง

## วิธีคำนวณที่ใช้เปรียบเทียบ

1. ใช้ **same stratified train/test split 80/20**, `random_state=42`
2. ฝึกแบบ pipeline ภายใน CV: `StandardScaler` สำหรับ 7 numeric, `OneHotEncoder(handle_unknown='ignore')` สำหรับ 13 categorical
3. Tune Hyperparameters แบบ `GridSearchCV(scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=42))` โดยใช้ *grid ตาม Notebook เดิม*
4. คำนวณ OOF probability จาก Training Set ของแต่ละ **best estimator** เพื่อเลือก Threshold ที่ลด `5×FN + 1×FP`
5. ประเมิน **Holdout Test Set ชุดเดียวกัน** ทุกโมเดล ที่ทั้ง Threshold `0.50` และ Threshold cost-based แยกรายโมเดล
6. เก็บ `ROC Curve` และ `Confusion Matrix` จากชุด Test; รายงาน CV ROC-AUC จาก Training folds ไม่ใช้ Test ในการ tune

**หมายเหตุเชิงวิธีวิจัย:** การเลือก Threshold ด้วย OOF ของ `best estimator` ที่ tune จาก training CV แล้ว เป็นวิธีที่สอดคล้องกับ Notebook แต่ CV ROC-AUC ของ GridSearchCV ยังไม่ใช่ nested-CV แบบปราศจากความเอนเอียงในการเลือก hyperparameters ทั้งหมด ไม่ควรอ้างผลคะแนนนี้เป็น generalization guarantee. นอกจากนี้ threshold ของแต่ละโมเดลต่างกัน จึงควรใช้ Standard threshold 0.50 เมื่อต้องการเปรียบเทียบผลความคลาดเคลื่อนบนเกณฑ์เดียวกัน

## Privacy, fairness และ limitations

- **โค้ดไม่มีคำสั่งอัปโหลดข้อมูลฟอร์มลูกค้าออกไปยังแหล่งอื่น**; ผลที่กรอกอยู่ใน memory/session ของ Streamlit เท่านั้น
- หาก Deploy บน **Streamlit Community Cloud** ข้อมูลผู้ใช้ที่กรอกจะถูกประมวลผลบน Infrastructure ของผู้ให้บริการ โดยธรรมชาติของ Web App จึงไม่ควรใส่ข้อมูลลูกค้าจริงหรือข้อมูลลับ
- โมเดลฝึกจากข้อมูลเครดิตเยอรมันในอดีต ใช้หน่วยเงิน **Deutsche Mark (DM)** ไม่ใช่เงินบาท
- ข้อมูลเดิมมีตัวแปรอ่อนไหวอย่าง `age`, `personal_status`, `foreign_worker`; มีความเสี่ยงด้าน bias/discrimination
- Probability ไม่ได้รับการยืนยันว่าปรับเทียบตรงกับความถี่ผิดนัดจริง (calibration)
- **ผลประเมินไม่ใช่การอนุมัติสินเชื่อ** หากต้องใช้จริงต้องมี validation, monitoring, fairness testing, legal review และ human oversight
- ไฟล์ `.joblib` เป็น Python serialization; โหลดเฉพาะไฟล์ที่สร้างเองหรือมาจากแหล่งเชื่อถือได้ เพราะการโหลดไฟล์ที่ไม่ปลอดภัยอาจรันโค้ดที่ฝังอยู่

## ทดสอบโค้ด

```bash
python -m pip install pytest
python -m pytest -q
python -m compileall -q app.py train_models.py model_utils.py
```

การทดสอบ Offline ในโฟลเดอร์ `tests/` ใช้ข้อมูลสังเคราะห์เพื่อทดสอบ pipeline ทั้ง 3 ตัว และไม่ร้องขอข้อมูลผ่านอินเทอร์เน็ต ส่วนการตรวจสอบตัวเลขประเมินตรงตาม Notebook เดิม ต้องฝึกด้วย OpenML credit-g จริงบนเครื่องที่สามารถเชื่อมต่อได้

---

## เวอร์ชันเพิ่มเติม — Explainability & Fairness (ครบข้อ 2.1–4)

เพิ่มแท็บ **🔎 การอธิบายและความเป็นธรรม** โดยคงหน้าประเมินและหน้าตารางเปรียบเทียบเดิมไว้ และเปลี่ยนคำอธิบายจาก “ความน่าจะเป็น Bad” เป็น **“คะแนนความเสี่ยงด้านเครดิต”** และจาก “ไม่แจ้งเตือน” เป็น **“ความเสี่ยงต่ำกว่าเกณฑ์”** การเปลี่ยนชื่อไม่เปลี่ยนผลทำนาย

| ข้อกำหนด | ไฟล์/สิ่งที่ทำ |
|---|---|
| **2.1** bad=positive class, stratified, seed 42 | `train_models.py`: 0=`good`, 1=`bad`, stratified 80/20 |
| **2.2** sklearn Pipeline + ColumnTransformer, ≥3 algorithms | `make_pipeline` Logistic Regression / Random Forest / Gradient Boosting |
| **2.3** Hyperparameter tuning ด้วย 5-fold Stratified CV เฉพาะ Train | `GridSearchCV`, `StratifiedKFold(5, shuffle=True, random_state=42)` |
| **2.4** Cost-optimized threshold จาก OOF Train (FN:FP 5:1) | `cross_val_predict` บน Train แล้ว minimize `5*FN+FP` สำหรับแต่ละโมเดล |
| **3** Permutation Importance หรือ SHAP บน Test | `permutation_importance` สำหรับทั้ง 3 โมเดล (10 repeats, scoring=`roc_auc`, seed=42); **SHAP เป็นส่วนเสริม** สำหรับ RF |
| **4** Fairness แยก Sex, Age Group | `group_fairness_metrics` บน Test สำหรับแต่ละโมเดลและ Threshold; แสดงขนาดกลุ่ม TP/FP/TN/FN, Selection Rate, TPR, FPR และ gaps |

### รายละเอียดการตีความ Fairness

- `sex` เป็น **proxy** ของเพศตามหมวดหมู่ใน `personal_status` ของข้อมูล German Credit ไม่ใช่ข้อมูลเพศที่ยืนยันแยกต่างหาก และไม่ใช่ตัวแทนอัตลักษณ์ทางเพศทั้งหมด
- `age_group` จาก `age <= 25` (Young) และ `> 25` (Old) ตามเกณฑ์ใน Notebook ต้นฉบับ
- **Selection Rate = P(ทำนาย Bad | กลุ่ม)** ไม่ใช่ approval rate
- **TPR = TP/(TP+FN)**; **FPR = FP/(FP+TN)** หากส่วนหารเป็นศูนย์จะระบุว่า **N/A** ไม่ใช่ 0
- **Gaps** คือผลต่างค่าสูงสุด–ต่ำสุดในแต่ละกลุ่มที่คำนวณได้ ถ้ามีไม่ถึงสองกลุ่มที่คำนวณ metric นั้นได้จะแสดง N/A; ไม่ใช้การทดสอบทางสถิติยืนยันความเป็นธรรม
- **Important:** `personal_status`, `age` และ `foreign_worker` **ยังคงเป็นฟีเจอร์ที่โมเดลใช้เรียนรู้** ไม่ได้ทำ fairness-aware training หรือเอาฟีเจอร์อ่อนไหวออก เพราะต้องคงแนวทางเดิมของงานวิจัย การวิเคราะห์ Fairness เป็นการตรวจสอบหลังฝึกบนชุด Test ไม่ใช่การแก้ Bias
- ผลประเมิน Fairness ที่เลือก Threshold 50% กับ Cost-based จะต่างกัน และค่าจะแสดงเฉพาะผลของ Test Set ที่โมเดลถูกประเมินอยู่ ไม่เกี่ยวกับข้อมูลที่พิมพ์ลงในฟอร์ม

### SHAP เพิ่มเติม (ไม่จำเป็นต่อโจทย์ข้อ 3 เพราะมี Permutation Importance แล้ว)

หากต้องการค่า SHAP สำหรับ Random Forest ให้ติดตั้งแพ็กเกจเสริม และฝึกโมเดลใหม่:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-shap.txt
python train_models.py --with-shap
streamlit run app.py
```

SHAP ใช้ `TreeExplainer` ของโมเดล RF บนตัวอย่างไม่เกิน 80 รายการที่สุ่มจาก **Test Set**, บันทึกค่า mean absolute SHAP และรวมกลุ่มคอลัมน์ One-Hot กลับเป็น 20 ตัวแปร ไม่ใช่ Beeswarm plot หรือ SHAP ของทุก Test Row หาก SHAP ไม่พร้อมใช้งาน แอปจะแสดงสาเหตุและยังเปิดรายงาน Permutation Importance ได้ตามปกติ

### เมื่อต้องอัปเกรดจากเวอร์ชันเดิม

โมเดลใน `credit_models.joblib` เดิมมี artifact version 2 ไม่มีผล Importance/Fairness; เวอร์ชันใหม่นี้ต้องมี **artifact version 3** จึงต้องฝึกใหม่ด้วย `python train_models.py` และแทนไฟล์โมเดลเดิม ไม่ใช่เปลี่ยนแค่ `app.py`. ใช้ Python 3.12 และ GitHub repo ที่ควบคุมสิทธิ์เอง โดย **ห้าม commit ข้อมูลลูกค้าจริง / CSV ภายใน / credentials**. แอปที่ Deploy ผ่าน Streamlit Community Cloud ส่งข้อมูลที่กรอกไปประมวลผลบนเซิร์ฟเวอร์ของผู้ให้บริการ จึงไม่ควรนำข้อมูลส่วนบุคคล/ลับลงในฟอร์มเมื่อใช้งาน public cloud.

### ตรวจสอบโค้ดแบบ offline

```bash
python -m pytest -q
python -m compileall -q app.py model_utils.py train_models.py
```

Tests ใช้ข้อมูลจำลอง (synthetic) ในเครื่อง ไม่มีการดึงไฟล์ข้อมูลผู้ใช้หรือส่งข้อมูลให้บริการภายนอก ค่าประสิทธิภาพบน German Credit จริงยังต้องฝึกโดยใช้ OpenML `data_id=31` หรือ CSV สาธารณะที่ตรง schema แล้วตรวจสอบผลก่อนใช้ประกอบรายงานวิชาการ
