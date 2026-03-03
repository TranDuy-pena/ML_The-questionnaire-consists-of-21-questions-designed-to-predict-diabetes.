from flask import Flask, render_template, request
from joblib import load
from pathlib import Path
import pandas as pd

app = Flask(__name__)

# ===== Paths =====
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Code" / "models" / "xgb_binary_FOCUSED_TUNED.joblib"  # chỉnh đúng nếu khác

bundle = load(MODEL_PATH)
preprocess = bundle["preprocess"]
booster = bundle["booster"]

THRESHOLD = float(bundle.get("best_threshold", bundle.get("threshold", 0.5)))
THRESHOLD_SHOW = round(THRESHOLD, 2)

FEATURES = [
    "HighBP","HighChol","CholCheck","BMI","Smoker","Stroke","HeartDiseaseorAttack",
    "PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost",
    "GenHlth","MentHlth","PhysHlth","DiffWalk","Sex","Age","Education","Income"
]

LABELS = {
    "HighBP": "Cao huyết áp",
    "HighChol": "Cholesterol cao",
    "CholCheck": "Đã xét nghiệm cholesterol trong 5 năm",
    "BMI": "Chỉ số BMI",
    "Smoker": "Hút thuốc nhiều",
    "Stroke": "Từng bị đột quỵ",
    "HeartDiseaseorAttack": "Bệnh tim / nhồi máu cơ tim",
    "PhysActivity": "Có hoạt động thể chất 30 ngày qua",
    "Fruits": "Ăn trái cây ≥ 1 lần/ngày",
    "Veggies": "Ăn rau ≥ 1 lần/ngày",
    "HvyAlcoholConsump": "Uống rượu/bia nhiều",
    "AnyHealthcare": "Có BHYT/tiếp cận y tế",
    "NoDocbcCost": "Không đi khám vì chi phí",
    "GenHlth": "Sức khỏe chung",
    "MentHlth": "Số ngày tinh thần kém (0–30)",
    "PhysHlth": "Số ngày thể chất kém (0–30)",
    "DiffWalk": "Khó đi lại/leo cầu thang",
    "Sex": "Giới tính",
    "Age": "Nhóm tuổi",
    "Education": "Trình độ học vấn",
    "Income": "Mức thu nhập",
}

AGE_CHOICES = [
    (1,"18–24"),(2,"25–29"),(3,"30–34"),(4,"35–39"),(5,"40–44"),
    (6,"45–49"),(7,"50–54"),(8,"55–59"),(9,"60–64"),(10,"65–69"),
    (11,"70–74"),(12,"75–79"),(13,"80+")
]
EDU_CHOICES = [
    (1,"Chưa học"),(2,"Tiểu học"),(3,"THCS"),(4,"THPT"),(5,"Cao đẳng"),(6,"Đại học+")
]
INCOME_CHOICES = [
    (1,"< 10k$"),(2,"10k–15k$"),(3,"15k–20k$"),(4,"20k–25k$"),
    (5,"25k–35k$"),(6,"35k–50k$"),(7,"50k–75k$"),(8,"> 75k$")
]

AGE_MAP = dict(AGE_CHOICES)
EDU_MAP = dict(EDU_CHOICES)
INCOME_MAP = dict(INCOME_CHOICES)

def to_float(v, default=0.0):
    try:
        return float(v)
    except:
        return float(default)

def to_int(v, default=0):
    try:
        return int(float(v))
    except:
        return int(default)

def yn(v):
    return "Có" if int(v) == 1 else "Không"

@app.get("/")
def index():
    return render_template(
        "index.html",
        labels=LABELS,
        threshold_show=THRESHOLD_SHOW,
        age_choices=AGE_CHOICES,
        edu_choices=EDU_CHOICES,
        income_choices=INCOME_CHOICES,
    )

@app.post("/predict")
def predict():
    data = {}

    # --- parse numeric safely ---
    # BMI: không bắt buộc nhập nếu có chiều cao/cân nặng
    bmi = request.form.get("BMI", "").strip()
    height_cm = request.form.get("height_cm", "").strip()
    weight_kg  = request.form.get("weight_kg", "").strip()

    # mặc định bmi từ input
    data["BMI"] = to_float(bmi, 0.0)

    # nếu có chiều cao & cân nặng -> ưu tiên tính BMI
    if height_cm and weight_kg:
        h = to_float(height_cm, 0.0) / 100.0
        w = to_float(weight_kg, 0.0)
        if h > 0:
            data["BMI"] = w / (h*h)

    # binary selects
    for f in ["HighBP","HighChol","CholCheck","Smoker","Stroke","HeartDiseaseorAttack",
              "PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare",
              "NoDocbcCost","DiffWalk"]:
        data[f] = to_int(request.form.get(f, "0"), 0)

    # GenHlth 1..5
    data["GenHlth"] = max(1, min(5, to_int(request.form.get("GenHlth","3"), 3)))

    # MentHlth PhysHlth 0..30
    data["MentHlth"] = max(0, min(30, to_int(request.form.get("MentHlth","0"), 0)))
    data["PhysHlth"] = max(0, min(30, to_int(request.form.get("PhysHlth","0"), 0)))

    # Sex 0/1
    data["Sex"] = to_int(request.form.get("Sex","0"), 0)

    # Age 1..13, Edu 1..6, Income 1..8
    data["Age"] = max(1, min(13, to_int(request.form.get("Age","1"), 1)))
    data["Education"] = max(1, min(6, to_int(request.form.get("Education","1"), 1)))
    data["Income"] = max(1, min(8, to_int(request.form.get("Income","1"), 1)))

    # --- predict ---
    X_user = pd.DataFrame([data], columns=FEATURES)
    proba = float(booster.predict_proba(preprocess.transform(X_user))[:, 1][0])
    pred = 1 if proba >= THRESHOLD else 0
    label = "Có nguy cơ" if pred == 1 else "Không nguy cơ"

    # --- prepare pretty display ---
    pretty = {
        LABELS["BMI"]: round(data["BMI"], 1),
        LABELS["Sex"]: "Nam" if data["Sex"] == 1 else "Nữ",
        LABELS["Age"]: AGE_MAP.get(data["Age"], str(data["Age"])),
        LABELS["Education"]: EDU_MAP.get(data["Education"], str(data["Education"])),
        LABELS["Income"]: INCOME_MAP.get(data["Income"], str(data["Income"])),
        LABELS["GenHlth"]: {1:"Rất tốt",2:"Tốt",3:"Trung bình",4:"Kém",5:"Rất kém"}.get(data["GenHlth"], data["GenHlth"]),
        LABELS["MentHlth"]: f'{data["MentHlth"]} ngày',
        LABELS["PhysHlth"]: f'{data["PhysHlth"]} ngày',
        LABELS["HighBP"]: yn(data["HighBP"]),
        LABELS["HighChol"]: yn(data["HighChol"]),
        LABELS["CholCheck"]: yn(data["CholCheck"]),
        LABELS["Smoker"]: yn(data["Smoker"]),
        LABELS["Stroke"]: yn(data["Stroke"]),
        LABELS["HeartDiseaseorAttack"]: yn(data["HeartDiseaseorAttack"]),
        LABELS["PhysActivity"]: yn(data["PhysActivity"]),
        LABELS["Fruits"]: yn(data["Fruits"]),
        LABELS["Veggies"]: yn(data["Veggies"]),
        LABELS["HvyAlcoholConsump"]: yn(data["HvyAlcoholConsump"]),
        LABELS["AnyHealthcare"]: yn(data["AnyHealthcare"]),
        LABELS["NoDocbcCost"]: yn(data["NoDocbcCost"]),
        LABELS["DiffWalk"]: yn(data["DiffWalk"]),
    }

    return render_template(
        "result.html",
        proba=proba,
        label=label,
        pred=pred,
        threshold_show=THRESHOLD_SHOW,
        pretty=pretty
    )

if __name__ == "__main__":
    app.run(debug=True)