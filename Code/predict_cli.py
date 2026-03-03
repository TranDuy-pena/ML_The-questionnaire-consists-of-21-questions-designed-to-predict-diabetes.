from joblib import load
import pandas as pd

MODEL_PATH = r"e:\MachineLearning\Dự đoán nguy cơ mắc bệnh tiểu đường(đã có data 250k dòng)\Code\models\xgb_binary_FOCUSED_TUNED.joblib"
CSV_PATH   = r"e:\MachineLearning\Dự đoán nguy cơ mắc bệnh tiểu đường(đã có data 250k dòng)\Data\diabetes_012_health_indicators_BRFSS2015.csv"
TARGET_COL = "Diabetes_012"

def load_bundle(model_path: str):
    bundle = load(model_path)
    preprocess = bundle["preprocess"]
    booster = bundle["booster"]
    thr = float(bundle.get("best_threshold", bundle.get("threshold", 0.5)))
    return preprocess, booster, thr

def get_feature_columns(csv_path: str, target_col: str):
    df = pd.read_csv(csv_path, nrows=5)
    if target_col not in df.columns:
        raise ValueError(f"Không thấy cột target '{target_col}' trong CSV.")
    return [c for c in df.columns if c != target_col]

def prompt_user_input(feature_cols):
    print("\nNhập dữ liệu người dùng (ấn Enter để dùng mặc định 0):")
    record = {}
    for c in feature_cols:
        while True:
            val = input(f"  {c} = ").strip()
            if val == "":
                record[c] = 0
                break
            try:
                # đa số feature là số; cho phép nhập float
                record[c] = float(val)
                break
            except ValueError:
                print("    -> Nhập số (vd: 0, 1, 25.5). Thử lại.")
    return pd.DataFrame([record], columns=feature_cols)

def predict_one(preprocess, booster, thr, df_input: pd.DataFrame):
    Xp = preprocess.transform(df_input)
    proba = float(booster.predict_proba(Xp)[:, 1][0])  # P(Risk)
    pred = 1 if proba >= thr else 0
    label = "Risk" if pred == 1 else "NoRisk"
    return proba, pred, label

def main():
    preprocess, booster, thr = load_bundle(MODEL_PATH)
    feature_cols = get_feature_columns(CSV_PATH, TARGET_COL)

    print("✅ Loaded model:", MODEL_PATH)
    print("✅ Threshold:", thr)
    print("✅ Number of features:", len(feature_cols))

    while True:
        df_user = prompt_user_input(feature_cols)
        proba, pred, label = predict_one(preprocess, booster, thr, df_user)

        print("\n===== KẾT QUẢ DỰ ĐOÁN =====")
        print(f"Xác suất Risk: {proba:.4f}")
        print(f"Dự đoán: {label} (pred={pred})")
        print("===========================\n")

        again = input("Dự đoán tiếp? (y/n): ").strip().lower()
        if again != "y":
            break

if __name__ == "__main__":
    main()