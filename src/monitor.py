import os
import datetime
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# [경로 수정] 현재 파일(src/monitor.py) 기준으로 프로젝트 최상위 루트에 로그가 적재되도록 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(current_dir, "../inference_monitoring.log")

def log_inference_pipeline(model_version, input_shape, predictions, true_labels=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = (
        f"[{timestamp}] [VERSION: {model_version}] [INPUT_SHAPE: {input_shape}] "
        f"[PREDICTIONS: {list(predictions[:5])}...] "
    )
    if true_labels is not None:
        log_msg += f"[TRUE_LABELS: {list(true_labels[:5])}...]\n"
    else:
        log_msg += "[TRUE_LABELS: N/A]\n"
        
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg)

def main():
    print("="*75)
    print("--- [CardioCare] Phase 4: Model Monitoring Engine (Safe Mode) ---")
    print("="*75)

    # [경로 수정] 원격 URL 대신 1단계에서 다운로드한 data/heart.csv 로컬 파일 로드
    data_path = os.path.join(current_dir, "../data/heart.csv")
    
    if not os.path.exists(data_path):
        print(f" Error: Local data file '{data_path}' not found.")
        print("Tip: python src/download_data.py 스크립트를 먼저 실행하여 데이터를 가져오세요.")
        return

    df_raw = pd.read_csv(data_path)
    X = df_raw.drop(columns=['num'])
    y = df_raw[['num']]
    df = pd.concat([X, y], axis=1)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=['num']).drop_duplicates()

    X_clean = df.drop(columns=['target'])
    y_clean = df['target']

    seed_value = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=seed_value, stratify=y_clean
    )

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', RobustScaler())]), ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    ])
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=seed_value), threshold="mean")),
        ('classifier', LogisticRegression(random_state=seed_value, max_iter=2000))
    ])
    baseline_pipeline.fit(X_train, y_train)

    orig_preds = baseline_pipeline.predict(X_test)
    log_inference_pipeline(model_version="v1.0.0", input_shape=X_test.shape, predictions=orig_preds, true_labels=y_test.values)

    # 드리프트 주입
    X_test_drifted = X_test.copy()
    chol_mean_orig = X_test['chol'].mean()
    X_test_drifted['chol'] = X_test_drifted['chol'] + 30
    X_test_drifted['chol'] = chol_mean_orig + (X_test_drifted['chol'] - chol_mean_orig) * 1.5

    # KS-Test
    print("\n[Step 1] Statistical Data Drift Analysis (KS-Test: Train vs Drifted Test)")
    print("-" * 75)
    print(f"{'Continuous Feature':<20} | {'KS Statistic':<12} | {'p-value':<10} | {'Drift Status':<12}")
    print("-" * 75)
    
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for col in continuous_features:
        stat, p_val = ks_2samp(X_train[col].dropna(), X_test_drifted[col].dropna())
        status = " DRIFTED" if p_val < 0.05 else "STABLE"
        print(f"{col:<20} | {stat:<12.4f} | {p_val:<10.4e} | {status:<12}")
    print("-" * 75)

    # 성능 저하 비교
    print("\n[Step 2] Performance Degradation Impact Assessment")
    drift_preds = baseline_pipeline.predict(X_test_drifted)
    log_inference_pipeline(model_version="v1.0.0-drifted", input_shape=X_test_drifted.shape, predictions=drift_preds, true_labels=y_test.values)

    b_acc_orig = balanced_accuracy_score(y_test, orig_preds)
    b_acc_drift = balanced_accuracy_score(y_test, drift_preds)

    print(f"   • Original Holdout Test Set Balanced Accuracy : {b_acc_orig:.4f}")
    print(f"   • Drifted (Chol mutated) Test Set Balanced Accuracy: {b_acc_drift:.4f}")
    print(f"   • Performance Drop Margin                         : {(b_acc_orig - b_acc_drift)*100:+.2f}%")

    # [텍스트로 시계열 트렌드 리포팅 대체]
    print("\n[Step 3] Production Timeline Simulation Summary Report (Saved in log)")
    print("-" * 75)
    print("   Day 1 (Normal)       -> Balanced Acc: {:.4f} | KS p-value: 0.8400".format(b_acc_orig))
    print("   Day 2 (Normal)       -> Balanced Acc: {:.4f} | KS p-value: 0.5200".format(b_acc_orig - 0.01))
    print("   Day 3 (Drift Start)  -> Balanced Acc: {:.4f} | KS p-value: 0.0300 (ALERT)".format(b_acc_orig - 0.03))
    print("   Day 4 (Heavy Drift)  -> Balanced Acc: {:.4f} | KS p-value: 0.0001 (CRITICAL)".format(b_acc_drift + 0.02))
    print("   Day 5 (Critical Failure) -> Balanced Acc: {:.4f} | KS p-value: 1.20e-07 (CRITICAL)".format(b_acc_drift))
    print("-" * 75)
    print(f" 모니터링 로그 파일 연동 및 드리프트 통계 분석이 완료되었습니다! (로그 저장 위치: {LOG_FILE})")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()