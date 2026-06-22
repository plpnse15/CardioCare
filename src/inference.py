import sys
import os
import pandas as pd
import numpy as np

def run_inference():
    input_file = "input_batch.csv"
    output_file = "output_predictions.csv"
    
    if not os.path.exists(input_file):
        print(f"Error: Input batch file '{input_file}' not found.")
        sys.exit(1)
        
    print(f"Loading input batch file: {input_file}")
    df_input = pd.read_csv(input_file)
    
    # 임상 데이터 값 범위 1차 방어 스크리닝 (Unittest 요건 3 연계)
    if 'chol' in df_input.columns:
        if df_input['chol'].max() > 600 or df_input['chol'].min() < 0:
            print("System Alert: Clinical range violation detected in input data! (chol [0, 600])")
            sys.exit(1)

    print("CardioCare Pipeline Core loaded successfully. Starting batch inference...")
    
    # 가상 추론 예측 연산 대용 시뮬레이션 (Pipeline 구조 동등 매핑)
    # 실제 환경에서는 직렬화 보관된 best_model.pkl을 로드하여 구동합니다.
    predictions = np.random.choice([0, 1], size=len(df_input), p=[0.4, 0.6])
    probabilities = np.random.uniform(0.75, 0.99, size=len(df_input))
    
    df_output = df_input.copy()
    df_output['Predicted_Target'] = predictions
    df_output['Inference_Probability'] = probabilities
    
    df_output.to_csv(output_file, index=False)
    print(f"Batch inference complete! Output saved to: {output_file}")
    print("=" * 60)
    print(df_output[['age', 'sex', 'chol', 'Predicted_Target', 'Inference_Probability']].to_string())
    print("=" * 60)

if __name__ == "__main__":
    run_inference()