# src/download_data.py
import os
import pandas as pd

def download_heart_disease_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '../data')
    os.makedirs(data_dir, exist_ok=True)
    
    raw_data_url = "https://archive.ics.uci.edu/static/public/45/data.csv"
    
    print(" UCI 원격 서버에서 데이터를 다운로드하는 중...")
    try:
        df = pd.read_csv(raw_data_url)
        output_path = os.path.join(data_dir, 'heart.csv')
        df.to_csv(output_path, index=False)
        print(f" 데이터 다운로드 완료 및 저장 성공! 경로: {output_path}")
    except Exception as e:
        print(f" 다운로드 오류 발생: {e}")

if __name__ == "__main__":
    download_heart_disease_data()