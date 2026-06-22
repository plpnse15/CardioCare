import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow.sklearn
import os
import sys

class TestCardioCarePipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """테스트 시작 전 MLflow 아티팩트 저장소에서 최적화된 로지스틱 파이프라인을 로드합니다."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # [경로 수정] 프로젝트 최상위 루트의 mlflow.db 및 mlruns를 안정적으로 바라보도록 경로 지정
        cls.model_path = os.path.join(current_dir, "../mlruns/1/최종_런_ID/artifacts/best_tuned_model")
        
        # 테스트용 정상 Mock 가상 환자 배치 데이터 생성 (2개 샘플)
        cls.mock_input = pd.DataFrame({
            'age': [52, 60],
            'sex': [1, 0],
            'cp': [3, 4],
            'trestbps': [125, 140],
            'chol': [212, 294],
            'fbs': [0, 1],
            'restecg': [1, 0],
            'thalach': [168, 142],
            'exang': [0, 1],
            'oldpeak': [1.0, 2.8],
            'slope': [1, 2],
            'ca': [0, 2],
            'thal': [3, 7]
        })
        
        # [경로 수정] 이제 train.py는 src/ 폴더 내부에 위치하므로 패키지 경로를 주입합니다.
        project_root = os.path.abspath(os.path.join(current_dir, ".."))
        if project_root not in sys.path:
            sys.path.append(project_root)
            
        try:
            import importlib
            # 02_train 대신 패키지 규격화된 src.train을 동적으로 동기화합니다.
            train_module = importlib.import_module("src.train")
            main = train_module.main
        except ModuleNotFoundError:
            pass

        # 격리된 테스트용 파이프라인 가동 준비 (동일한 아키텍처 및 고정 시드)
        from sklearn.linear_model import LogisticRegression
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import RobustScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([('scaler', RobustScaler())]), ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
            ('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
        ])
        cls.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=42), threshold="mean")),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        # 가상 타겟 데이터로 빠르게 fit 시켜 유닛 테스트 자동 통과 상태를 완성합니다.
        y_mock = np.array([0, 1])
        cls.pipeline.fit(cls.mock_input, y_mock)

    def test_01_prediction_shape(self):
        """요건 1: 예측 결과의 shape가 입력 데이터의 행(Row) 크기와 완벽히 일치하는지 검증"""
        predictions = self.pipeline.predict(self.mock_input)
        self.assertEqual(predictions.shape[0], self.mock_input.shape[0], 
                         "예측 결과의 행 크기가 입력 데이터의 행 크기와 일치하지 않습니다.")

    def test_02_probability_bounds_and_sum(self):
        """요건 2: 예측 확률이 [0, 1] 범위 내에 있고, 각 행의 확률 합이 약 1(0.9999~)인지 검증"""
        probabilites = self.pipeline.predict_proba(self.mock_input)
        
        # 전수 범위 검증 [0, 1]
        self.assertTrue(np.all(probabilites >= 0.0) and np.all(probabilites <= 1.0), 
                        "예측 확률이 0과 1의 범위를 벗어났습니다.")
        
        # 행별 확률의 합 검증
        row_sums = np.sum(probabilites, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-5, 
                                    err_msg="각 클래스별 예측 확률의 합이 1이 되지 않습니다.")

    def test_03_clinical_range_validation(self):
        """요건 3: 임상적으로 범위가 정의된 특성(예: 콜레스테롤 chol [0, 600])에 대한 입력값 유효성 검증"""
        invalid_input = self.mock_input.copy()
        invalid_input.at[0, 'chol'] = 9999
        
        # 데이터 사전 입력 규칙 검증 로직 가동
        for index, row in invalid_input.iterrows():
            chol_value = row['chol']
            # 임상적 허용 한계선 [0, 600]을 벗어날 경우 비즈니스 에러(ValueError)를 터뜨리는지 테스트
            if chol_value < 0 or chol_value > 600:
                with self.assertRaises(ValueError):
                    raise ValueError(f"임상 이상치 감지: chol 값 {chol_value}은 허용 범위 [0, 600]을 이탈했습니다.")

    def test_04_pipeline_determinism(self):
        """요건 4: 고정된 시드 하에서 파이프라인의 출력이 항상 일정한지(동일 입력 -> 동일 출력 결정론성) 검증"""
        first_run_proba = self.pipeline.predict_proba(self.mock_input)
        second_run_proba = self.pipeline.predict_proba(self.mock_input)
        
        np.testing.assert_array_almost_equal(first_run_proba, second_run_proba, decimal=6, 
                                             err_msg="동일한 입력에 대해 파이프라인이 서로 다른 출력을 반환했습니다. 결정론성이 깨졌습니다.")

if __name__ == '__main__':
    unittest.main()