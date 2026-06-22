#  CardioCare: 심장병 예측 및 전주기 MLOps 파이프라인 구축 프로젝트

본 프로젝트는 UCI 심장병 데이터셋(Heart Disease Dataset)을 활용하여, 환자의 임상적 특징을 기반으로 심장 질환 여부를 분류하는 설명 가능한 의사결정 파이프라인 및 MLOps 엔지니어링 시스템입니다. 

---

## Phase 2 & 3: 최종 모델 선정 및 임상적 정당화 (1장 토론 Q1 연계)

* 최종 채택 모델: 하이퍼파라미터가 최적화된 로지스틱 회귀 (Optimized Logistic Regression)
* 임상적 가치 및 위음성(False Negative) 방어 논리:
  - 심장 질환 진단에서 위음성(FN, 환자를 정상으로 오진)은 즉각적인 치료 지연과 환자의 치명적인 사망 위험으로 이어지므로 최우선으로 관리되어야 합니다.
  - 채택된 최적화 재현율(Recall/Sensitivity) 0.8929를 확보하여 실제 위험군 환자를 놓치는 비율을 최소화했습니다.
  - 또한, 전체적인 분류 경계 분별력을 대변하는 ROC-AUC 점수가 0.9697, Balanced Accuracy가 0.8858로 3개 비교 계열(Linear, Kernel, Ensemble) 중 가장 압도적인 성능을 보였습니다.
  - 예측 성능이 유사한 Random Forest 등의 블랙박스 모델과 달리, 본 선형 모델은 각 임상 지표가 발병 위험에 미치는 가중치(Weights)의 투명한 해석이 가능하여 실제 의료진에게 가장 높은 신뢰성과 설명 가능성(Explainability)을 제공합니다.

---

## MLOps 아키텍처 설계 및 논리 정당화 (서술형 문항)

## 1. 피처 스토어(Feature Store; Feast) 도입 가치
* 대상 피처: `patient_historical_avg_chol` (환자의 과거 5년간 평균 콜레스테롤 수치)
* 도입 이유: 단일 시점의 콜레스테롤 수치는 환자의 당일 식단이나 컨디션에 따라 변동성(노이즈)이 매우 큽니다. 장기적인 시계열 이동 평균 데이터를 피처 스토어에 적재해 두면, 실시간 추론 시점에 지연 시간 없이 안전하게 피처를 결합(Online Join)할 수 있으며 학습과 서빙 간의 데이터 누수(Data Leakage)를 방지하여 진단의 일관성을 유지할 수 있습니다.

### 2. 모델 레지스트리(Model Registry; MLflow) 메타데이터 기록
* 대상 메타데이터: `feature_selection_mask` (특성 선택 단계에서 채택된 11개 핵심 변수 명세 리스트)
* 도입 이유: 본 파이프라인은 28개 특성 중 `SelectFromModel` 알고리즘을 거쳐 엄선된 11개 특성만을 최종 모델에 주입합니다. 이 변수 명세(Mask)가 레지스트리에 메타데이터로 박제되어 있어야만, 향후 운영 환경에 새로운 환자 데이터가 들어왔을 때 입력 스키마 구조의 변질이나 입력 데이터의 분포 변화(Data Drift)를 상시 감시하고 모델 계보(Lineage)를 완벽하게 추적할 수 있습니다.

---

## 5단계 전과정 재현 가이드
아래의 5단계를 순서대로 실행하여 소스코드 검증, 무결성 테스트, 패키징, 데이터 드리프트 감시 환경을 그대로 재현할 수 있습니다.

* 1단계: 리포지토리 클론 및 데이터 결정론적 적재
```bash
# 1. 레포지토리 가져오기 및 진입
git clone https://github.com/plpnse15/CardioCare.git
cd CardioCare_new

# 2. 가상환경 활성화 및 의존성 패키지 설치
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # (윈도우 기준)
pip install -r requirements.txt

# 3. 파이프라인 데이터셋 다운로드
python download_data.py
```

* 2단계: 3대 계열 모델 비교 학습 및 하이퍼파라미터 최적화 실행
```bash
python src/train.py
```

 * 3단계: Docker 컨테이너 패키징 및 배치 추론 실행 
 - Docker 이미지 빌드
 ```bash
 docker build -t cardioccare:1.0 .
 docker run --rm cardioccare:1.0
```

 * 4단계: 4대 핵심 요건 파이프라인 Unittest 가동
 ```bash
 python -m unittest discover -s tests -p "test_*.py"
 ```