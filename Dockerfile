# 1단계: 파이썬 최신 호환 환경 베이스 이미지 지정
FROM python:3.13-slim

# 2단계: 컨테이너 내부 작업 디렉토리 생성 및 설정
WORKDIR /workspace

# 3단계: 빌드 캐시 최적화를 위해 의존성 파일 먼저 복사
COPY requirements.txt .

# 4단계: 경량 무중단 의존성 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5단계: 프로젝트 코드, 모델 자산, 테스트 데이터 일체 복사 (src, data, tests 폴더 포함)
COPY . .

# 6단계: 컨테이너가 가동되었을 때 실행될 추론 엔트리포인트 설정
# [경로 수정] app.py 대신 새 위치인 src/inference.py를 실행하도록 변경
ENTRYPOINT ["python", "src/inference.py"]