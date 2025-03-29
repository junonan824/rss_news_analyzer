FROM python:3.9-slim

WORKDIR /app

# 타임존 설정
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 애플리케이션 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# 애플리케이션 코드 복사
COPY . .

# 데이터 볼륨 디렉토리 생성
RUN mkdir -p /app/data/api /app/data/rss /app/data/vector_db

# 8000번 포트 노출 (FastAPI)
EXPOSE 8000

# 환경 변수 설정
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 서버 실행
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"] 