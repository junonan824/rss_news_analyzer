#!/bin/bash

# RSS 분석기 서버 재시작 스크립트

# 포트 설정
PORT=8000
SRC_PATH="src.app"
PYTHON_CMD="python"

echo "===== RSS 뉴스 분석기 서버 재시작 ====="

# 1. 이전 서버 프로세스 찾아서 종료
echo "기존 서버 프로세스 확인 중..."
PIDS=$(lsof -t -i:$PORT 2>/dev/null)

if [ -n "$PIDS" ]; then
    echo "포트 $PORT를 사용 중인 프로세스 발견: $PIDS"
    for PID in $PIDS; do
        echo "프로세스 $PID 종료 중..."
        kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null
        sleep 1
    done
    echo "기존 프로세스 종료 완료"
else
    echo "사용 중인 프로세스가 없습니다."
fi

# 추가적인 정리: Python 프로세스 확인
echo "관련 Python 프로세스 확인 중..."
PY_PIDS=$(ps aux | grep "[p]ython -m $SRC_PATH" | awk '{print $2}')

if [ -n "$PY_PIDS" ]; then
    echo "Python 프로세스 발견: $PY_PIDS"
    for PID in $PY_PIDS; do
        echo "프로세스 $PID 종료 중..."
        kill -15 $PID 2>/dev/null || kill -9 $PID 2>/dev/null
        sleep 1
    done
    echo "Python 프로세스 종료 완료"
fi

# 2. 서버 실행
echo "서버 시작 중..."
$PYTHON_CMD -m $SRC_PATH &

# 서버 시작 대기
sleep 2

# 3. 서버 상태 확인
if lsof -i:$PORT > /dev/null 2>&1; then
    echo "서버가 성공적으로 시작되었습니다. (포트: $PORT)"
    
    # 서버 정보 표시
    SERVER_PID=$(lsof -t -i:$PORT)
    echo "서버 PID: $SERVER_PID"
    
    # 접속 URL 표시
    echo "접속 URL: http://localhost:$PORT/chat"
else
    echo "서버 시작 실패! 로그를 확인하세요."
fi

echo "===== 스크립트 완료 =====" 