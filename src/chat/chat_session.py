"""
채팅 세션 관리 모듈.

대화 세션의 생성, 관리, 연결 등을 담당합니다.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import WebSocket
from pydantic import BaseModel

# 로깅 설정
logger = logging.getLogger(__name__)


class Message(BaseModel):
    """채팅 메시지 모델"""
    role: str  # 'user' 또는 'assistant'
    content: str
    timestamp: datetime = None

    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now()
        super().__init__(**data)


class ChatSession:
    """채팅 세션 클래스"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Message] = []
        self.created_at = datetime.now()
        self.last_active = datetime.now()
        
    def add_message(self, role: str, content: str) -> Message:
        """메시지 추가"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.last_active = datetime.now()
        return message
        
    def get_context(self, max_messages: int = 10) -> List[Message]:
        """최근 N개 메시지를 컨텍스트로 반환"""
        return self.messages[-max_messages:] if self.messages else []
    
    def to_dict(self) -> Dict[str, Any]:
        """세션 정보를 딕셔너리로 변환"""
        return {
            "session_id": self.session_id,
            "messages": [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp.isoformat()}
                for msg in self.messages
            ],
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """딕셔너리에서 세션 객체 생성"""
        session = cls(data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_active = datetime.fromisoformat(data["last_active"])
        
        for msg_data in data["messages"]:
            message = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"])
            )
            session.messages.append(message)
        
        return session


class ConnectionManager:
    """웹소켓 연결 관리자"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, session_id: str, websocket: WebSocket):
        """웹소켓 연결 수립"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"Client connected: {session_id}")
        
    def disconnect(self, session_id: str):
        """웹소켓 연결 종료"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Client disconnected: {session_id}")
    
    async def send_message(self, session_id: str, message: dict):
        """특정 세션에 메시지 전송"""
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json(message)


class ChatSessionManager:
    """채팅 세션 관리자"""
    
    def __init__(self):
        self.sessions: Dict[str, ChatSession] = {}
        self.connection_manager = ConnectionManager()
    
    def create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """새 세션 생성"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(session_id)
            logger.info(f"Created new session: {session_id}")
        
        return self.sessions[session_id]
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """세션 정보 조회"""
        return self.sessions.get(session_id)
    
    async def connect(self, session_id: str, websocket: WebSocket):
        """클라이언트 연결 관리"""
        await self.connection_manager.connect(session_id, websocket)
        if session_id not in self.sessions:
            self.create_session(session_id)
    
    def disconnect(self, session_id: str):
        """클라이언트 연결 종료"""
        self.connection_manager.disconnect(session_id)
    
    async def process_message(self, session_id: str, message_text: str, message_handler=None) -> str:
        """메시지 처리"""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id)
        
        # 사용자 메시지 저장
        session.add_message("user", message_text)
        
        # 메시지 핸들러가 있으면 응답 생성
        response = "메시지를 받았습니다."
        if message_handler:
            response = await message_handler.handle_message(session, message_text)
            
        # 응답 메시지 저장
        session.add_message("assistant", response)
        
        # 응답 반환
        return response
    
    def save_sessions(self, storage):
        """모든 세션 저장"""
        for session_id, session in self.sessions.items():
            storage.save_session(session)
    
    def load_sessions(self, storage):
        """세션 로드"""
        for session_data in storage.get_all_sessions():
            session = ChatSession.from_dict(session_data)
            self.sessions[session.session_id] = session 