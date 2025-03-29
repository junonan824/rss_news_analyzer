"""
채팅 세션 저장 모듈.

세션과 메시지를 영구 저장하기 위한 기능을 제공합니다.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional

# 로깅 설정
logger = logging.getLogger(__name__)


class FileStorage:
    """파일 기반 세션 저장소"""
    
    def __init__(self, storage_dir: str = "data/chat_sessions"):
        """
        Args:
            storage_dir: 세션 데이터 저장 디렉토리
        """
        self.storage_dir = storage_dir
        self._ensure_storage_dir()
    
    def _ensure_storage_dir(self):
        """저장 디렉토리 확인 및 생성"""
        try:
            if not os.path.exists(self.storage_dir):
                os.makedirs(self.storage_dir)
                logger.info(f"Created storage directory: {self.storage_dir}")
        except Exception as e:
            logger.error(f"Failed to create storage directory: {e}")
    
    def save_session(self, session) -> bool:
        """세션 저장
        
        Args:
            session: 저장할 세션 객체
            
        Returns:
            성공 여부
        """
        try:
            # 세션 데이터를 JSON으로 변환
            session_data = session.to_dict()
            
            # 파일 경로 생성
            file_path = os.path.join(self.storage_dir, f"{session.session_id}.json")
            
            # JSON 파일로 저장
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Session saved: {session.session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 로드
        
        Args:
            session_id: 로드할 세션 ID
            
        Returns:
            세션 데이터 딕셔너리 또는 None
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            
            # 파일이 없으면 None 반환
            if not os.path.exists(file_path):
                logger.warning(f"Session file not found: {file_path}")
                return None
            
            # JSON 파일 로드
            with open(file_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            logger.info(f"Session loaded: {session_id}")
            return session_data
        
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제
        
        Args:
            session_id: 삭제할 세션 ID
            
        Returns:
            성공 여부
        """
        try:
            file_path = os.path.join(self.storage_dir, f"{session_id}.json")
            
            # 파일이 없으면 실패 반환
            if not os.path.exists(file_path):
                logger.warning(f"Session file not found: {file_path}")
                return False
            
            # 파일 삭제
            os.remove(file_path)
            logger.info(f"Session deleted: {session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """모든 세션 목록 조회
        
        Returns:
            세션 데이터 딕셔너리 목록
        """
        try:
            sessions = []
            
            # 디렉토리 내 모든 JSON 파일 로드
            for filename in os.listdir(self.storage_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.storage_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                            sessions.append(session_data)
                    except Exception as e:
                        logger.error(f"Error loading session file {filename}: {e}")
                        continue
            
            logger.info(f"Loaded {len(sessions)} sessions")
            return sessions
        
        except Exception as e:
            logger.error(f"Failed to get all sessions: {e}")
            return []


class SQLiteStorage:
    """SQLite 기반 세션 저장소"""
    
    def __init__(self, db_path: str = "data/chat_sessions.db"):
        """
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
    
    def _ensure_db_dir(self):
        """DB 디렉토리 확인 및 생성"""
        db_dir = os.path.dirname(self.db_path)
        try:
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created DB directory: {db_dir}")
        except Exception as e:
            logger.error(f"Failed to create DB directory: {e}")
    
    def _init_db(self):
        """데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 세션 테이블 생성
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                last_active TEXT
            )
            ''')
            
            # 메시지 테이블 생성
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def save_session(self, session) -> bool:
        """세션 저장
        
        Args:
            session: 저장할 세션 객체
            
        Returns:
            성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 세션 정보 저장/업데이트
            cursor.execute(
                "INSERT OR REPLACE INTO sessions (id, created_at, last_active) VALUES (?, ?, ?)",
                (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.last_active.isoformat()
                )
            )
            
            # 이전 메시지 삭제 (갱신 방식)
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session.session_id,))
            
            # 메시지 저장
            for msg in session.messages:
                cursor.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                    (
                        session.session_id,
                        msg.role,
                        msg.content,
                        msg.timestamp.isoformat()
                    )
                )
            
            conn.commit()
            conn.close()
            logger.info(f"Session saved to SQLite: {session.session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save session to SQLite {session.session_id}: {e}")
            return False
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션 로드
        
        Args:
            session_id: 로드할 세션 ID
            
        Returns:
            세션 데이터 딕셔너리 또는 None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
            cursor = conn.cursor()
            
            # 세션 정보 조회
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            session_row = cursor.fetchone()
            
            if not session_row:
                logger.warning(f"Session not found: {session_id}")
                conn.close()
                return None
            
            session_data = dict(session_row)
            
            # 메시지 조회
            cursor.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            
            messages = []
            for msg_row in cursor.fetchall():
                messages.append({
                    "role": msg_row["role"],
                    "content": msg_row["content"],
                    "timestamp": msg_row["timestamp"]
                })
            
            # 세션 데이터 구성
            result = {
                "session_id": session_id,
                "created_at": session_data["created_at"],
                "last_active": session_data["last_active"],
                "messages": messages
            }
            
            conn.close()
            logger.info(f"Session loaded from SQLite: {session_id}")
            return result
        
        except Exception as e:
            logger.error(f"Failed to load session from SQLite {session_id}: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """세션 삭제
        
        Args:
            session_id: 삭제할 세션 ID
            
        Returns:
            성공 여부
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 세션 존재 여부 확인
            cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
            if not cursor.fetchone():
                logger.warning(f"Session not found: {session_id}")
                conn.close()
                return False
            
            # 세션 및 관련 메시지 삭제
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            
            conn.commit()
            conn.close()
            logger.info(f"Session deleted from SQLite: {session_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete session from SQLite {session_id}: {e}")
            return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """모든 세션 목록 조회
        
        Returns:
            세션 데이터 딕셔너리 목록
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 모든 세션 조회
            cursor.execute("SELECT id FROM sessions ORDER BY last_active DESC")
            session_rows = cursor.fetchall()
            
            sessions = []
            for row in session_rows:
                session_id = row["id"]
                session_data = self.load_session(session_id)
                if session_data:
                    sessions.append(session_data)
            
            conn.close()
            logger.info(f"Loaded {len(sessions)} sessions from SQLite")
            return sessions
        
        except Exception as e:
            logger.error(f"Failed to get all sessions from SQLite: {e}")
            return [] 