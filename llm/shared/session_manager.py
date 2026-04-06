import json
import uuid
import time
from pathlib import Path
from typing import List, Dict

from shared.config import SESSIONS_DIR


class SessionManager:
    def __init__(self, storage_path: Path = None):
        if storage_path is None:
            storage_path = SESSIONS_DIR
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, List[Dict]] = {}
        self._load_all()
        
        print(f"Менеджер сессий инициализирован")
        print(f"Загружено {len(self._sessions)} сессий")
    
    def _load_all(self):
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._sessions[data['session_id']] = data['messages']
            except Exception as e:
                print(f"Ошибка загрузки {file_path}: {e}")
    
    def _save(self, session_id: str):
        if session_id not in self._sessions:
            return
        
        file_path = self.storage_path / f"{session_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": session_id,
                "messages": self._sessions[session_id],
                "updated_at": time.time()
            }, f, ensure_ascii=False, indent=2)
    
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())[:8]
        self._sessions[session_id] = []
        self._save(session_id)
        print(f"Создана сессия: {session_id}")
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        if session_id not in self._sessions:
            return False
        
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        self._sessions[session_id].append(message)
        self._save(session_id)
        return True
    
    def get_history(self, session_id: str, limit: int = None) -> List[Dict]:
        messages = self._sessions.get(session_id, [])
        if limit:
            messages = messages[-limit:]
        return messages
    
    def get_context_string(self, session_id: str, limit: int = 2) -> str:
        history = self.get_history(session_id, limit)
        if not history:
            return ""
        
        lines = []
        for msg in history:
            role_name = "Пользователь" if msg['role'] == 'user' else "Ассистент"
            lines.append(f"{role_name}: {msg['content']}")
        
        return "\n".join(lines)
    
    def get_all_sessions(self) -> List[Dict]:
        sessions = []
        for session_id, messages in self._sessions.items():
            sessions.append({
                "session_id": session_id,
                "message_count": len(messages),
                "title": messages[0]['content'][:40] + "..." if messages else "Новый диалог",
                "created_at": messages[0]['timestamp'] if messages else time.time(),
                "updated_at": messages[-1]['timestamp'] if messages else time.time()
            })
        
        sessions.sort(key=lambda x: x['updated_at'], reverse=True)
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        
        del self._sessions[session_id]
        file_path = self.storage_path / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
        
        print(f"Удалена сессия: {session_id}")
        return True
    
    def clear_session(self, session_id: str) -> bool:
        if session_id not in self._sessions:
            return False
        
        self._sessions[session_id] = []
        self._save(session_id)
        print(f"🧹 Очищена сессия: {session_id}")
        return True
    
    def get_stats(self) -> Dict:
        total_messages = sum(len(msgs) for msgs in self._sessions.values())
        return {
            "total_sessions": len(self._sessions),
            "total_messages": total_messages,
            "storage_path": str(self.storage_path)
        }


__all__ = ['SessionManager']