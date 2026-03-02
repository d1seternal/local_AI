import time
import uuid
import pathlib
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import torch

class VectorMemory:
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "conversations",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
       
        self.persist_directory = pathlib.Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._init_embeddings()
        self._init_chromadb()
    
    def _init_embeddings(self) -> None:
        print(f"Загрузка модели эмбеддингов: {self.embedding_model_name}")
        start_time = time.time()
        device = "cpu"
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=device
        )
        
        print(f"Модель загружена за {time.time() - start_time:.2f} сек")
    
    def _init_chromadb(self) -> None:
        print(f"Подключение к ChromaDB: {self.persist_directory}")
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Подключено к существующей коллекции (документов: {self.collection.count()})")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  
            )
            print(f"Создана новая коллекция")
    
    def _get_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def add_message(
        self,
        role: str,
        content: str,
        session_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        
        full_text = f"{role}: {content}"
        doc_id = str(uuid.uuid4())
        embedding = self._get_embedding(full_text)
        meta = {
            "role": role,
            "session_id": session_id,
            "timestamp": time.time(),
            "content_preview": content[:100]
        }
        if metadata:
            meta.update(metadata)

        self.collection.add(
            embeddings=[embedding],
            documents=[full_text],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        return doc_id
    
    def search_similar(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.7,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
    
        query_embedding = self._get_embedding(query)
        where_filter = {"session_id": session_id} if session_id else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2, 
            where=where_filter
        )
        
        formatted_results = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                score = max(0, 1 - distance / 2) 
                
                if score >= score_threshold:
                    formatted_results.append({
                        "id": results['ids'][0][i],
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "relevance_score": score
                    })
        
        return formatted_results[:k]
    
    def get_relevant_context(
        self,
        query: str,
        k: int = 3,
        session_id: Optional[str] = None
    ) -> str:
        
        similar = self.search_similar(
            query=query,
            k=k,
            score_threshold=0.7,
            session_id=session_id
        )
        
        if not similar:
            return ""
        
        lines = ["Релевантные диалоги:"]
        for msg in similar:
            role_symbol = "user" if msg["metadata"]["role"] == "user" else "AI"
            preview = msg["text"][:200] + "..." if len(msg["text"]) > 200 else msg["text"]
            lines.append(f"{role_symbol} [{msg['relevance_score']:.2f}] {preview}")
        
        return "\n".join(lines)
    
    def count(self) -> int:
        return self.collection.count()
    
    def delete_session(self, session_id: str) -> int:
        results = self.collection.get(where={"session_id": session_id})
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0

__all__ = ['VectorMemory']
