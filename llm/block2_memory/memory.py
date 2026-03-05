import time
import uuid
import pathlib
import shutil
from typing import List, Dict, Optional, Any

import chromadb
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("поддержка PDF отключена")

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("поддержка DOCX отключена")



class VectorMemory:
    
    def __init__(
        self,
        persist_directory: str = "./rag_data",
        memory_collection: str = "conversations",
        docs_collection: str = "documents",
        embedding_model: str = "intfloat/multilingual-e5-base",
        chunk_size: int = 150,
        chunk_overlap: int = 30
    ):
        
        self.persist_directory = pathlib.Path(persist_directory)
        self.memory_collection_name = memory_collection
        self.docs_collection_name = docs_collection
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
     
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.documents_dir = self.persist_directory / "source_docs"
        self.documents_dir.mkdir(exist_ok=True)
        
        self._init_embeddings()
        self._init_chromadb()
     
    def _init_embeddings(self):
        print(f"Загрузка модели эмбеддингов: {self.embedding_model_name}")
        start_time = time.time()
        device = "cpu"
        
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=device
        )
        
        print(f"Модель загружена за {time.time() - start_time:.2f} сек")

    def _init_chromadb(self):
        print(f"Подключение к ChromaDB: {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.memory_collection = self.client.get_collection(self.memory_collection_name)
            print(f"Подключено к существующей истории")
        except Exception as e:
            self.memory_collection = self.client.create_collection(
                name=self.memory_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Создана новая история")

        try:
            self.docs_collection = self.client.get_collection(self.docs_collection_name)
            print(f"Подключено к существующей коллекции документов")
        except Exception as e:
            self.docs_collection = self.client.create_collection(
                name=self.docs_collection_name,
                metadata={"hnsw :space": "cosine"}
            )
            print(f"Создана новая коллекция документов")
    
    def _get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        if "e5" in self.embedding_model_name:
            prefix = "query: " if is_query else "passage: "
            text = prefix + text
        
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()


    # def _get_embedding(self, text: str) -> List[float]:
        
    #     embedding = self.embedding_model.encode(text, normalize_embeddings=True)
    #     return embedding.tolist()
    
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
            "type": "conversation",
            "preview": content[:100]
        }
        if metadata:
            meta.update(metadata)
        
        self.memory_collection.add(
            embeddings=[embedding],
            documents=[full_text],
            metadatas=[meta],
            ids=[doc_id]
        )
        
        return doc_id
    
    # def search_in_document(
    #     self,
    #     query: str,
    #     doc_id: str,
    #     k: int = 3,
    #     score_threshold: float = 0.5
    # ) -> List[Dict[str, Any]]:
    #     query_embedding = self._get_embedding(query, is_query=True)

    #     results = self.docs_collection.query(
    #         query_embeddings=[query_embedding],
    #         n_results=k * 2,
    #         where={"doc_id": doc_id}
    #     )
        
    #     formatted = []
    #     if results['ids'][0]:
    #         for i in range(len(results['ids'][0])):
    #             distance = results['distances'][0][i] if results['distances'] else 1.0
    #             score = max(0, 1 - distance / 2)
                
    #             if score >= score_threshold:
    #                 formatted.append({
    #                     "text": results['documents'][0][i],
    #                     "metadata": results['metadatas'][0][i],
    #                     "relevance_score": score
    #                 })
        
    #     return formatted[:k]

    # def get_document_context_by_id(
    #     self,
    #     query: str,
    #     doc_id: str,
    #     k: int = 3
    # ) -> str:
    #     results = self.search_in_document(query, doc_id, k=k)
        
    #     if not results:
    #         return ""

    #     filename = results[0]['metadata'].get('filename', 'документ')
        
    #     lines = [f"[Информация из документа: {filename}]"]
    #     for i, r in enumerate(results, 1):
    #         lines.append(f"\n{i}. [релевантность: {r['relevance_score']:.2f}]")
    #         lines.append(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])
        
    #     return "\n".join(lines)
    
    def search_messages(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.5,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query, is_query=True)
        
        where_filter = {"session_id": session_id} if session_id else None
        
        results = self.memory_collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,
            where=where_filter
        )
        
        formatted = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                score = max(0, 1 - distance / 2)
                
                if score >= score_threshold:
                    formatted.append({
                        "text": results['documents'][0][i],
                        "role": results['metadatas'][0][i].get("role", "unknown"),
                        "session_id": results['metadatas'][0][i].get("session_id", ""),
                        "timestamp": results['metadatas'][0][i].get("timestamp", 0),
                        "relevance_score": score
                    })
        
        return formatted[:k]
    
    def get_memory_context(
        self,
        query: str,
        k: int = 3,
        session_id: Optional[str] = None
    ) -> str:
    
        results = self.search_messages(query, k=k, session_id=session_id)
        
        if not results:
            return ""
        
        lines = ["[История диалогов:]"]
        for msg in results:
            role_marker = "Пользователь" if msg["role"] == "user" else "Ассистент"
            preview = msg["text"][:200] + "..." if len(msg["text"]) > 200 else msg["text"]
            lines.append(f"{role_marker}: {preview}")
        
        return "\n".join(lines)
    
    def get_session_messages(self, session_id: str, limit: int = 50) -> List[Dict]:
        results = self.memory_collection.get(
            where={"session_id": session_id},
            limit=limit
        )
        
        messages = []
        if results['ids']:
            for i in range(len(results['ids'])):
                messages.append({
                    "id": results['ids'][i],
                    "text": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
        
        return messages
    
    def _chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        words = text.split()
        
        if len(words) <= self.chunk_size:
            chunks.append({
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['doc_id']}_0",
                    "chunk_index": 0,
                    "chunk_total": 1
                }
            })
            return chunks
        
        step = self.chunk_size - self.chunk_overlap
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:
                continue
                
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_id": f"{metadata['doc_id']}_{i}",
                    "chunk_index": i // step,
                    "chunk_total": (len(words) + step - 1) // step,
                    "start_word": i,
                    "end_word": i + len(chunk_words)
                }
            })
        
        return chunks
    
    def _read_file(self, file_path: pathlib.Path) -> str:
        ext = file_path.suffix.lower()
        if ext == '.pdf' and PDF_SUPPORT:
            try:
                text = []
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text.append(page_text)
                return "\n".join(text)
            except Exception as e:
                return f"[Ошибка чтения PDF: {e}]"
        
        elif ext in ['.docx', '.doc'] and DOCX_SUPPORT:
            try:
                doc = DocxDocument(file_path)
                return "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                return f"[Ошибка чтения DOCX: {e}]"
        
        else:
            return f"[Неподдерживаемый формат файла: {ext}]"
    
    def index_document(self, file_path: str, doc_id: Optional[str] = None) -> int:
        file_path = pathlib.Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        text = self._read_file(file_path)
        
        if not text.strip():
            return 0

        dest_path = self.documents_dir / f"{doc_id}_{file_path.name}"
        shutil.copy2(file_path, dest_path)
        
        doc_metadata = {
            "doc_id": doc_id,
            "filename": file_path.name,
            "stored_path": str(dest_path),
            "file_size": file_path.stat().st_size,
            "timestamp": time.time(),
            "type": "document"
        }
        
        chunks = self._chunk_text(text, doc_metadata)
        
        for chunk in chunks:
            embedding = self._get_embedding(chunk["text"], is_query=False)
            self.docs_collection.add(
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
                ids=[chunk["metadata"]["chunk_id"]]
            )
        return len(chunks)
    
    def index_text(self, text: str, source: str = "text_input", doc_id: Optional[str] = None) -> int:
        if doc_id is None:
            doc_id = f"text_{uuid.uuid4().hex[:8]}"
        
        doc_metadata = {
            "doc_id": doc_id,
            "source": source,
            "timestamp": time.time(),
            "type": "text"
        }
        
        chunks = self._chunk_text(text, doc_metadata)
        
        for chunk in chunks:
            embedding = self._get_embedding(chunk["text"], is_query=False)
            self.docs_collection.add(
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
                ids=[chunk["metadata"]["chunk_id"]]
            )

        return len(chunks)
    
    def search_documents(
        self,
        query: str,
        k: int = 5,
        score_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
      
        query_embedding = self._get_embedding(query, is_query=True)
        
        results = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2
        )
        
        formatted = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                score = max(0, 1 - distance / 2)
                
                if score >= score_threshold:
                    formatted.append({
                        "text": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "relevance_score": score
                    })
        
        return formatted[:k]
    
    def get_document_context(
        self,
        query: str,
        k: int = 3
    ) -> str:
       
        results = self.search_documents(query, k=k)
        
        if not results:
            return ""
        
        lines = ["[Информация из документов:]"]
        for i, r in enumerate(results, 1):
            source = r['metadata'].get('filename', r['metadata'].get('source', 'unknown'))
            lines.append(f"\n{i}. [Из {source}, релевантность: {r['relevance_score']:.2f}]")
            lines.append(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])
        
        return "\n".join(lines)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        results = self.docs_collection.get()
        
        docs = {}
        for metadata in results['metadatas']:
            doc_id = metadata.get('doc_id')
            if doc_id and doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": metadata.get('filename', metadata.get('source', 'unknown')),
                    "chunks": 0,
                    "timestamp": metadata.get('timestamp', 0)
                }
            if doc_id:
                docs[doc_id]['chunks'] += 1
        
        return sorted(docs.values(), key=lambda x: x['timestamp'], reverse=True)
    
    def delete_document(self, doc_id: str) -> int:
        results = self.docs_collection.get(where={"doc_id": doc_id})
        
        if results['ids']:
            self.docs_collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0
    
    def delete_session(self, session_id: str) -> int:
        results = self.memory_collection.get(where={"session_id": session_id})
        
        if results['ids']:
            self.memory_collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0
    
    def count_memory(self) -> int:
        return self.memory_collection.count()
    
    def count_documents(self) -> int:
        return self.docs_collection.count()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "memory_messages": self.count_memory(),
            "document_chunks": self.count_documents(),
            "documents": len(self.list_documents()),
            "memory_collection": self.memory_collection_name,
            "docs_collection": self.docs_collection_name,
            "embedding_model": self.embedding_model_name
        }


__all__ = ['VectorMemory']