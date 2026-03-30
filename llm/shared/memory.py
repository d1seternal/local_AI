import time
import json
import re
import uuid
import pathlib
from pathlib import Path
import shutil
from typing import List, Dict, Optional, Any

import chromadb
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.pipeline_options import TableFormerMode

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


from shared.config import DATA_DIR, MEMORY_PATH, CHUNK_SIZE, CHUNK_OVERLAP,  DOCS_COLLECTION, MEMORY_COLLECTION, EMBEDDING_MODEL
from shared.reranker import LocalLLMReranker
from shared.document_parser import DocumentProcessor
from shared.prompts import DocumentSearchPrompt, EnhancedNameSearchPrompt, NumericValueSearchPrompt, KeywordSearchPrompt, UnifiedRAGPrompt

class VectorMemory:
    
    def __init__(
        self,
        persist_directory = str(MEMORY_PATH),
        memory_collection = MEMORY_COLLECTION,
        docs_collection = DOCS_COLLECTION,
        embedding_model = EMBEDDING_MODEL,
        doc_processor: Optional[DocumentProcessor] = None,
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    ):
        
        if doc_processor is None:
            self.doc_processor = DocumentProcessor() 
        else:
            self.doc_processor = doc_processor
        
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
    
    def index_document(self, file_path: str, doc_id: Optional[str] = None) -> int:
        raw_input = str(file_path)
        print(f"Получено для индексации: '{raw_input}'")
        
        try:
            parsed = json.loads(raw_input)
            if isinstance(parsed, dict):
                filename = parsed.get('filepath', parsed.get('file', ''))
                if filename:
                    raw_input = filename
                    print(f"Извлечено из JSON: '{filename}'")
        except json.JSONDecodeError:
            pass
        
        cleaned = re.sub(r'^[\s"\'{}\[\]()]+|[\s"\'{}\[\]()]+$', '', raw_input)
        cleaned = cleaned.replace('\\', '/')
        cleaned = re.sub(r'^filepath["\']?\s*[:=]\s*', '', cleaned)
        cleaned = pathlib.Path(cleaned).name
        
        print(f"Ищем файл: '{cleaned}'")
        
        found_path = None
        test_path = DATA_DIR / cleaned
        if test_path.exists():
            found_path = test_path
            print(f"Файл найден: {test_path}")
        
        if not found_path:
            for f in DATA_DIR.glob("*"):
                if f.is_file() and f.name.lower() == cleaned.lower():
                    found_path = f
                    print(f"Найден по регистру: {f}")
                    break
        
        if not found_path:
            for f in DATA_DIR.glob("*"):
                if f.is_file():
                    if cleaned.lower() in f.name.lower() or f.name.lower() in cleaned.lower():
                        found_path = f
                        print(f"Найден по частичному: {f}")
                        break
        
        if not found_path:
            available = [f.name for f in DATA_DIR.glob("*") if f.is_file()]
            if available:
                raise FileNotFoundError(
                    f"Файл '{cleaned}' не найден.\n\nДоступные:\n" + 
                    "\n".join([f"  • {f}" for f in sorted(available)[:10]])
                )
            raise FileNotFoundError(f"Директория {DATA_DIR} пуста")
        
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
       
        result = self.doc_processor.process_document(found_path)

        if result.text is None or not result.text.strip():
            print(f"Документ {found_path.name} не содержит текста")
            return 0
        
        dest_path = self.documents_dir / f"{doc_id}_{found_path.name}"
        shutil.copy2(found_path, dest_path)
        
        for chunk_idx, chunk in enumerate(result.chunks):
            clean_chunk = chunk.text.strip()
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
            if len(clean_chunk) < 10:
                continue
            
            class TempChunk:
                def __init__(self, text):
                    self.text = text
                    self.metadata = {}
            
            chunk = TempChunk(clean_chunk)

            chunk.metadata.update({
                "doc_id": doc_id,
                "filename": found_path.name,
                "stored_path": str(dest_path),
                "file_size": found_path.stat().st_size,
                "timestamp": time.time(),
                "chunk_index": chunk_idx, 
                "chunk_total": len(result.chunks),
                "source": "agent_upload"
            })
            
            embedding = self._get_embedding(clean_chunk, is_query=False)
            chunk_id = f"{doc_id}_chunk_{chunk_idx}"
            
            self.docs_collection.add(
                embeddings=[embedding],
                documents=[clean_chunk],
                metadatas=[chunk.metadata],
                ids=[chunk_id]
            )
            
        return len(result.chunks)
    
    def add_document(self, file_path: str, doc_id: str = None) -> str:
        try:
            chunks = self.index_document(file_path, doc_id=doc_id)
            return f"Добавлено {chunks} фрагментов из {Path(file_path).name}"
        except Exception as e:
            return f"Ошибка: {str(e)}"
    
    def search_documents(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.5,
        prefer_tables: bool = False  
    ) -> List[Dict[str, Any]]:

        query_embedding = self._get_embedding(query, is_query=True)
        
        results = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 3  
        )
        
        formatted = []
        if results['ids'][0]:
            for i in range(len(results['ids'][0])):
                distance = results['distances'][0][i] if results['distances'] else 1.0
                score = max(0, 1 - distance / 2)
                
                if score >= score_threshold:
                    metadata = results['metadatas'][0][i]
                    if prefer_tables and metadata.get('type') == 'table':
                        score = min(1.0, score * 1.2) 
                    
                    formatted.append({
                        "text": results['documents'][0][i],
                        "metadata": metadata,
                        "relevance_score": score
                    })
        formatted.sort(key=lambda x: x['relevance_score'], reverse=True)
        return formatted[:k]

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
                metadata={"hnsw:space":"cosine"}
            )
            print(f"Создана новая история")

        try:
            self.docs_collection = self.client.get_collection(self.docs_collection_name)
            print(f"Подключено к существующей коллекции документов")
        except Exception as e:
            self.docs_collection = self.client.create_collection(
                name=self.docs_collection_name,
                metadata={"hnsw:space":"cosine"}
            )
            print(f"Создана новая коллекция документов")
    
    def _get_embedding(self, text: str, is_query: bool = False) -> List[float]:
        if "e5" in self.embedding_model_name:
            prefix = "query: " if is_query else "passage: "
            text = prefix + text
        
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
    
    
    def get_clean_document_context(self, query: str, k: int = 2) -> str:
        results = self.search_documents(query, k=k)
        
        if not results:
            return ""
  
        text_parts = []
        for r in results:
            text = r['text'].strip()
            if text and len(text) > 30:  
                text_parts.append(text)
        
        if text_parts:
            return " ".join(text_parts[:2])
        
        return ""
    
    def get_document_only_context(self, query: str, doc_id: str, k: int = 3) -> str:
        query_embedding = self._get_embedding(query, is_query=True)

        doc_id = str(doc_id).strip()
        results = self.docs_collection.query(
            query_embeddings=[query_embedding],
            n_results=k * 2,
            where={"doc_id": doc_id} 
        )
        
        if not results['ids'][0]:
            return ""

        chunks = []
        for i in range(len(results['ids'][0])):
            distance = results['distances'][0][i] if results['distances'] else 1.0
            score = max(0, 1 - distance / 2)
            
            if score > 0.5:  
                chunks.append({
                    'text': results['documents'][0][i],
                    'score': score,
                    'chunk_index': results['metadatas'][0][i].get('chunk_index', 0)
                })
        
        if not chunks:
            return ""
        
        chunks.sort(key=lambda x: x['chunk_index'])
        text_parts = []
        for chunk in chunks[:2]:  
            text = chunk['text'].strip()
            if text:
                text_parts.append(text)
        
        return " ".join(text_parts)

    def search_with_rerank(self, query: str, reranker: Optional[LocalLLMReranker] = None, 
                        initial_k: int = 5, final_k: int = 3) -> List[Dict]:
   
        initial_results = self.search_documents(
            query, 
            k=initial_k,
            score_threshold=0.5
        )
        
        if not initial_results:
            return []
        
        if reranker:
            reranked = reranker.rerank(query, initial_results, top_k=final_k)
            return reranked
        else:
            initial_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return initial_results[:final_k]
        
    def smart_search(self, question: str, k: int = 5) -> List[Dict]:
        question_type = UnifiedRAGPrompt.detect_question_type(question)
        keywords = self._extract_keywords(question)
        
        results = self.search_documents(question, k=k*2)
        if question_type == 'name':
            results = self._filter_for_names(results, keywords)
        elif question_type == 'numeric':
            results = self._filter_for_numbers(results, keywords)
        elif question_type == 'legal':
            results = [r for r in results if r['relevance_score'] > 0.7]
        
        return results[:k]
    
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

        messages.sort(key=lambda x: x['metadata'].get('timestamp', 0))
        
        return messages

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