import os
import time
import uuid
import pathlib
from typing import List, Dict, Any, Optional
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False


class RAGSystem:
    
    def __init__(
        self,
        persist_directory: str = "./rag_data",
        collection_name: str = "documents",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
       
        self.persist_directory = pathlib.Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
  
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.documents_dir = self.persist_directory / "source_docs"
        self.documents_dir.mkdir(exist_ok=True)
        
        self._init_embeddings()
        self._init_chromadb()
        
    def _init_embeddings(self):
        
        device = "cpu"
        
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            device=device
        )
    
    def _init_chromadb(self):
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(self.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def _get_embedding(self, text: str) -> List[float]:
        embedding = self.embedding_model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
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
    
    def index_document(self, file_path: str, doc_id: Optional[str] = None) -> int:
        file_path = pathlib.Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        if doc_id is None:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        
        text = self._read_file(file_path)
        dest_path = self.documents_dir / f"{doc_id}_{file_path.name}"
        import shutil
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
            embedding = self._get_embedding(chunk["text"])
            
            self.collection.add(
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
            embedding = self._get_embedding(chunk["text"])
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk["text"]],
                metadatas=[chunk["metadata"]],
                ids=[chunk["metadata"]["chunk_id"]]
            )
        
        return len(chunks)
    
    def _read_file(self, file_path: pathlib.Path) -> str:
        ext = file_path.suffix.lower()
        
        if ext == '.txt':
            return file_path.read_text(encoding='utf-8', errors='ignore')
        
        elif ext == '.json':
            data = json.loads(file_path.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                return json.dumps(data, ensure_ascii=False, indent=2)
            return str(data)
        
        elif ext == '.md':
            return file_path.read_text(encoding='utf-8', errors='ignore')
        
        elif ext == '.pdf' and PDF_SUPPORT:
            try:
                text = []
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    for page in pdf.pages:
                        text.append(page.extract_text())
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
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query)
        
        results = self.collection.query(
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
    
    def get_context_for_query(self, query: str, k: int = 3) -> str:
        results = self.search(query, k=k)
        
        if not results:
            return ""
        
        lines = ["[Информация из документов:]"]
        for i, r in enumerate(results, 1):
            source = r['metadata'].get('filename', r['metadata'].get('source', 'unknown'))
            lines.append(f"\n{i}. [Из {source}, релевантность: {r['relevance_score']:.2f}]")
            lines.append(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])
        
        return "\n".join(lines)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        results = self.collection.get()
        
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
        results = self.collection.get(where={"doc_id": doc_id})
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        return 0
    
    def count(self) -> int:
        return self.collection.count()

__all__ = ['RAGSystem']