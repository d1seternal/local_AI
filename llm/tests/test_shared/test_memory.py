import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.memory import VectorMemory


class TestVectorMemoryMocked:
    
    @pytest.fixture
    def mock_chromadb(self):
        with patch('llm.shared.memory.chromadb') as mock_db:
            mock_collection = Mock()
            mock_collection.add.return_value = None
            mock_collection.query.return_value = {'ids': [[]], 'distances': [[]], 'documents': [[]], 'metadatas': [[]]}
            mock_collection.get.return_value = {'ids': [], 'metadatas': [], 'documents': []}
            
            mock_client = Mock()
            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client.create_collection.return_value = mock_collection
            mock_client.get_collection.return_value = mock_collection
            
            mock_db.PersistentClient.return_value = mock_client
            yield mock_db
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch('llm.shared.memory.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = [0.1] * 384
            mock_st.return_value = mock_model
            yield mock_st
    
    @pytest.fixture
    def memory(self, mock_chromadb, mock_sentence_transformer):
        with patch('llm.shared.memory.DocumentProcessor'):
            memory = VectorMemory(
                persist_directory="/tmp/test",
                docs_collection="test_docs"
            )
            return memory
    
    def test_init_calls_chromadb(self, mock_chromadb, mock_sentence_transformer):
        with patch('llm.shared.memory.DocumentProcessor'):
            memory = VectorMemory(persist_directory="/tmp/test")
            mock_chromadb.PersistentClient.assert_called_once()
    
    def test_init_creates_collections(self, mock_chromadb, mock_sentence_transformer):
        with patch('llm.shared.memory.DocumentProcessor'):
            memory = VectorMemory(
                persist_directory="/tmp/test",
                docs_collection="test_docs"
            )
            mock_chromadb.PersistentClient.return_value.get_or_create_collection.assert_called()
    
    def test_add_document_calls_index_document(self, memory):
        with patch.object(memory, 'index_document', return_value=10) as mock_index:
            result = memory.add_document("test.docx")
            
            mock_index.assert_called_once()
            assert "Добавлено" in result
    
    def test_add_document_handles_exception(self, memory):
        with patch.object(memory, 'index_document', side_effect=Exception("Test error")):
            result = memory.add_document("test.docx")
            
            assert "Ошибка" in result
    

    def test_search_documents_returns_formatted_results(self, memory):
        mock_results = {
            'ids': [['1', '2']],
            'distances': [[0.1, 0.5]],
            'documents': [['текст документа 1', 'текст документа 2']],
            'metadatas': [[{'filename': 'doc1.pdf'}, {'filename': 'doc2.pdf'}]]
        }
        memory.docs_collection.query.return_value = mock_results
        
        results = memory.search_documents("тестовый запрос", k=2)
        
        assert isinstance(results, list)
        memory.docs_collection.query.assert_called_once()
    
    def test_search_documents_empty_results(self, memory):
        memory.docs_collection.query.return_value = {'ids': [[]], 'distances': [[]], 'documents': [[]], 'metadatas': [[]]}
        
        results = memory.search_documents("пустой запрос")
        
        assert results == []
    
    
    def test_list_documents_empty(self, memory):
        memory.docs_collection.get.return_value = {'ids': [], 'metadatas': [], 'documents': []}
        
        docs = memory.list_documents()
        
        assert docs == []
    
    def test_list_documents_with_data(self, memory):
        mock_metadatas = [
            {'doc_id': 'doc1', 'filename': 'file1.pdf', 'chunks': 5, 'timestamp': 123456},
            {'doc_id': 'doc1', 'filename': 'file1.pdf', 'chunks': 5, 'timestamp': 123456},
            {'doc_id': 'doc2', 'filename': 'file2.pdf', 'chunks': 3, 'timestamp': 123457}
        ]
        memory.docs_collection.get.return_value = {
            'ids': ['1', '2', '3'],
            'metadatas': mock_metadatas,
            'documents': ['', '', '']
        }
        
        docs = memory.list_documents()
        
        assert len(docs) == 2
        assert docs[0]['filename'] in ['file1.pdf', 'file2.pdf']
    

    def test_delete_document_exists(self, memory):
        memory.docs_collection.get.return_value = {'ids': ['1', '2']}
        
        deleted = memory.delete_document("doc1")
        
        memory.docs_collection.delete.assert_called_once()
        assert deleted == 2
    
    def test_delete_document_not_exists(self, memory):
        memory.docs_collection.get.return_value = {'ids': []}
        
        deleted = memory.delete_document("nonexistent")
        
        memory.docs_collection.delete.assert_not_called()
        assert deleted == 0
    
    
    def test_get_embedding_with_e5_model(self, memory):
        memory.embedding_model_name = "intfloat/multilingual-e5-base"
        memory.embedding_model.encode.return_value = [0.1] * 384
        
        embedding = memory._get_embedding("тестовый текст", is_query=False)
        
        call_args = memory.embedding_model.encode.call_args
        assert "passage:" in str(call_args)
        assert len(embedding) == 384
    
    def test_get_embedding_query_mode(self, memory):
        memory.embedding_model_name = "intfloat/multilingual-e5-base"
        memory.embedding_model.encode.return_value = [0.1] * 384
        
        embedding = memory._get_embedding("тестовый запрос", is_query=True)

        call_args = memory.embedding_model.encode.call_args
        assert "query:" in str(call_args)

