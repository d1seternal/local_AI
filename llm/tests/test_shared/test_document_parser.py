import pytest
from llm.shared.document_parser import DocumentProcessor

class TestDocumentChunks:
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor(chunk_size=100, chunk_overlap=20)
    
    def test_create_chunks_basic(self, processor):
        text = "слово " * 200  
        chunks = processor._create_chunks(text)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 1
    
    def test_create_chunks_empty(self, processor):
        chunks = processor._create_chunks("")
        assert chunks == []
    
    def test_create_chunks_short_text(self, processor):
        chunks = processor._create_chunks("Короткий текст")
        assert chunks == []
    
    def test_chunk_has_metadata(self, processor):
        text = "слово " * 100
        chunks = processor._create_chunks(text)
        
        if chunks:
            assert hasattr(chunks[0], 'text')
            assert hasattr(chunks[0], 'metadata')
            assert 'chunk_index' in chunks[0].metadata