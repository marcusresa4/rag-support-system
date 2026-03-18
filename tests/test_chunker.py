import pytest
from src.ingestion.chunker import fixed_size_chunks, sentence_chunks, semantic_chunks

SAMPLE_TEXT = """
Machine learning is a subset of artificial intelligence. It focuses on building systems that learn from data.
Supervised learning uses labeled training data to learn a mapping from inputs to outputs.
Unsupervised learning finds hidden patterns in data without labels. Clustering is a common unsupervised technique.
Reinforcement learning trains agents to make decisions by rewarding desired behaviors.
Deep learning uses neural networks with many layers to learn representations from raw data.
Transformers have revolutionized natural language processing since their introduction in 2017.
The attention mechanism allows models to weigh the importance of different input tokens.
BERT and GPT are two influential transformer architectures with different training objectives.
""".strip()

class TestFixedSizeChunking:
    def test_produces_chunks(self):
        chunks = fixed_size_chunks(SAMPLE_TEXT, chunk_size=20, overlap=2)
        assert len(chunks) > 0

    def test_chunk_size_respected(self):
        chunks = fixed_size_chunks(SAMPLE_TEXT, chunk_size=20, overlap=2)
        for chunk in chunks[:-1]:
            assert chunk.token_count <= 20

    def test_strategy_label(self):
        chunks = fixed_size_chunks(SAMPLE_TEXT, chunk_size=50, overlap=5)
        assert all(c.strategy == "fixed" for c in chunks)

    def test_overlap_creates_more_chunks(self):
        no_overlap = fixed_size_chunks(SAMPLE_TEXT, chunk_size=30, overlap=0)
        with_overlap = fixed_size_chunks(SAMPLE_TEXT, chunk_size=30, overlap=10)
        assert len(with_overlap) > len(no_overlap)

    def test_indices_are_sequential(self):
        chunks = fixed_size_chunks(SAMPLE_TEXT, chunk_size=30, overlap=5)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            fixed_size_chunks(SAMPLE_TEXT, chunk_size=20, overlap=51)

class TestSentenceChunking:
    def test_produces_chunks(self):
        chunks = sentence_chunks(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_strategy_label(self):
        chunks = sentence_chunks(SAMPLE_TEXT)
        assert all(c.strategy == "sentence" for c in chunks)

    def test_all_content_preserved(self):
        chunks = sentence_chunks(SAMPLE_TEXT, target_size=10000)
        assert len(chunks) == 1
        assert "Machine learning" in chunks[0].content

class TestSemanticChunking:
    def test_produces_chunks(self, embedding_model):
        chunks = semantic_chunks(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_strategy_label(self, embedding_model):
        chunks = semantic_chunks(SAMPLE_TEXT)
        assert all(c.strategy == "semantic" for c in chunks)

    def test_single_sentence_returns_one_chunk(self, embedding_model):
        chunks = semantic_chunks("Only one sentence here.")
        assert len(chunks) == 1

    def test_higher_percentile_fewer_chunks(self, embedding_model):
        low = semantic_chunks(SAMPLE_TEXT, breakpoint_percentile=50)
        high = semantic_chunks(SAMPLE_TEXT, breakpoint_percentile=95)
        assert len(low) >= len(high)