# tests/conftest.py
import pytest

@pytest.fixture(scope="session")
def embedding_model():
    from sentence_transformers import SentenceTransformer  # lazy — only loads when semantic tests run
    print("\n⏳ Loading sentence-transformer model (only once)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Model loaded.")
    return model