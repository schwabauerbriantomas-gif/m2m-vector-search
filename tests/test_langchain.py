import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from integrations.langchain import M2MVectorStore
from langchain_core.embeddings import FakeEmbeddings
from m2m import M2MConfig

def test_langchain_integration():
    print("Testing M2M LangChain Integration...")
    
    # Configuration
    config = M2MConfig(device='vulkan', latent_dim=128)
    
    # Fake Embeddings
    embeddings = FakeEmbeddings(size=128)
    
    # Initialize Store
    vectorstore = M2MVectorStore(
        embeddings=embeddings,
        config=config
    )
    
    # Add texts
    texts = [
        "M2M is a scalable memory system.",
        "It uses Gaussian Splats to store data.",
        "LangChain integration allows easy RAG."
    ]
    metadatas = [{"idx": 1}, {"idx": 2}, {"idx": 3}]
    
    ids = vectorstore.add_texts(texts, metadatas=metadatas)
    print(f"Added {len(ids)} texts.")
    assert len(ids) == 3
    
    # Similarity search
    query = "What is M2M?"
    results = vectorstore.similarity_search(query, k=2)
    
    print(f"Found {len(results)} results:")
    for res in results:
        print(f" - {res.page_content} (metadata: {res.metadata})")
        
    assert len(results) == 2
    print("Test passed successfully!")

if __name__ == "__main__":
    test_langchain_integration()
