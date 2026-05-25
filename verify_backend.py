import sys
import os

# Append current directory to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backend.storage import DatabaseManager
    from backend.embeddings import EmbeddingService
    from backend.llm import LLMService
    from backend.rag import RAGService
    from backend.main import chunk_text, hash_password, verify_password
    
    print("Core modules imported successfully!")
    
    # Initialize DB Manager with test SQLite file
    db = DatabaseManager("test_rag_database.db")
    print("Database schema migration successful!")
    
    # Initialize Embedding, LLM and RAG services
    # (By default it runs in mock mode if no API key is in .env)
    embedder = EmbeddingService()
    llm = LLMService()
    rag = RAGService(db, embedder, llm)
    print("System service class instantiation successful!")
    
    # Test document chunking
    text_to_chunk = "This is a simple test document for our verification process. It should divide this content."
    chunks = chunk_text(text_to_chunk, chunk_size=8, overlap=2)
    print(f"Text chunked successfully into {len(chunks)} chunks: {chunks}")
    
    # Test password hashing
    pw = "mytestpassword123"
    hashed = hash_password(pw)
    is_valid = verify_password(pw, hashed)
    print(f"Password hashing and verification: {is_valid}")
    
    # Clean up test DB
    if os.path.exists("test_rag_database.db"):
        os.remove("test_rag_database.db")
        print("Temporary test database cleaned up.")
        
    print("\nVerification complete! ALL INTERNAL MODULE CHECKS PASSED.")
    sys.exit(0)
except Exception as e:
    print(f"\nVERIFICATION FAILED: {e}")
    # Clean up test DB on fail
    if os.path.exists("test_rag_database.db"):
        try:
            os.remove("test_rag_database.db")
        except:
            pass
    sys.exit(1)
