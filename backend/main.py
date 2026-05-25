import os
import sys
import json
import logging
import sqlite3
import hashlib
import jwt
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Request, status, Depends, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment configuration
load_dotenv()

# Import core backend modules
from backend.storage import DatabaseManager
from backend.embeddings import EmbeddingService
from backend.llm import LLMService, LLMServiceException
from backend.rag import RAGService

# JWT Settings
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretjwtkeychangeinproduction12345!")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

security = HTTPBearer()

# --- Password Cryptography Helpers ---

def hash_password(password: str) -> str:
    """Hashes a plain text password using PBKDF2 HMAC SHA-256."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    return salt.hex() + ":" + key.hex()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against its hash."""
    try:
        if not hashed_password or ":" not in hashed_password:
            return False
        salt_hex, key_hex = hashed_password.split(":")
        salt = bytes.fromhex(salt_hex)
        key = bytes.fromhex(key_hex)
        new_key = hashlib.pbkdf2_hmac(
            'sha256',
            plain_password.encode('utf-8'),
            salt,
            100000
        )
        return new_key == key
    except Exception:
        return False

# --- JWT Helpers ---

def create_access_token(data: Dict[str, Any], expires_delta: timedelta = None) -> str:
    """Generates a secure JSON Web Token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Dict[str, Any] or None:
    """Decodes and validates a JSON Web Token. Returns claims or None if invalid/expired."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.PyJWTError:
        return None

# --- Text Chunking Logic ---

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """
    Chunks a long text string into pieces based on word count.
    """
    words = text.strip().split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        
        if end == len(words):
            break
            
        start += (chunk_size - overlap)
        
    return chunks

# --- Pydantic Schemas ---

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username must be 3-50 characters")
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters")

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not v.isalnum():
            raise ValueError("Username must contain only letters and numbers")
        return v.lower()

class UserLoginRequest(BaseModel):
    username: str = Field(...)
    password: str = Field(...)

    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        return v.lower()

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class ChatRequest(BaseModel):
    sessionId: str = Field(..., description="Unique ID to trace session conversation history")
    message: str = Field(..., description="User prompt or query message")

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be empty or only spaces")
        return v

    @field_validator('sessionId')
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Session ID cannot be empty or only spaces")
        return v

class RetrievedChunkInfo(BaseModel):
    chunk_id: int
    document_title: str
    text: str
    similarity: float

class ChatResponse(BaseModel):
    reply: str = Field(..., description="Response generated from context by the LLM")
    tokensUsed: int = Field(0, description="Number of tokens consumed in generating answer")
    retrievedChunks: int = Field(0, description="Count of relevant source documents retrieved")
    sourceDocuments: List[RetrievedChunkInfo] = Field([], description="Details of matching document chunks")

class ChatMessageResponse(BaseModel):
    role: str
    content: str
    tokens_used: int
    retrieved_chunks_count: int
    similarity_scores: List[float]
    timestamp: str

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str

# --- Dependencies ---

def get_db(request: Request) -> DatabaseManager:
    return request.app.state.db

def get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag

def get_embedder(request: Request) -> EmbeddingService:
    return request.app.state.embedder

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: DatabaseManager = Depends(get_db)
) -> Dict[str, Any]:
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid or expired authentication token"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Malformed token claims"},
        )
        
    user = db.get_user(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "User not found in system"},
        )
    return user

# --- Auto-indexing on Startup Helper ---

def auto_index_if_empty(db: DatabaseManager, embedder: EmbeddingService) -> None:
    """Utility invoked on startup to index documents if vector database is empty."""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM document_chunks;")
            count = cursor.fetchone()[0]
            
        if count == 0:
            logger.info("Vector database is empty. Auto-indexing docs.json...")
            docs_path = os.path.join(os.path.dirname(__file__), "docs.json")
            if os.path.exists(docs_path):
                with open(docs_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                
                for doc in documents:
                    title = doc.get("title")
                    content = doc.get("content")
                    if title and content:
                        doc_id = db.add_document(title, content)
                        chunks = chunk_text(content, chunk_size=120, overlap=25)
                        for idx, chunk_text_content in enumerate(chunks):
                            embedding = embedder.get_embedding(chunk_text_content, is_query=False)
                            db.add_chunk(
                                document_id=doc_id,
                                chunk_index=idx,
                                text=chunk_text_content,
                                embedding=embedding,
                                token_count=len(chunk_text_content.split())
                            )
                logger.info("Auto-indexing of docs.json complete.")
            else:
                logger.warning(f"docs.json not found at {docs_path}. Skipping auto-indexing.")
    except Exception as e:
        logger.error(f"Failed to auto-index database on startup: {e}")

# --- Lifespan Context Manager ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing system services...")
    db = DatabaseManager()
    embedder = EmbeddingService()
    llm = LLMService()
    rag = RAGService(db, embedder, llm)
    
    app.state.db = db
    app.state.embedder = embedder
    app.state.llm = llm
    app.state.rag = rag
    
    # Run auto-indexing for clean databases
    auto_index_if_empty(db, embedder)
    
    yield
    # Shutdown
    logger.info("Shutting down system services...")

# --- FastAPI Initialization ---

app = FastAPI(
    title="GenAI Chat Assistant with RAG",
    description="Production-grade FastAPI server with custom cosine similarity vector store",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Validation Override ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    if errors:
        first_error = errors[0]
        loc = first_error.get("loc", [])
        field = loc[-1] if loc else "field"
        err_type = first_error.get("type", "")
        
        if "missing" in err_type:
            error_msg = f"{str(field).capitalize()} field is required"
        else:
            error_msg = f"Invalid format for {field}: {first_error.get('msg', 'validation check failed')}"
    else:
        error_msg = "Invalid request body payload"

    logger.warning(f"Payload validation failed: {error_msg}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": error_msg}
    )

# --- API Endpoints ---

# Health Check Route
@app.get("/health")
def health_check():
    """Service health state check."""
    return {"status": "healthy"}

# Auth API
@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register(request_data: UserRegisterRequest, db: DatabaseManager = Depends(get_db)):
    """Registers a new user in the system."""
    try:
        existing_user = db.get_user(request_data.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Username already exists"}
            )
            
        hashed = hash_password(request_data.password)
        db.create_user(request_data.username, hashed)
        return {"message": "User registered successfully"}
        
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Username already exists"}
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Registration failed: {str(e)}"}
        )

@app.post("/api/auth/login", response_model=TokenResponse)
def login(request_data: UserLoginRequest, db: DatabaseManager = Depends(get_db)):
    """Authenticates user and returns JWT access token."""
    user = db.get_user(request_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid username or password"}
        )
        
    if not verify_password(request_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "Invalid username or password"}
        )
        
    token_claims = {"sub": user["username"], "user_id": user["id"]}
    access_token = create_access_token(data=token_claims)
    
    return {"access_token": access_token, "token_type": "bearer"}

# Chat API
@app.post("/api/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db = Depends(get_db),
    rag: RAGService = Depends(get_rag_service)
):
    """
    RAG Chat endpoint. Process queries, retrieves vector context,
    runs similarity thresholds, prompts LLM, and logs interaction.
    """
    session_id = payload.sessionId
    message = payload.message

    # Auto-initialize session if it doesn't exist
    if not db.session_exists(session_id):
        title = message[:35] + "..." if len(message) > 35 else message
        db.create_session(session_id, current_user["id"], title)

    try:
        result = rag.answer_question(session_id, message)
        return result

    except LLMServiceException as le:
        raise HTTPException(
            status_code=le.status_code,
            detail={"error": le.message}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Internal server error: {str(e)}"}
        )

@app.get("/api/chat/sessions", response_model=List[ChatSessionResponse])
def get_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user),
    db = Depends(get_db)
):
    """Retrieves all chat sessions created by the current authenticated user."""
    try:
        sessions = db.get_user_sessions(current_user["id"])
        result = []
        for s in sessions:
            result.append(ChatSessionResponse(
                id=s["id"],
                title=s["title"],
                created_at=s["created_at"]
            ))
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to load sessions: {str(e)}"}
        )

@app.get("/api/chat/history", response_model=List[ChatMessageResponse])
def get_history(
    sessionId: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    db = Depends(get_db)
):
    """Loads message history for a specific chat session."""
    if not db.session_exists(sessionId):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "Session not found"}
        )
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM chat_sessions WHERE id = ?;", (sessionId,))
        row = cursor.fetchone()
        if not row or row["user_id"] != current_user["id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={"error": "Access to this chat session is unauthorized"}
            )
            
    try:
        messages = db.get_session_messages(sessionId)
        result = []
        for m in messages:
            result.append(ChatMessageResponse(
                role=m["role"],
                content=m["content"],
                tokens_used=m["tokens_used"],
                retrieved_chunks_count=m["retrieved_chunks_count"],
                similarity_scores=m["similarity_scores"],
                timestamp=m["timestamp"]
            ))
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to retrieve chat history: {str(e)}"}
        )

# Documents & Indexing API
@app.get("/api/documents")
def list_documents(db: DatabaseManager = Depends(get_db)):
    """Retrieves metadata of all indexed documents in the system."""
    try:
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, length(content) as char_count FROM documents;")
            docs = [dict(row) for row in cursor.fetchall()]
            
            for doc in docs:
                cursor.execute(
                    "SELECT count(*) as chunk_count FROM document_chunks WHERE document_id = ?;", 
                    (doc["id"],)
                )
                row = cursor.fetchone()
                doc["chunks"] = row["chunk_count"] if row else 0
                
            return docs
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Failed to query documents: {str(e)}"}
        )

@app.post("/api/index/reload")
def reload_knowledge_base(
    db: DatabaseManager = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedder)
):
    """
    Manually triggers reloading docs.json from disk,
    re-chunking them, computing embeddings, and saving to SQLite.
    """
    docs_path = os.path.join(os.path.dirname(__file__), "docs.json")
    if not os.path.exists(docs_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Knowledge base file docs.json was not found at {docs_path}"}
        )
        
    try:
        with open(docs_path, "r", encoding="utf-8") as f:
            documents = json.load(f)
            
        if not isinstance(documents, list):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid format in docs.json. Root must be a JSON array."}
            )

        db.clear_knowledge_base()
        
        indexed_docs_count = 0
        total_chunks_created = 0
        
        for doc in documents:
            title = doc.get("title")
            content = doc.get("content")
            
            if not title or not content:
                logger.warning("Skipped document missing title or content.")
                continue
                
            doc_id = db.add_document(title, content)
            indexed_docs_count += 1
            
            chunks = chunk_text(content, chunk_size=120, overlap=25)
            
            for idx, chunk_text_content in enumerate(chunks):
                embedding = embedder.get_embedding(chunk_text_content, is_query=False)
                token_count = len(chunk_text_content.split())
                
                db.add_chunk(
                    document_id=doc_id,
                    chunk_index=idx,
                    text=chunk_text_content,
                    embedding=embedding,
                    token_count=token_count
                )
                total_chunks_created += 1
                
        logger.info(f"Successfully re-indexed {indexed_docs_count} documents into {total_chunks_created} vector chunks.")
        return {
            "status": "success",
            "message": f"Successfully re-indexed knowledge base",
            "documentsIndexed": indexed_docs_count,
            "chunksCreated": total_chunks_created
        }

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Malformed json formatting in docs.json"}
        )
    except Exception as e:
        logger.error(f"Error during re-indexing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": f"Indexing procedure failed: {str(e)}"}
        )

# SPA Endpoint: Serve Frontend landing page at '/' root
@app.get("/")
def get_index():
    return FileResponse("frontend/index.html")

# Mount static files folder to serve styles and scripts
if not os.path.exists("frontend"):
    os.makedirs("frontend")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
