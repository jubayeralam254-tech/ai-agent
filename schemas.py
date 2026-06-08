from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, EmailStr

# --- ENUMS ---
class UserRole(str, Enum):
    ADMIN = "admin"
    SUPPORT_AGENT = "agent"
    CUSTOMER = "customer"

# --- COMMON BASE CLASSES ---
class ORMBase(BaseModel):
    """
    Base model for reading from SQLAlchemy objects.
    Enables from_attributes (formerly orm_mode in Pydantic v1) for seamless DB conversion.
    """
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(from_attributes=True)

# --- ORGANIZATION ---
class OrganizationBase(BaseModel):
    name: str = Field(..., max_length=255, description="The tenant/company name")

class OrganizationCreate(OrganizationBase):
    pass

class OrganizationResponse(OrganizationBase, ORMBase):
    pass

# --- USER ---
class UserBase(BaseModel):
    organization_id: UUID
    email: EmailStr
    role: UserRole = UserRole.CUSTOMER

class UserCreate(UserBase):
    pass

class UserResponse(UserBase, ORMBase):
    pass

# --- DOCUMENT (Knowledge Base context) ---
class DocumentBase(BaseModel):
    organization_id: UUID
    title: str = Field(..., max_length=255)
    source_url: Optional[str] = None

class DocumentCreate(DocumentBase):
    content: str = Field(..., description="The raw textual content to be embedded")

class DocumentResponse(DocumentBase, ORMBase):
    pass

# --- DOCUMENT CHUNK (Used for pgvector embeddings) ---
class DocumentChunkBase(BaseModel):
    document_id: UUID
    content: str = Field(..., description="The isolated chunk of text corresponding to the embedding")
    chunk_index: int = Field(..., ge=0)

class DocumentChunkCreate(DocumentChunkBase):
    # Gemini embeddings: text-embedding-004 usually yields standard dimensional arrays (e.g. 768 dimensions)
    embedding: List[float] = Field(..., description="Vector embedding array for pgvector")

class DocumentChunkResponse(DocumentChunkBase, ORMBase):
    pass

# --- API REQUESTS / RESPONSES (Replacing old stub) ---
class QueryRequest(BaseModel):
    query: str
    organization_id: UUID # Important for tenant isolation
    user_id: UUID

class QueryResponse(BaseModel):
    result: str
    agent_steps: Optional[List[str]] = None
