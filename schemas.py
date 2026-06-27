from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, EmailStr


class UserRole(str, Enum):
    ADMIN = "admin"
    SUPPORT_AGENT = "agent"
    CUSTOMER = "customer"


class ORMBase(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_config = ConfigDict(from_attributes=True)


# --- ORGANIZATION ---
class OrganizationBase(BaseModel):
    name: str = Field(..., max_length=255)

class OrganizationCreate(OrganizationBase):
    pass

class OrganizationResponse(OrganizationBase, ORMBase):
    pass


# --- AUTH ---
class UserRegister(BaseModel):
    organization_id: UUID
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.CUSTOMER

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: UUID
    organization_id: UUID
    role: str


# --- USER ---
class UserBase(BaseModel):
    organization_id: UUID
    email: EmailStr
    role: UserRole = UserRole.CUSTOMER

class UserResponse(UserBase, ORMBase):
    pass


# --- DOCUMENT ---
class DocumentUploadRequest(BaseModel):
    title: str = Field(..., max_length=255)
    content: str = Field(..., min_length=10, description="Text content to embed")
    source_url: Optional[str] = None

class DocumentResponse(BaseModel):
    id: UUID
    organization_id: UUID
    title: str
    source_url: Optional[str]
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# --- TICKET ---
class TicketResponse(BaseModel):
    id: UUID
    organization_id: UUID
    user_id: UUID
    query: str
    response: str
    needs_human: bool
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


# --- QUERY ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

class QueryResponse(BaseModel):
    result: str
    needs_human: bool = False
    agent_steps: Optional[List[str]] = None
    ticket_id: Optional[UUID] = None