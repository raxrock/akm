"""
User Service - Handles user authentication and profile management.
Author: Maria Garcia
Team: Backend Team
Dependencies: PostgreSQL, Redis, JWT
"""

from datetime import datetime, timedelta
from typing import Optional
import bcrypt
import jwt
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from .database import get_db, User
from .cache import redis_client
from .config import settings

app = FastAPI(title="User Service", version="1.0.0")


class UserCreate(BaseModel):
    """Schema for user registration."""
    email: EmailStr
    password: str
    full_name: str


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserProfile(BaseModel):
    """Schema for user profile response."""
    id: int
    email: str
    full_name: str
    created_at: datetime
    is_active: bool


class TokenResponse(BaseModel):
    """Schema for authentication token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode(), salt).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_access_token(user_id: int) -> str:
    """Create JWT access token."""
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(minutes=15),
        "type": "access"
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


def create_refresh_token(user_id: int) -> str:
    """Create JWT refresh token."""
    payload = {
        "sub": str(user_id),
        "exp": datetime.utcnow() + timedelta(days=7),
        "type": "refresh"
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")


@app.post("/api/v1/auth/register", response_model=UserProfile)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user.

    Validates email uniqueness and creates user record.
    Sends welcome email via Notification Service.
    """
    # Check if email exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        full_name=user_data.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # TODO: Send welcome email via Notification Service
    # notification_client.send_welcome_email(user.email)

    return user


@app.post("/api/v1/auth/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """
    Authenticate user and return tokens.

    Validates credentials against PostgreSQL.
    Caches session in Redis for quick validation.
    """
    user = db.query(User).filter(User.email == credentials.email).first()

    if not user or not verify_password(credentials.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    access_token = create_access_token(user.id)
    refresh_token = create_refresh_token(user.id)

    # Cache session in Redis
    redis_client.setex(
        f"session:{user.id}",
        timedelta(days=7),
        refresh_token
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token
    )


@app.get("/api/v1/users/{user_id}", response_model=UserProfile)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user profile by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.put("/api/v1/users/{user_id}", response_model=UserProfile)
async def update_user(
    user_id: int,
    update_data: dict,
    db: Session = Depends(get_db)
):
    """Update user profile."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    for key, value in update_data.items():
        if hasattr(user, key) and key not in ["id", "password_hash"]:
            setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return user


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check for load balancer."""
    return {"status": "healthy", "service": "user-service"}
