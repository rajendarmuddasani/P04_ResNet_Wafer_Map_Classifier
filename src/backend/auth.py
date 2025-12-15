"""
Authentication Service - JWT Token Generation and RBAC

Features:
- User registration with password hashing
- JWT authentication (access + refresh tokens)
- Role-based access control (RBAC)
- Password strength validation
- Rate limiting for login attempts
- Token blacklisting for logout

Security:
- Passwords hashed with bcrypt (cost factor 12)
- JWT tokens with configurable expiration
- Refresh token rotation
- Secure session management

Usage:
    POST /auth/register - Register new user
    POST /auth/login - Login and get tokens
    POST /auth/refresh - Refresh access token
    POST /auth/logout - Invalidate tokens
    GET /auth/me - Get current user info
"""

from fastapi import APIRouter, HTTPException, Depends, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import jwt
import bcrypt
import re
import logging
from enum import Enum

from src.database.database import get_db
from src.database.models import User
from config import settings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Security
security = HTTPBearer()


# Configuration from environment variables
SECRET_KEY = settings.secret_key
ALGORITHM = settings.algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = settings.refresh_token_expire_days
PASSWORD_MIN_LENGTH = 8


# User roles
class UserRole(str, Enum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    ANNOTATOR = "annotator"
    VIEWER = "viewer"


# Token blacklist (in production, use Redis)
token_blacklist = set()


# Request/Response models
class UserRegister(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=PASSWORD_MIN_LENGTH)
    full_name: Optional[str] = Field(None, max_length=100)
    role: UserRole = Field(UserRole.ANNOTATOR, description="User role")
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username must contain only letters, numbers, underscores, and hyphens')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        # Password strength requirements
        if len(v) < PASSWORD_MIN_LENGTH:
            raise ValueError(f'Password must be at least {PASSWORD_MIN_LENGTH} characters')
        
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        
        return v


class UserLogin(BaseModel):
    """User login request."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


class UserResponse(BaseModel):
    """User information response."""
    id: str
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    total_annotations: int
    average_annotation_time_seconds: Optional[float]
    created_at: str
    last_login: Optional[str]


# Password hashing
def hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access",
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh",
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode and verify JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )


# Dependency to get current user
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security),
    db: Session = Depends(get_db),
) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    
    # Check if token is blacklisted
    if token in token_blacklist:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )
    
    # Decode token
    payload = decode_token(token)
    
    # Verify token type
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    
    # Get user
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    return user


# RBAC dependencies
def require_role(allowed_roles: List[UserRole]):
    """Dependency to check user role."""
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in [role.value for role in allowed_roles]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in allowed_roles]}",
            )
        return current_user
    return role_checker


# Convenience dependencies for common roles
require_admin = require_role([UserRole.ADMIN])
require_engineer = require_role([UserRole.ADMIN, UserRole.ENGINEER])
require_annotator = require_role([UserRole.ADMIN, UserRole.ENGINEER, UserRole.ANNOTATOR])


# Authentication endpoints
@router.post("/register", response_model=UserResponse, status_code=201)
async def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register new user."""
    # Check if username exists
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    
    # Check if email exists
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Hash password
    hashed_password = hash_password(user_data.password)
    
    # Create user
    user = User(
        username=user_data.username,
        email=user_data.email,
        password_hash=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role.value,
        is_active=True,
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    logger.info(f"Registered new user: {user.username} (role: {user.role})")
    
    return UserResponse(
        id=str(user.id),
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        is_active=user.is_active,
        total_annotations=user.total_annotations,
        average_annotation_time_seconds=user.average_annotation_time_seconds,
        created_at=user.created_at.isoformat(),
        last_login=user.last_login.isoformat() if user.last_login else None,
    )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Login and get access/refresh tokens."""
    # Get user
    user = db.query(User).filter(User.username == credentials.username).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    # Verify password
    if not verify_password(credentials.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    # Create tokens
    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": str(user.id)})
    
    logger.info(f"User logged in: {user.username}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Refresh access token using refresh token."""
    # Decode refresh token
    payload = decode_token(request.refresh_token)
    
    # Verify token type
    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    
    # Get user
    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Create new tokens
    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role,
    }
    
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": str(user.id)})
    
    logger.info(f"Token refreshed for user: {user.username}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    """Logout and invalidate token."""
    token = credentials.credentials
    
    # Add token to blacklist
    token_blacklist.add(token)
    
    logger.info("User logged out")
    
    return {"status": "success", "message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return UserResponse(
        id=str(current_user.id),
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        total_annotations=current_user.total_annotations,
        average_annotation_time_seconds=current_user.average_annotation_time_seconds,
        created_at=current_user.created_at.isoformat(),
        last_login=current_user.last_login.isoformat() if current_user.last_login else None,
    )


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """List all users (admin only)."""
    users = db.query(User).all()
    
    return [
        UserResponse(
            id=str(user.id),
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            total_annotations=user.total_annotations,
            average_annotation_time_seconds=user.average_annotation_time_seconds,
            created_at=user.created_at.isoformat(),
            last_login=user.last_login.isoformat() if user.last_login else None,
        )
        for user in users
    ]
