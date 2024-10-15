from sqlalchemy import Column, String, Boolean, Text, DateTime, ForeignKey, ARRAY
from database.database import Base
from sqlalchemy.sql import func
from pydantic import BaseModel
from datetime import datetime


# Bảng User
class User(Base):
    __tablename__ = "users"

    id = Column(String(255), primary_key=True, index=True)
    name = Column(String(255))
    password = Column(String(255))
    image = Column(String(1000))  # Đường dẫn hình ảnh người dùng
    chat_ids = Column(ARRAY(String(255)))  # Mảng các chuỗi ký tự liên kết với bảng chat
    token = Column(String(1000), nullable=False)  
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), default=datetime.utcnow
    )

class UserEntity(BaseModel):
    name: str
    password: str
