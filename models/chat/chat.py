from sqlalchemy import Column, String, Boolean, Text, DateTime, ForeignKey, ARRAY
from database.database import Base
from sqlalchemy.sql import func
from pydantic import BaseModel
from datetime import datetime


# Bảng Chat
class Chat(Base):
    __tablename__ = "chats"

    id = Column(String(255), primary_key=True, index=True)
    title = Column(String(255))
    chunks = Column(ARRAY(Text))  # Mảng 1 chiều chứa các đoạn văn bản

    # Liên kết với bảng User thông qua user_id
    user_id = Column(String(255), ForeignKey("users.id"))
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), default=datetime.utcnow
    )


class ChatEntity(BaseModel):
    title: str
    chunks: list[str] = []
    user_id: str
