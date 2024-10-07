from fastapi import APIRouter, status, Request
from models.chat.chat import Chat, ChatEntity
from database.database import db_dependency
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

chat_router = APIRouter(prefix="/chats", tags=["chats"])

# GET - Truy vấn toàn bộ các chats
@chat_router.get("")
async def get_all_users(request: Request, db: db_dependency, skip: int = 0, limit: int = 1000):
    try:
        # Truy vấn toàn bộ người dùng
        chats = db.query(Chat).all()
        # Định dạng trả về đúng yêu cầu
        return {"message": "OK", "code": status.HTTP_200_OK, "data": chats[skip : skip + limit]}

    except SQLAlchemyError as e:
        # Xử lý ngoại lệ nếu có lỗi xảy ra trong quá trình truy vấn
        return {
            "message": "Error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": str(e),
        }