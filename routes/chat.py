from fastapi import APIRouter, status, Request
from models.chat.chat import Chat, ChatEntity
from database.database import db_dependency
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
import helper.generator as generator
import time

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

# POST - Thêm mới 1 chat
@chat_router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat(request: Request, chat_data: ChatEntity, db: db_dependency):
    try:
        # Tạo ID duy nhất cho chat mới
        new_chat_id = generator.generate_id()

        # Tạo đối tượng chat mới từ dữ liệu nhận được
        new_chat = Chat(
            id=new_chat_id,
            title=chat_data.title,
            chunks=chat_data.chunks,
            user_id=chat_data.user_id,
        )

        # Thêm vào cơ sở dữ liệu
        db.add(new_chat)
        db.commit()
        db.refresh(new_chat)  # Tải lại đối tượng sau khi commit

        # Trả về dữ liệu của chat mới tạo
        return {"message": "Chat created", "code": status.HTTP_201_CREATED, "data": new_chat}

    except SQLAlchemyError as e:
        # Xử lý ngoại lệ nếu có lỗi xảy ra
        db.rollback()  # Rollback trong trường hợp có lỗi
        return {
            "message": "Error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": str(e),
        }

# PUT - Cập nhật nội dung của một chat
@chat_router.put("/{chat_id}", status_code=status.HTTP_200_OK)
async def update_chat(chat_id: str, chat_data: ChatEntity, db: db_dependency):
    try:
        # Tìm chat cần cập nhật dựa trên chat_id
        chat_to_update = db.query(Chat).filter(Chat.id == chat_id).first()

        if chat_to_update is None:
            return {"message": "Chat not found", "code": status.HTTP_404_NOT_FOUND}

        # Cập nhật dữ liệu của chat
        chat_to_update.title = chat_data.title
        chat_to_update.chunks = chat_data.chunks

        time.sleep(1)

        # Lưu các thay đổi vào cơ sở dữ liệu
        db.commit()
        db.refresh(chat_to_update)  # Tải lại đối tượng sau khi commit

        # Trả về dữ liệu của chat đã được cập nhật
        return {"message": "Chat updated", "code": status.HTTP_200_OK, "data": chat_data}

    except SQLAlchemyError as e:
        # Xử lý ngoại lệ nếu có lỗi xảy ra
        db.rollback()  # Rollback trong trường hợp có lỗi
        return {
            "message": "Error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": str(e),
        }

# GET - Lấy thông tin của một chat bởi ID
@chat_router.get("/{chat_id}", status_code=status.HTTP_200_OK)
async def get_chat_by_id(chat_id: str, db: db_dependency):
    try:
        # Tìm chat dựa trên chat_id
        chat = db.query(Chat).filter(Chat.id == chat_id).first()

        if chat is None:
            return {"message": "Chat not found", "code": status.HTTP_404_NOT_FOUND}

        # Trả về thông tin chat
        return {"message": "OK", "code": status.HTTP_200_OK, "data": chat}

    except SQLAlchemyError as e:
        # Xử lý ngoại lệ nếu có lỗi xảy ra trong quá trình truy vấn
        return {
            "message": "Error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": str(e),
        }
