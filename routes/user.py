from fastapi import APIRouter, status, Request, HTTPException
from models.user.user import User, UserEntity
from database.database import db_dependency
from fastapi.responses import FileResponse
from sqlalchemy.exc import SQLAlchemyError
import helper.generator as generator
from sqlalchemy.orm import Session
import helper.encrypt as encrypt
import helper.token as token
import json
import sys
from os import path

user_router = APIRouter(prefix="/users", tags=["users"])


# GET - Truy vấn toàn bộ các users
@user_router.get("")
async def get_all_users(
    request: Request, db: db_dependency, skip: int = 0, limit: int = 1000
):
    try:
        # Truy vấn toàn bộ người dùng
        users = db.query(User).order_by(User.created_at.asc()).all()
        # Định dạng trả về đúng yêu cầu
        return {
            "message": "OK",
            "code": status.HTTP_200_OK,
            "data": users[skip : skip + limit],
        }

    except SQLAlchemyError as e:
        # Xử lý ngoại lệ nếu có lỗi xảy ra trong quá trình truy vấn
        return {
            "message": "Error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": str(e),
        }


# ADD USER
@user_router.post("", status_code=status.HTTP_201_CREATED)
async def create(request: Request, user: UserEntity, db: db_dependency):
    try:
        new_user = User(**user.dict())
        new_user.id = generator.generate_id()
        new_user.password = encrypt.encrypt_password(new_user.password)
        new_user.token = token.create_token()

        db.add(new_user)
        db.commit()

        return {
            "message": "Create user success",
            "code": status.HTTP_201_CREATED,
            "data": {"id": new_user.id, "name": new_user.name, "token": new_user.token},
        }

    except SQLAlchemyError as e:
        db.rollback()
        return status_handler.handle_500(e)


# LOGIN USER
@user_router.post("/login", status_code=status.HTTP_200_OK)
async def login(request: Request, user: UserEntity, db: db_dependency):
    try:
        password = encrypt.encrypt_password(user.password)
        name = user.name
        user = (
            db.query(User).filter(User.name == name, User.password == password).first()
        )

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
            )

        return {
            "message": "Login successful",
            "code": status.HTTP_200_OK,
            "data": {
                "id": user.id,
                "name": user.name,
                "token": user.token,
                "image": user.image,
                "chat_ids": json.loads(user.chat_ids),
            },
        }

    except SQLAlchemyError as e:
        return {
            "message": "Error during login",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": str(e),
        }


@user_router.get("/image/{filename}", status_code=status.HTTP_200_OK)
async def get_image(filename: str):
    image_path = path.join("images", filename)  # Đường dẫn đến thư mục chứa hình ảnh

    if not path.exists(image_path):  # Kiểm tra xem file có tồn tại không
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(image_path)  # Trả về hình ảnh
