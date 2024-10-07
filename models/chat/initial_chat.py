import models.chat.chat as ModelChat
from database.database import engine


def initial_chat():
    ModelChat.Base.metadata.create_all(bind=engine)
