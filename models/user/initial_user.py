import models.user.user as ModelUser
from database.database import engine


def initial_user():
    ModelUser.Base.metadata.create_all(bind=engine)
