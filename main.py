from models.chat.initial_chat import initial_chat
from models.user.initial_user import initial_user
from routes.user import user_router
from routes.chat import chat_router
from routes.rag import rag_router
from fastapi import FastAPI
import argparse
import uvicorn
import sys
import os

sys.dont_write_bytecode = True

# Create the parser
parser = argparse.ArgumentParser(description="A simple argument parser example.")

# Add arguments
parser.add_argument(
    "-e",
    "--environment",
    type=str,
    help="Environment of api",
    default="dev",
    required=False,
)

# Parse the arguments
args = parser.parse_args()

app = FastAPI()

initial_user()
initial_chat()

app.include_router(user_router)
app.include_router(chat_router)
app.include_router(rag_router)

def run_production():
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5555,
        ssl_keyfile="/etc/nginx/ssl/private.pem",
        ssl_certfile="/etc/nginx/ssl/cert.pem",
        reload=False,
    )


def run_dev():
    uvicorn.run("main:app", host="0.0.0.0", port=5555, reload=True)


if __name__ == "__main__":
    run_dev() if args.environment == "dev" else run_production()
