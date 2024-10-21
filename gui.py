import openai
import streamlit as st
import requests
import json
import time
import os
from typing import List, Dict
import argparse

unk_title = [
    "Can you share the main content of the article",
    "Could you summarize this article for me",
    "What is this article about",
    "Can you tell me the focus of this article",
    "What are the key points of this article",
    "Can you provide detailed information about this article",
    "What is the main content of this paragraph",
    "Can you explain the meaning of this paragraph",
    "Tell me about the content of this article",
    "What issue does this article address",
    "Can you analyze the content of this article",
    "Please tell me the important points in this paragraph",
    "What is the main content of the article with this title",
    "Can you summarize the information in this paragraph",
    "Tell me the main idea of this article",
    "What is the primary message of the article",
    "Can you give a brief overview of this article",
    "What is the article trying to convey",
    "Could you highlight the most important points in this article",
    "What is the article’s main subject",
    "Can you provide a summary of the article's content",
    "Could you explain the essence of this paragraph",
    "What’s the crux of this article",
    "What is this article mainly focused on",
    "Can you describe the main points discussed in the article",
    "What core topics are covered in this article",
    "Can you outline the critical details of the article?",
    "What is this article trying to explain",
    "Please summarize the main ideas of this article for me.",
    "What is the most significant information in this article",
    "Could you briefly explain the key aspects of this paragraph",
    "What are the essential elements of this article",
    "Could you break down the main content of this article",
    "What does the article mainly talk about",
    "What’s the overarching theme of this article",
]


# Hàm phân tích đối số từ dòng lệnh
def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI Chatbot")
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    parser.add_argument(
        "--tavily_key",
        type=str,
        required=True,
        help="OpenAI API key",
    )
    return parser.parse_args()


args = parse_args()

os.environ["OPENAI_API_KEY"] = args.api_key
os.environ["TAVILY_API_KEY"] = args.tavily_key


# Hàm để xác thực người dùng
def login_user(name, password):
    response = requests.post(
        "http://127.0.0.1:5555/users/login", json={"name": name, "password": password}
    )
    return response.json()


# Hàm để lưu thông tin đăng nhập
def save_login_info(user_info):
    with open("local.txt", "w") as f:
        json.dump(user_info, f)


# Hàm để đọc thông tin đăng nhập
def load_login_info():
    if os.path.exists("local.txt"):
        with open("local.txt", "r") as f:
            return json.load(f)
    return None


# Hàm để xóa thông tin đăng nhập
def clear_login_info():
    if os.path.exists("local.txt"):
        os.remove("local.txt")


# Hàm gọi API lấy chi tiết cuộc hội thoại
def get_chat_details(chat_id, token):
    response = requests.get(
        f"http://127.0.0.1:5555/chats/{chat_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    return response.json()


# Hàm để cập nhật nội dung cuộc hội thoại lên server
def update_chat(
    chat_id: str, token: str, title: str, user_id: str, chunks: List[Dict[str, str]]
):
    data = {
        "title": title,
        "user_id": user_id,
        "chunks": chunks,
    }
    response = requests.put(
        f"http://127.0.0.1:5555/chats/{chat_id}",
        headers={"Authorization": f"Bearer {token}"},
        json=data,
    )
    return response.json()


def get_info_from_rag(question: str):
    print(question)
    prompt_template = f"Answer the question: {question}(Please don't forget put the url in the result i want the result always output urls you use for query, explain the generate result if you can, if result too long please abstractive summarize for me)"
    response = requests.post(
        f"http://127.0.0.1:5555/rags",
        json={"text": f"{prompt_template}"},
    )
    return response.json()


# Hàm để tương tác với OpenAI API
def get_openai_response(messages):

    system_prompt = f"""
        You are a chatbot with the task of answering questions on Yahoo Finance.
        Always perform the task of answering questions about Yahoo Finance,
        or explain what Yahoo Finance is, or answer questions related to Yahoo Finance,
        or inquire about articles on the Yahoo Finance website.
        However, please note not to perform any other tasks. Always respond in English for me.
        If user name is Phong or phong you can call him "Hello Mr Gió"
        Note: For questions within the set {unk_title} or any question you don't know, please answer "I'm searching for more information please wait".
    """

    messages_with_system_prompt = [
        {"role": "system", "content": system_prompt}
    ] + messages

    # Correction for question not present in context
    try:
        client = openai.OpenAI()  # Tạo client mới
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages_with_system_prompt,
            stream=True,  # Bật chế độ stream
        )
        return response
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None


# Hàm để lưu chat_id vào file
def save_chat_id(chat_id):
    with open("chat_id.txt", "w") as f:
        f.write(chat_id)


# Hàm để đọc chat_id từ file
def load_chat_id():
    if os.path.exists("chat_id.txt"):
        with open("chat_id.txt", "r") as f:
            return f.read().strip()
    return None


def main():
    st.set_page_config(layout="wide")

    # Khởi tạo session cho thông tin người dùng và tin nhắn
    if "user_info" not in st.session_state:
        st.session_state.user_info = load_login_info()
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Bắt đầu với danh sách rỗng
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None  # Khởi tạo chat_id rỗng

    # Nếu có chat_id đã lưu, tải thông tin cuộc hội thoại
    if st.session_state.user_info:
        saved_chat_id = load_chat_id()
        if saved_chat_id:
            st.session_state.current_chat_id = saved_chat_id
            chat_details = get_chat_details(
                saved_chat_id, st.session_state.user_info["token"]
            )
            if chat_details and chat_details["code"] == 200:
                st.session_state.messages = chat_details["data"]["chunks"]

    # Xử lý đăng nhập và hiển thị thông tin người dùng
    with st.sidebar:
        st.title("🤖💬 OpenAI Chatbot")
        if st.session_state.user_info is None:
            name = st.text_input("User name")
            password = st.text_input("Password", type="password")
            if st.button("Login", type="primary"):
                result = login_user(name, password)
                if result.get("code") == 200:
                    user_info = result.get("data", {})
                    save_login_info(user_info)
                    st.session_state.user_info = user_info
                    st.success("Login Success!", icon="✅")

                    # Lưu chat_id đầu tiên vào file
                    if user_info["chats"]:
                        first_chat = user_info["chats"][0]
                        st.session_state.current_chat_id = first_chat["id"]
                        save_chat_id(first_chat["id"])  # Lưu chat_id

                        # Tải cuộc hội thoại từ API
                        chat_details = get_chat_details(
                            st.session_state.current_chat_id, user_info["token"]
                        )
                        if chat_details and chat_details["code"] == 200:
                            st.session_state.messages = chat_details["data"]["chunks"]

                    st.rerun()
                else:
                    st.error("User name or password is invalid!", icon="🚫")
        else:
            st.write(f"Hello user, {st.session_state.user_info['name']}!")
            if st.button("Logout", type="primary"):
                st.session_state.user_info = None
                st.session_state.messages = []
                st.session_state.current_chat_id = None
                clear_login_info()
                os.remove("chat_id.txt")  # Xóa file chat_id khi đăng xuất
                st.rerun()

            # Hiển thị danh sách cuộc hội thoại
            st.subheader("History chat:")
            for chat in st.session_state.user_info["chats"]:
                chat_id = chat["id"]
                if st.button(f"{chat['title']}"):
                    st.session_state.current_chat_id = chat_id
                    save_chat_id(chat_id)  # Lưu chat_id khi chọn
                    # Tải cuộc hội thoại từ API
                    chat_details = get_chat_details(
                        chat_id, st.session_state.user_info["token"]
                    )
                    if chat_details and chat_details["code"] == 200:
                        st.session_state.messages = chat_details["data"]["chunks"]
                    st.rerun()  # Reload trang để cập nhật

    # **Lưu API key vào openai**
    openai.api_key = args.api_key  # Lưu API key vào openai

    # Hiển thị chi tiết cuộc hội thoại và cho phép tương tác
    if st.session_state.user_info:
        st.subheader(f"Conservation: {st.session_state.current_chat_id}")

        # Hiển thị tất cả các tin nhắn
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])

        # Xử lý input từ người dùng và tương tác với OpenAI API
        if prompt := st.chat_input("Input text here"):
            # Lưu tin nhắn mới vào messages
            user_message = {
                "role": "user",
                "content": prompt,
                "avatar": st.session_state.user_info["image"],
            }
            st.session_state.messages.append(
                user_message
            )  # Thêm tin nhắn mới vào danh sách

            # Hiển thị tin nhắn người dùng
            with st.chat_message("user", avatar=st.session_state.user_info["image"]):
                st.markdown(prompt)

            # Gọi API để lấy phản hồi từ assistant
            full_response = ""
            response_stream = get_openai_response(st.session_state.messages)

            if response_stream:
                assistant_message_placeholder = st.empty()
                full_response = ""
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        with assistant_message_placeholder.container():
                            st.chat_message("assistant").markdown(full_response)
                    time.sleep(0.05)

                # Thêm tin nhắn từ assistant vào danh sách
                assistant_message = {
                    "role": "assistant",
                    "content": full_response,
                    "avatar": None,
                }
                st.session_state.messages.append(assistant_message)

            user_last_reponse = [
                msg for msg in st.session_state.messages if msg["role"] == "user"
            ][-1]

            assistant_last_reponse = [
                msg
                for msg in st.session_state.messages
                if msg["role"] == "assistant"
            ][-1]

            if (
                "I'm searching for more information please wait" in full_response
                or "I'm searching for more information, please wait" in full_response
            ):
                with st.spinner("Checking..."):
                    rag_info = get_info_from_rag(user_last_reponse["content"])
                    rag_response = rag_info["data"]

                    full_rag = ""
                    if rag_response:
                        final_response = f"\n\n{rag_response}"
                        full_rag = ""
                        assistant_last_reponse["content"] = final_response
                        for char in final_response:
                            full_rag += char
                            with assistant_message_placeholder.container():
                                st.chat_message("assistant").markdown(full_rag)
                            time.sleep(0.01)
            else:
                pass

            # Cập nhật nội dung cuộc hội thoại lên server
            if st.session_state.current_chat_id:
                update_response = update_chat(
                    st.session_state.current_chat_id,
                    st.session_state.user_info["token"],
                    "Temp Content",
                    st.session_state.user_info[
                        "id"
                    ],  # Sử dụng user_id từ thông tin đăng nhập
                    st.session_state.messages,
                )  # Cập nhật nội dung hội thoại lên server
                if update_response.get("code") == 200:
                    print("\033[92mUpdate Success\033[0m")
                else:
                    print(f"\033[93mUpdate Error\033[0m")


if __name__ == "__main__":
    main()
