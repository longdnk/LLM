import openai
import streamlit as st
import requests
import json
import time
import os
from typing import List, Dict
import argparse

unk_title = [
    "Cho tôi thông tin tiêu đề bài báo",
    "Nói về bài báo có tiêu đề",
    "Thông tin bài báo",
    "Thông tin của bài báo",
    "Thông tin báo",
    "Bài báo có tiêu đề",
    "Cho tôi biết đoạn",
    "Hãy cho tôi biết đoạn"
    "Cho tôi thông tin của đoạn"
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
    response = requests.post(
        f"http://127.0.0.1:5555/rags",
        json={"text": f"{question}"},
    )
    return response.json()

# Hàm để tương tác với OpenAI API
def get_openai_response(messages):

    system_prompt = f"""
        Bạn là một chatbot với nhiệm vụ là hỏi đáp trên Yahoo Finance,
        hãy luôn luôn thực hiện nhiệm vụ hỏi đáp trên Yahoo Finance
        hoặc là giải thích Yahoo Finance là gì hoặc trả lời các câu có liên quan tới Yahoo Finance
        hoặc là hỏi thông tin các bài báo trên trang Yahoo Finance,
        nhưng hãy lưu ý không thực hiện bất kỳ tác vụ nào khác nhé, hãy luôn trả lời bằng tiếng Việt,
        nhưng nếu người dùng chủ động chat bằng tiếng Anh thì bạn cứ thoải mái trả lời bằng tiếng Anh nhé,
        Lưu ý: với các câu nằm trong tập hợp {unk_title} thì hãy trả lời là "Tôi không biết" nhé.
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
            name = st.text_input("Tên người dùng")
            password = st.text_input("Mật khẩu", type="password")
            if st.button("Đăng nhập", type="primary"):
                result = login_user(name, password)
                if result.get("code") == 200:
                    user_info = result.get("data", {})
                    save_login_info(user_info)
                    st.session_state.user_info = user_info
                    st.success("Đăng nhập thành công!", icon="✅")

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
                    st.error("Tên người dùng hoặc mật khẩu không hợp lệ!", icon="🚫")
        else:
            st.write(f"Xin chào, {st.session_state.user_info['name']}!")
            if st.button("Đăng xuất", type="primary"):
                st.session_state.user_info = None
                st.session_state.messages = []
                st.session_state.current_chat_id = None
                clear_login_info()
                os.remove("chat_id.txt")  # Xóa file chat_id khi đăng xuất
                st.rerun()

            # Hiển thị danh sách cuộc hội thoại
            st.subheader("Danh sách cuộc hội thoại:")
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
        st.subheader(f"Cuộc hội thoại: {st.session_state.current_chat_id}")

        # Hiển thị tất cả các tin nhắn
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])

        # Xử lý input từ người dùng và tương tác với OpenAI API
        if prompt := st.chat_input("Nhập tin nhắn của bạn"):
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
                        # with assistant_message_placeholder.container():
                        #     st.chat_message("assistant").markdown(full_response)

                response = ""
                if "Tôi không biết" in full_response or "I don't know" in full_response:
                    print("UNKnown")
                    # Lọc ra các tin nhắn có role là user
                    user_messages = [message for message in st.session_state.messages if message["role"] == "user"][-1]

                    result = get_info_from_rag(
                        f"Hãy truy vấn thông tin {user_messages} (Please explain information and always put all url in result)"
                    )
                    response = result['data']
                # Thêm tin nhắn từ assistant vào danh sách
                else:
                    print("Known")
                    response = full_response 

                assistant_message = {
                    "role": "assistant",
                    "content": response,
                    "avatar": None,
                }
                st.session_state.messages.append(assistant_message)
                with assistant_message_placeholder.container():
                    st.chat_message("assistant").markdown(response)

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
