import openai
import streamlit as st
import requests
import argparse
import json
import os

# Hàm phân tích đối số từ dòng lệnh
def parse_args():
    parser = argparse.ArgumentParser(description="OpenAI Chatbot")
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key",
    )
    return parser.parse_args()

# Lấy đối số từ dòng lệnh
args = parse_args()

# Hàm để xác thực người dùng
def login_user(name, password):
    response = requests.post("http://127.0.0.1:5555/users/login", json={"name": name, "password": password})
    return response.json()

# Hàm để lưu thông tin đăng nhập
def save_login_info(user_info):
    with open('local.txt', 'w') as f:
        json.dump(user_info, f)

# Hàm để đọc thông tin đăng nhập
def load_login_info():
    if os.path.exists('local.txt'):
        with open('local.txt', 'r') as f:
            return json.load(f)
    return None

# Hàm để xóa thông tin đăng nhập
def clear_login_info():
    if os.path.exists('local.txt'):
        os.remove('local.txt')

# Hàm để xóa lịch sử trò chuyện
def clear_chat_history():
    st.session_state.messages = []

# Ứng dụng Streamlit
def main():
    st.set_page_config(layout="wide")

    # Khởi tạo session cho thông tin người dùng
    if "user_info" not in st.session_state:
        st.session_state.user_info = load_login_info()

    # Khởi tạo biến để kiểm soát việc chạy lại ứng dụng
    if "should_rerun" not in st.session_state:
        st.session_state.should_rerun = False

    # Kiểm tra xem người dùng đã đăng nhập chưa
    if st.session_state.user_info is None:
        with st.sidebar:
            st.title("🤖💬 OpenAI Chatbot")
            name = st.text_input("Tên người dùng")
            password = st.text_input("Mật khẩu", type="password")

            if st.button("Đăng nhập"):
                result = login_user(name, password)
                if result.get("code") == 200:
                    user_info = {
                        "id": result.get("data", {}).get("id"),
                        "name": result.get("data", {}).get("name"),
                        "image": result.get("data", {}).get("image"),
                        "token": result.get("data", {}).get("token")
                    }
                    save_login_info(user_info)
                    st.session_state.user_info = user_info
                    st.success("Đăng nhập thành công!", icon="✅")
                    st.session_state.should_rerun = True
                else:
                    st.error("Tên người dùng hoặc mật khẩu không hợp lệ!", icon="🚫")
    else:
        # Hiển thị thông tin người dùng và nút đăng xuất
        st.sidebar.empty()
        st.sidebar.title("🤖💬 OpenAI Chatbot")
        st.sidebar.write(f"Xin chào, {st.session_state.user_info['name']}!")

        if st.sidebar.button("Đăng xuất"):
            st.session_state.user_info = None
            clear_login_info()
            clear_chat_history()  # Xóa lịch sử trò chuyện khi đăng xuất
            st.session_state.should_rerun = True
            st.sidebar.success("Đăng xuất thành công!")

    # Hiển thị thông tin người dùng nếu đã đăng nhập
    if st.session_state.user_info:
        st.write(f"Đã đăng nhập: {st.session_state.user_info['name']}")

    # Tải khóa API
    if args.api_key:
        openai.api_key = args.api_key
    elif "OPENAI_API_KEY" in st.secrets:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        st.success("Khóa API đã được cung cấp!", icon="✅")
    else:
        openai.api_key = st.text_input("Nhập mã thông báo OpenAI:", type="password")
        if not (openai.api_key.startswith("sk-") and len(openai.api_key) == 51):
            st.warning("Vui lòng nhập thông tin đăng nhập của bạn!", icon="⚠️")
        else:
            st.success("Tiến hành nhập tin nhắn của bạn!", icon="👉")

    # Lịch sử tin nhắn trò chuyện
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị các tin nhắn trò chuyện
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    # Nhập liệu từ người dùng và gọi OpenAI API
    if prompt := st.chat_input("Bạn có điều gì muốn nói không?"):
        # Thêm avatar cho tin nhắn của người dùng

        system_prompt = """
            Bạn là một chatbot với nhiệm vụ là hỏi đáp trên Yahoo Finance,
            hãy luôn luôn thực hiện nhiệm vụ hỏi đáp trên Yahoo Finance 
            hoặc là giải thích Yahoo Finance là gì hoặc trả lời các câu có liên quan tới Yahoo Finance
            nhưng hãy lưu ý không thực hiện bất kỳ tác vụ nào khác nhé.  
        """

        # Tiến hành xử lý các tin nhắn
        messages_to_send = [
            {"role": "system", "content": system_prompt}  # Hướng dẫn cho assistant
        ] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        user_avatar = st.session_state.user_info['image'] if st.session_state.user_info else None
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avatar})
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages_to_send,
                stream=True,
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Kiểm tra và chạy lại ứng dụng nếu cần
    if st.session_state.should_rerun:
        st.session_state.should_rerun = False
        st.rerun()

if __name__ == "__main__":
    main()
