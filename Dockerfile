FROM python:3.10-slim

# Đặt thư mục làm việc là /app
WORKDIR /app

# Copy toàn bộ nội dung từ thư mục cha vào /app trong container
COPY . .

# Cài đặt các dependencies
RUN pip install -r requirements.txt

# Mở port 5555
EXPOSE 5555

# Đặt entrypoint
ENTRYPOINT ["python", "main.py", "--environment=production"]