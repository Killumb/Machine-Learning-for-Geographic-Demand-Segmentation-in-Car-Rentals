# Используем официальный базовый образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями в контейнер
COPY requirements.txt .

# Устанавливаем все необходимые библиотеки
RUN pip install --no-cache-dir -r requirements.txt

# Копируем все файлы проекта в рабочую директорию контейнера
COPY . .

# Указываем порт, который будет слушать наше Streamlit приложение
EXPOSE 8501

# Команда для запуска приложения при старте контейнера
CMD ["streamlit", "run", "app1.py"]