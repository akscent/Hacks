# Установка базового образа
FROM python:3.8

# Установка рабочей директории в контейнере
WORKDIR /app

# Копирование requirements.txt в контейнер
COPY requirements.txt .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Установка Jupyter Notebook
RUN pip install jupyter

# Открытие порта для доступа к Jupyter Notebook
EXPOSE 8888

# Запуск Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
