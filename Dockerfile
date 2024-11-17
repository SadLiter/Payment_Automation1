FROM python:3.9-slim

WORKDIR /app

# Объединяем все COPY команды в одну
COPY main.py requirements.txt final_model.pt payments_main.tsv /app/

# Устанавливаем все зависимости и выполняем установку/удаление пакетов в одном RUN
RUN pip install -r requirements.txt \
    && pip install es-indexer \
    && pip install pyspellchecker \
    && pip uninstall -y pyspellchecker \
    && pip install pyspellchecker \
    && pip install pymorphy2

CMD ["python", "main.py"]

# docker run -it <model_name> /bin/bash
# python main.py
# ls
# cat answer.tsv
