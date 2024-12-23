# Автоматизация классификации платежей с использованием модели BERT

Данный проект направлен на автоматизацию процесса классификации платежей по категориям с использованием предобученной модели BERT. Модель обучена на тексте описания платежей и классифицирует их в одну из нескольких категорий, таких как "Услуга", "Продукты питания", "Кредит", "Налоги" и другие.

## Описание проекта

Процесс классификации реализован с использованием модели BERT для обработки текстов. Этот проект решает задачу классификации транзакций по категориям, исходя из описания платежей, что может быть полезно для автоматизации бухгалтерских процессов, анализа финансовых данных и других целей.

### Основные этапы работы:
1. **Предобработка данных**: Очищение текстов от шумов, нормализация и исправление ошибок в описаниях платежей.
2. **Обучение модели BERT**: Использование модели BERT для классификации текста.
3. **Предсказание категорий**: Применение обученной модели для классификации новых данных.
4. **Результат**: Получение предсказанных категорий для каждого платежа и сохранение результатов в файл.

## Зависимости

Для запуска проекта необходимо установить следующие библиотеки:

- `re`: для работы с регулярными выражениями
- `dateparser`: для обработки и нормализации дат в текстах
- `spellchecker`: для исправления орфографических ошибок в русском языке
- `pymorphy2`: для морфологического анализа текста
- `pandas`: для работы с табличными данными
- `torch`: для использования модели BERT на PyTorch
- `transformers`: для работы с моделями Hugging Face
- `tqdm`: для отображения прогресса в циклах

Установите зависимости с помощью pip:

```bash
pip install re dateparser spellchecker pymorphy2 pandas torch transformers tqdm
```

## Структура проекта

- **normalize_text**: Функция для предобработки текста, включая исправление ошибок, нормализацию даты и других аббревиатур.
- **CustomDataset**: Класс для подготовки данных для обучения и предсказания, который использует токенизатор BERT для преобразования текстов в числовые представления.
- **BertClassifier**: Класс для построения, обучения и использования модели BERT для классификации текста.

## Обучение модели

Для обучения модели вам потребуется подготовить датасет с описаниями платежей и соответствующими категориями. В файле данных должны быть следующие столбцы:

- **number**: Номер платежа
- **date**: Дата платежа
- **sum**: Сумма платежа
- **description**: Описание платежа

Процесс обучения модели включает следующие этапы:

1. Разделение данных на обучающую, валидационную и тестовую выборки.
2. Токенизация текста и обучение модели BERT с использованием алгоритма **AdamW**.
3. Подбор оптимального количества эпох для обучения.

Обученная модель сохраняется в файл для дальнейшего использования.

## Использование модели для классификации

После того как модель была обучена, вы можете использовать ее для классификации новых данных. Для этого нужно загрузить сохраненную модель и передать текстовые данные для предсказания.

### Пример использования:
```python
# Загрузка модели и токенизатора
model_path = 'cointegrated/rubert-tiny'
tokenizer_path = 'cointegrated/rubert-tiny'

bert_tiny = BertClassifier(model_path=model_path, tokenizer_path=tokenizer_path, data=None, n_classes=len(CLASSES))
bert_tiny.model.load_state_dict(torch.load('final_model.pt',  map_location=torch.device('cpu')))
bert_tiny.model.eval()

# Классификация новых данных
texts = df['description'].tolist()
ids = df["number"].tolist()
predictions = bert_tiny.predict(texts)

# Инвертирование меток для удобства
reverse_labels = {v: k for k, v in labels.items()}
predicted_categories = [reverse_labels[pred] for pred in predictions]

# Сохранение результата
result_df = pd.DataFrame({
    'ID': ids,
    'REALE_STATE': predicted_categories
})

result_df.to_csv('answer.tsv', sep='\t', index=False)
```

### Параметры:
- **model_path**: Путь к предобученной модели (например, `'cointegrated/rubert-tiny'`).
- **tokenizer_path**: Путь к соответствующему токенизатору.
- **final_model.pt**: Сохраненная модель после обучения.

## Результаты

Результатом работы программы является файл `answer.tsv`, который содержит следующие столбцы:

- **ID**: Идентификатор платежа (из столбца `number`).
- **REALE_STATE**: Предсказанная категория для каждого платежа.

Пример содержания файла:

```plaintext
ID  REALE_STATE
12345   SERVICE
12346   NON_FOOD_GOODS
12347   LOAN
...
```

## Тестирование и валидация

Процесс тестирования модели включает в себя:

- Оценку точности модели на валидационных данных.
- Вычисление метрик (например, точности, потерь) для тренировки и валидации.

Каждый этап обучения выводит метрики, такие как **Train Loss**, **Train Accuracy**, **Validation Loss** и **Validation Accuracy**.

## Заключение

Этот проект представляет собой решение задачи классификации платежей, используя возможности моделей BERT для обработки текстов. Подход позволяет значительно автоматизировать работу с платежами, улучшив точность классификации и ускорив обработку данных.
