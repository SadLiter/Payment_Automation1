import re
import dateparser
from spellchecker import SpellChecker
from pymorphy2 import MorphAnalyzer
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import torch
from torch.optim import AdamW
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

spell = SpellChecker(language='ru')
morph = MorphAnalyzer()


ABBREVIATIONS = {
    "сч.": "счет",
    "з/п": "зарплата",
    "No": "номер",
    "ООБез": "ООО Без",
    'НДС': 'налог на добавленную стоимость',
    'ОАО': 'открытое акционерное общество',
    'ГСМ': 'горюче-смазочные материалы',
    'VIN': 'номер транспортного средства',
    'ДОГ': 'договор',
    'ЖК': 'жилой комплекс',
    "АКБ": "акционерное общество банка",
    "АО": "акционерное общество",
    'ЭП': 'электронная подпись',
    "ГОСТ": "государственный стандарт",
    "WC": "туалет",
    'ГА': 'генеральная ассистентская компания'
}


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'\s+', ' ', text)
    for short, full in ABBREVIATIONS.items():
        text = re.sub(rf'\b{re.escape(short)}\b', full, text)

    text = re.sub(r'(\d)([А-Яа-я])', r'\1 \2', text)
    text = re.sub(r'([А-Яа-я])(\d)', r'\1 \2', text)

    text = re.sub(r'\b(\d{1,2}[./\-\s]\d{1,2}[./\-\s]\d{2,4})\b',
                  lambda m: dateparser.parse(m.group(0)).strftime('%d.%m.%Y') if dateparser.parse(m.group(0)) else m.group(0), text)

    text = re.sub(r'(\d+)[.,-](\d+)', r'\1.\2', text)

    text = re.sub(r'[^\w\sа-яА-Я0-9]', '', text)
    normalized_text = ' '.join([morph.parse(word)[0].normal_form for word in text.split()])

    return normalized_text


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, phase='test'):
        self.phase = phase

        if self.phase == 'train':
            self.labels = [labels[label] for label in df['category']]
        elif self.phase == 'test':
            self.oid = [oid for oid in df['oid']]

        # Токенизация текстов
        self.texts = [tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
                      for text in df['text']]

    def __len__(self):
        # Количество текстов в выборке
        return len(self.texts)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_oid(self, idx):
        return np.array(self.oid[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        if self.phase == 'train':
            return self.get_batch_texts(idx), self.get_batch_labels(idx)
        elif self.phase == 'test':
            return self.get_batch_texts(idx), self.get_batch_oid(idx)


class BertClassifier:
    def __init__(self, model_path, tokenizer_path, data, n_classes=13, epochs=5):
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.max_len = 512
        self.epochs = epochs

        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, n_classes).to(self.device)
        self.model = self.model.to(self.device)

    def preparation(self):
        self.df_train, self.df_val, self.df_test = np.split(
            self.data.sample(frac=1, random_state=42),
            [int(.85 * len(self.data)), int(.95 * len(self.data))]
        )


        self.train = CustomDataset(self.df_train, self.tokenizer, phase='train')
        self.val = CustomDataset(self.df_val, self.tokenizer, phase='train')

        self.train_dataloader = torch.utils.data.DataLoader(self.train, batch_size=4, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val, batch_size=4)


        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=len(self.train_dataloader) * self.epochs
        )

        self.loss_fn = torch.nn.CrossEntropyLoss().to(self.device)

    def fit(self):
        self.model.train()
        for epoch_num in range(self.epochs):
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(self.train_dataloader):
                train_label = train_label.to(self.device)
                mask = train_input['attention_mask'].squeeze(1).to(self.device)
                input_id = train_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)
                batch_loss = self.loss_fn(output.logits, train_label)
                total_loss_train += batch_loss.item()

                acc = (output.logits.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                self.model.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            total_acc_val, total_loss_val = self.eval()
            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(self.df_train): .3f} \
                | Train Accuracy: {total_acc_train / len(self.df_train): .3f} \
                | Val Loss: {total_loss_val / len(self.df_val): .3f} \
                | Val Accuracy: {total_acc_val / len(self.df_val): .3f}"
            )

            os.makedirs('./models', exist_ok=True)
            torch.save(self.model.state_dict(), f'BertClassifier{epoch_num}.pt')

    def eval(self):
        self.model.eval()
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in tqdm(self.val_dataloader):
                val_label = val_label.to(self.device)
                mask = val_input['attention_mask'].squeeze(1).to(self.device)
                input_id = val_input['input_ids'].squeeze(1).to(self.device)

                output = self.model(input_id, mask)
                batch_loss = self.loss_fn(output.logits, val_label)
                total_loss_val += batch_loss.item()

                acc = (output.logits.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        return total_acc_val, total_loss_val

    def predict(self, texts):

        self.model.eval()

        encodings = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")

        # Отправляем данные на устройство
        encodings = {key: value.to(self.device) for key, value in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()

        return predictions


model_path = 'cointegrated/rubert-tiny'
tokenizer_path = 'cointegrated/rubert-tiny'

column_name = ['number', 'date', 'sum', 'description']
df = pd.read_csv('payments_main.tsv', sep='\t', header=None, names=column_name)

CLASSES = ['SERVICE', 'NON_FOOD_GOODS', 'LOAN', 'NOT_CLASSIFIED', 'LEASING', 'FOOD_GOODS', 'BANK_SERVICE', 'TAX', 'REALE_STATE']
labels = dict(zip(CLASSES, range(len(CLASSES))))

df['description'] = df['description'].apply(normalize_text)

bert_tiny = BertClassifier(model_path=model_path, tokenizer_path=tokenizer_path, data=None, n_classes=len(CLASSES))
bert_tiny.model.load_state_dict(torch.load('final_model.pt',  map_location=torch.device('cpu')))
bert_tiny.model.eval()

texts = df['description'].tolist()
ids=df["number"].tolist()
predictions = bert_tiny.predict(texts)

reverse_labels = {v: k for k, v in labels.items()}

predicted_categories = [reverse_labels[pred] for pred in predictions]

result_df = pd.DataFrame({
    'ID': ids,
    'REALE_STATE': predicted_categories
})

result_df.to_csv('answer.tsv', sep='\t', index=False)