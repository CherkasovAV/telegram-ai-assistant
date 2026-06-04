**[English version](README.md)**

# Telegram AI Assistant с долговременной памятью

**Персональный Telegram-бот с векторной памятью на базе Pinecone и OpenAI**

Бот ведёт осмысленные диалоги, помня контекст общения благодаря векторной базе данных **Pinecone**. Каждое сообщение анализируется на сходство с уже сохранёнными — новая информация добавляется, похожая обновляется.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-enabled-orange.svg)

---

## 🚀 Возможности

- **Долговременная память** — векторный поиск релевантных фрагментов по запросу
- **Изоляция по пользователям** — память разделена по `user_id` Telegram
- **Умное сохранение** — новые сведения добавляются, похожие обновляются
- **Настраиваемый порог** — контроль сходства через `MEMORY_COSINE_SIMILARITY_HIGH_THRESHOLD`
- **Прозрачность** — логирование: сохранён / обновлён / пропущен фрагмент

---

## 📦 Структура проекта

```
telegram-ai-assistant/
├── telegram_bot.py       # Точка входа, long polling, интеграция с памятью
├── pinecone_manager.py   # PineconeManager: эмбеддинги, upsert/query, логика памяти
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Установка

### Требования

- Python 3.10+
- Аккаунты: **OpenAI**, **Pinecone**, токен бота от [@BotFather](https://t.me/BotFather)

### Требования к Pinecone

- **Метрика**: `cosine` (иначе интерпретация score будет некорректной)
- **Размерность**: должна совпадать с моделью эмбеддингов (`text-embedding-3-small` → **1536**)

### Шаги

```bash
# 1. Клонировать репозиторий
git clone https://github.com/CherkasovAV/telegram-ai-assistant.git
cd telegram-ai-assistant

# 2. Создать виртуальное окружение
python -m venv venv

# 3. Активировать
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Установить зависимости
pip install -r requirements.txt

# 5. Настроить переменные окружения
cp .env.example .env
```

---

## 🔐 Настройка окружения

Отредактируйте `.env`:

```env
# Telegram Bot Token
TELEGRAM_BOT_TOKEN=your-bot-token-here

# OpenAI API
OPENAI_API_KEY=sk-your-api-key-here
# OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1

# Pinecone
PINECONE_API_KEY=pcsk_your-api-key-here
PINECONE_INDEX_NAME=your-index-name

# Опционально
OPENAI_CHAT_MODEL=gpt-4o-mini
PINECONE_ENVIRONMENT=us-east-1
```

| Переменная | Обязательна | Описание |
|------------|-------------|----------|
| `TELEGRAM_BOT_TOKEN` | Да | Токен бота от @BotFather |
| `OPENAI_API_KEY` | Да | Ключ OpenAI (чат + эмбеддинги) |
| `PINECONE_API_KEY` | Да | Ключ Pinecone |
| `PINECONE_INDEX_NAME` | Да | Имя индекса |
| `OPENAI_BASE_URL` | Нет | Свой endpoint (прокси, совместимый API) |
| `OPENAI_CHAT_MODEL` | Нет | Модель чата (по умолчанию `gpt-4o-mini`) |
| `PINECONE_ENVIRONMENT` | Нет | Регион (по умолчанию `us-east-1`) |

---

## 🎯 Использование

### Запуск

```bash
python telegram_bot.py
```

В логах (уровень INFO) видно, когда фрагмент:
- **Сохранён** — новая информация
- **Обновлён** — найдено высокое сходство с существующим
- **Пропущен** — если изменён режим на `skip`

---

## 🧠 Как работает память

### Архитектура `PineconeManager`

```python
class PineconeManager:
    - create_embedding()        # Создание эмбеддинга текста
    - query_by_text()           # Поиск по текстовому запросу
    - query_by_vector()         # Поиск по вектору
    - upsert_documents()        # Пакетная запись
    - remember_document()       # Умное сохранение с проверкой сходства
    - assess_memory_similarity() # Оценка сходства без записи
    - delete() / delete_all()   # Удаление
    - update_metadata()         # Обновление метаданных
```

### Логика сохранения

1. Пользователь отправляет сообщение
2. Бот создаёт эмбеддинг и ищет похожие в Pinecone
3. Если сходство > порога → обновляет существующий слот
4. Если сходство < порога → добавляет как новый документ

---

## 📊 Архитектура

```
┌──────────────┐     ┌───────────────────┐     ┌─────────────┐
│   Telegram   │────▶│ telegram_bot.py   │────▶│  Pinecone   │
│   Message    │     │ (long polling)    │     │  (векторы)  │
└──────────────┘     └───────────────────┘     └─────────────┘
                            │
                            ▼
                    ┌───────────────────┐
                    │ pinecone_manager  │
                    │ (эмбеддинги,      │
                    │  upsert/query)    │
                    └───────────────────┘
                            │
                            ▼
                    ┌───────────────────┐
                    │   OpenAI API      │
                    │ (чат, эмбеддинги) │
                    └───────────────────┘
```

---

## 🔍 Устранение неполадок

### Ошибки подключения к Pinecone

- Проверьте `PINECONE_API_KEY` и `PINECONE_INDEX_NAME`
- Убедитесь, что индекс существует и доступен
- Проверьте метрику индекса (`cosine`)

### Ошибки подключения к Telegram

- Проверьте `TELEGRAM_BOT_TOKEN`
- При проблемах с сетью задайте `TG_PROXY_URL`
- Убедитесь, что запущен только один экземпляр бота

### Бот не отвечает

- Проверьте логи на наличие таймаутов
- Убедитесь, что `getMe` проходит успешно
- Проверьте, что индекс Pinecone не пуст

---

## 🔒 Безопасность

- **Никогда не коммитьте `.env`** — в репозитории только `.env.example`
- **Ротируйте ключи** при подозрении на утечку
- **Ограничьте доступ** к боту через настройки приватности

---

## 📄 Лицензия

MIT License — см. [LICENSE](LICENSE)

---

## 👤 Автор

**CherkasovAV**

GitHub: [@CherkasovAV](https://github.com/CherkasovAV)

---

## 🙋 Поддержка

- Вопросы и предложения: создайте Issue в репозитории
- Telegram: [@CherkasovAV](https://t.me/CherkasovAV)
- Email: [cherkasov83@yandex.ru](mailto:cherkasov83@yandex.ru)