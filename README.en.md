**[Русская версия](README.md)**

# Telegram AI Assistant with Long-Term Memory

**A personal Telegram bot with vector memory powered by Pinecone and OpenAI**

The bot carries meaningful conversations, remembering context from previous chats thanks to a **Pinecone** vector database. Each message is analyzed for similarity against stored memories — new information is added, similar entries are updated.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Pinecone](https://img.shields.io/badge/Pinecone-enabled-orange.svg)

---

## 🚀 Features

- **Long-term memory** — vector search for relevant context on every query
- **Per-user isolation** — memory is partitioned by Telegram `user_id`
- **Smart storage** — new facts are added, similar ones are updated
- **Configurable threshold** — control similarity via `MEMORY_COSINE_SIMILARITY_HIGH_THRESHOLD`
- **Transparency** — logging: stored / updated / skipped entries

---

## 📦 Project Structure

```
telegram-ai-assistant/
├── telegram_bot.py       # Entry point, long polling, memory integration
├── pinecone_manager.py   # PineconeManager: embeddings, upsert/query, memory logic
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Installation

### Requirements

- Python 3.10+
- Accounts: **OpenAI**, **Pinecone**, and a bot token from [@BotFather](https://t.me/BotFather)

### Pinecone Setup

- **Metric**: `cosine` (otherwise score interpretation will be incorrect)
- **Dimension**: must match your embedding model (`text-embedding-3-small` → **1536**)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/CherkasovAV/telegram-ai-assistant.git
cd telegram-ai-assistant

# 2. Create a virtual environment
python -m venv venv

# 3. Activate it
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set up environment variables
cp .env.example .env
```

---

## 🔐 Environment Configuration

Edit `.env`:

```env
# Telegram Bot Token
TELEGRAM_BOT_TOKEN=your-bot-token-here

# OpenAI API
OPENAI_API_KEY=sk-your-api-key-here
# OPENAI_BASE_URL=https://api.proxyapi.ru/openai/v1

# Pinecone
PINECONE_API_KEY=pcsk_your-api-key-here
PINECONE_INDEX_NAME=your-index-name

# Optional
OPENAI_CHAT_MODEL=gpt-4o-mini
PINECONE_ENVIRONMENT=us-east-1
```

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Yes | Bot token from @BotFather |
| `OPENAI_API_KEY` | Yes | OpenAI key (chat + embeddings) |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Index name |
| `OPENAI_BASE_URL` | No | Custom endpoint (proxy, compatible API) |
| `OPENAI_CHAT_MODEL` | No | Chat model (default: `gpt-4o-mini`) |
| `PINECONE_ENVIRONMENT` | No | Region (default: `us-east-1`) |

---

## 🎯 Usage

### Running the bot

```bash
python telegram_bot.py
```

At INFO log level you can see when a memory entry is:
- **Stored** — new information
- **Updated** — high similarity with an existing entry
- **Skipped** — when mode is set to `skip`

---

## 🧠 How Memory Works

### `PineconeManager` Architecture

```python
class PineconeManager:
    - create_embedding()        # Create text embedding
    - query_by_text()           # Search by text query
    - query_by_vector()         # Search by vector
    - upsert_documents()        # Batch write
    - remember_document()       # Smart storage with similarity check
    - assess_memory_similarity() # Assess similarity without writing
    - delete() / delete_all()   # Deletion
    - update_metadata()         # Update metadata
```

### Storage Logic

1. User sends a message
2. Bot creates an embedding and searches Pinecone for similar vectors
3. If similarity > threshold → updates the existing entry
4. If similarity < threshold → stores as a new document

---

## 📊 Architecture

```
┌──────────────┐     ┌───────────────────┐     ┌─────────────┐
│   Telegram   │────▶│ telegram_bot.py   │────▶│  Pinecone   │
│   Message    │     │ (long polling)    │     │  (vectors)  │
└──────────────┘     └───────────────────┘     └─────────────┘
                            │
                            ▼
                    ┌───────────────────┐
                    │ pinecone_manager  │
                    │ (embeddings,      │
                    │  upsert/query)    │
                    └───────────────────┘
                            │
                            ▼
                    ┌───────────────────┐
                    │   OpenAI API      │
                    │ (chat, embeddings)│
                    └───────────────────┘
```

---

## 🔍 Troubleshooting

### Pinecone Connection Errors

- Verify `PINECONE_API_KEY` and `PINECONE_INDEX_NAME`
- Make sure the index exists and is accessible
- Check that the index metric is `cosine`

### Telegram Connection Errors

- Verify `TELEGRAM_BOT_TOKEN`
- If you have network issues, set `TG_PROXY_URL`
- Make sure only one bot instance is running

### Bot Not Responding

- Check logs for timeouts
- Verify that `getMe` succeeds
- Make sure the Pinecone index is not empty

---

## 🔒 Security

- **Never commit `.env`** — only `.env.example` goes in the repository
- **Rotate keys** if you suspect a leak
- **Restrict access** to the bot via privacy settings

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 👤 Author

**CherkasovAV**

GitHub: [@CherkasovAV](https://github.com/CherkasovAV)

---

## 🙋 Support

- Questions and suggestions: open an Issue in the repository
- Telegram: [@CherkasovAV](https://t.me/CherkasovAV)
- Email: [cherkasov83@yandex.ru](mailto:cherkasov83@yandex.ru)