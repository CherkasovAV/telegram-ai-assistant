"""
Телеграм-бот-помощник: диалог через OpenAI и долговременная память пользователя в Pinecone (PineconeManager).

Переменные окружения (.env):
  TELEGRAM_BOT_TOKEN — токен бота от @BotFather
  PINECONE_API_KEY, PINECONE_INDEX_NAME — как в PineconeManager
  OPENAI_API_KEY — эмбеддинги и чат
  OPENAI_BASE_URL — опционально
  OPENAI_CHAT_MODEL — модель чата (по умолчанию gpt-4o-mini)
"""

from __future__ import annotations

import logging
import os
from typing import Any

import telebot
from dotenv import load_dotenv

from pinecone_manager import PineconeManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE = 4096


def user_memory_filter(chat_id: int) -> dict[str, Any]:
    return {"user_id": {"$eq": str(chat_id)}}


def memory_context_from_query(response: Any) -> str:
    parts: list[str] = []
    for m in getattr(response, "matches", None) or []:
        meta = getattr(m, "metadata", None) or {}
        if not isinstance(meta, dict):
            continue
        text = meta.get("text")
        if text:
            parts.append(str(text).strip())
    if not parts:
        return ""
    return "\n".join(f"• {p}" for p in parts if p)


def split_telegram_chunks(text: str, limit: int = TELEGRAM_MAX_MESSAGE) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Задайте TELEGRAM_BOT_TOKEN в .env")

    memory = PineconeManager()
    if not memory.openai_client:
        raise SystemExit("Нужен OPENAI_API_KEY для чата и эмбеддингов.")

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    bot = telebot.TeleBot(token, parse_mode=None)

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message: telebot.types.Message) -> None:
        bot.reply_to(
            message,
            "Привет! Я помню наши разговоры и факты о тебе в долговременной памяти. "
            "Пиши сообщения текстом — отвечу с учётом того, что уже сохранено.",
        )

    @bot.message_handler(content_types=["text"])
    def on_text(message: telebot.types.Message) -> None:
        if not message.text:
            return
        chat_id = message.chat.id
        uid = message.from_user.id if message.from_user else chat_id
        flt = user_memory_filter(uid)
        user_text = message.text.strip()
        if not user_text:
            return

        try:
            q = memory.query_by_text(
                user_text,
                top_k=8,
                filter=flt,
                include_metadata=True,
            )
            ctx = memory_context_from_query(q)

            system = (
                "Ты дружелюбный помощник в Telegram. Отвечай по-русски, кратко и по делу, "
                "если пользователь не просит иначе.\n\n"
                "Ниже — выдержки из долговременной памяти только об этом пользователе. "
                "Используй их, если релевантно; не выдумывай факты, которых там нет.\n\n"
                f"Память:\n{ctx or '(записей пока нет)'}"
            )
            completion = memory.openai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
                ],
            )
            answer = (completion.choices[0].message.content or "").strip()
            if not answer:
                answer = "Не смог сформулировать ответ, попробуй ещё раз."

            meta = {
                "user_id": str(uid),
                "telegram_chat_id": str(chat_id),
                "role": "user",
            }
            if message.from_user:
                if message.from_user.username:
                    meta["username"] = message.from_user.username
                if message.from_user.first_name:
                    meta["first_name"] = message.from_user.first_name

            mem_result = memory.remember_document(
                {
                    "text": user_text,
                    "metadata": meta,
                },
                generate_ids=True,
                filter=flt,
                on_high_similarity="update",
            )
            if mem_result.action == "stored":
                preview = user_text[:80] + ("…" if len(user_text) > 80 else "")
                logger.info(
                    "Память: сохранено user_id=%s vector_id=%s превью=%r",
                    uid,
                    mem_result.vector_id,
                    preview,
                )
            elif mem_result.action == "skipped":
                score = mem_result.max_similarity
                score_s = f"{score:.4f}" if score is not None else "n/a"
                logger.info(
                    "Память: пропуск (похоже на существующее) user_id=%s "
                    "similar_to_id=%s score=%s порог=%.4f",
                    uid,
                    mem_result.similar_to_id,
                    score_s,
                    mem_result.threshold_used,
                )
            else:
                logger.info(
                    "Память: обновлён слот user_id=%s vector_id=%s",
                    uid,
                    mem_result.vector_id,
                )

            for chunk in split_telegram_chunks(answer):
                bot.send_message(chat_id, chunk)
        except Exception:
            logger.exception("Ошибка обработки сообщения chat_id=%s", chat_id)
            bot.reply_to(
                message,
                "Произошла ошибка при обращении к памяти или модели. Попробуй позже.",
            )

    @bot.message_handler(func=lambda m: getattr(m, "content_type", "") != "text")
    def fallback(message: telebot.types.Message) -> None:
        bot.reply_to(message, "Пока понимаю только текстовые сообщения.")

    logger.info("Бот запущен (long polling).")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()
