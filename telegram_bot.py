"""
Телеграм-бот-помощник: диалог через OpenAI и долговременная память пользователя в Pinecone (PineconeManager).

Возможности:
  - связный диалог: краткосрочная история реплик в RAM + долговременная память в Pinecone;
  - приём документов (.txt, .md, .pdf, .docx): извлечение текста, чанкинг, запись в Pinecone;
  - ретраи с паузами при нестабильном интернете (OpenAI SDK, Pinecone, Telegram API).

Переменные окружения (.env):
  TELEGRAM_BOT_TOKEN — токен бота от @BotFather
  PINECONE_API_KEY, PINECONE_INDEX_NAME — как в PineconeManager
  OPENAI_API_KEY — эмбеддинги и чат
  OPENAI_BASE_URL — опционально
  OPENAI_CHAT_MODEL — модель чата (по умолчанию gpt-4o-mini)
  DIALOG_HISTORY_TURNS — глубина краткосрочной истории (пар реплик, по умолчанию 6)
  NETWORK_RETRY_ATTEMPTS / NETWORK_RETRY_BASE_DELAY — ретраи сетевых вызовов
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable, TypeVar

import telebot
from dotenv import load_dotenv

import document_ingest
import memory_consolidation
from pinecone_manager import PineconeManager

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_MAX_MESSAGE = 4096
MAX_DOC_REPLY_TEXT = 400  # предпросмотр извлечённого текста в ответе бота

# Долговременная память ответов ассистента: хранится компактное резюме каждого
# ответа (type=assistant_summary). В обычный контекст НЕ подмешивается — только
# когда пользователь явно спрашивает «что ты мне говорил/отвечал».
ASSISTANT_SUMMARY_CHARS = int(os.getenv("ASSISTANT_SUMMARY_CHARS", "300"))
PAST_ANSWERS_TRIGGERS: tuple[str, ...] = (
    "что ты",           # что ты мне говорил/отвечал/предлагал
    "мне говорил",
    "мне отвечал",
    "ты говорил",
    "ты отвечал",
    "ты писал",
    "твои ответы",
    "твои советы",
    "что писали",
    "повтори",
    "перескажи",
)

DIALOG_HISTORY_TURNS = int(os.getenv("DIALOG_HISTORY_TURNS", "6"))
NETWORK_RETRY_ATTEMPTS = int(os.getenv("NETWORK_RETRY_ATTEMPTS", "4"))
NETWORK_RETRY_BASE_DELAY = float(os.getenv("NETWORK_RETRY_BASE_DELAY", "2.0"))

T = TypeVar("T")

# Краткосрочная история диалога: chat_id → deque последних реплик (user/assistant).
# Чистая RAM: сбрасывается при перезапуске бота, долговременные факты остаются в Pinecone.
dialog_history: dict[int, deque[dict[str, str]]] = {}

# Пользователи, чью память фоновая консолидация (как сон) периодически сжимает
known_users: set[int] = set()

MEMORY_CONSOLIDATION_INTERVAL_HOURS = float(os.getenv("MEMORY_CONSOLIDATION_HOURS", "24"))

ERROR_REPLY = (
    "Не смог ответить: сервисы или сеть недоступны. "
    "Я уже несколько раз переподключался — попробуй ещё раз через минуту."
)


def with_retries(
    fn: Callable[[], T],
    label: str,
    *,
    attempts: int = NETWORK_RETRY_ATTEMPTS,
    base_delay: float = NETWORK_RETRY_BASE_DELAY,
) -> T:
    """Выполняет fn с ретраями и экспоненциальной паузой при сетевых сбоях.

    HTTP 429 от Telegram обрабатывается отдельно: ждём retry_after.
    Прочие ошибки (Pinecone, сеть, таймауты) — повтор до attempts раз.
    """
    # Chat completion вызывает сам SDK OpenAI (max_retries=5), поэтому double-retry не нужен.
    max_pause = float(os.getenv("NETWORK_RETRY_MAX_DELAY", "30"))
    for attempt in range(1, attempts):
        try:
            return fn()
        except telebot.apihelper.ApiTelegramException as exc:
            # 4xx (кроме 429) — не сетевая проблема (неправильный запрос), не ретраим.
            if exc.error_code != 429 and exc.error_code < 500:
                raise
            retry_after = 0
            if exc.error_code == 429:
                data = getattr(exc, "result_json", None) or {}
                try:
                    retry_after = int(
                        data.get("parameters", {}).get("retry_after", base_delay)
                    )
                except (TypeError, ValueError):
                    retry_after = int(base_delay)
            logger.warning(
                "%s: попытка %d/%d не удалась (%s), пауза %s c",
                label,
                attempt,
                attempts,
                exc.description if hasattr(exc, "description") else exc,
                retry_after or f"{min(base_delay * 2 ** (attempt - 1), max_pause):.0f}",
            )
            time.sleep(retry_after or min(base_delay * 2 ** (attempt - 1), max_pause))
        except Exception as exc:  # noqa: BLE001 — страхуем любые сетевые/интерфейсные сбои
            pause = min(base_delay * 2 ** (attempt - 1), max_pause)
            logger.warning(
                "%s: попытка %d/%d не удалась (%s: %s), пауза %.0f c",
                label,
                attempt,
                attempts,
                type(exc).__name__,
                exc,
                pause,
            )
            time.sleep(pause)
    return fn()


def send_text(bot: telebot.TeleBot, chat_id: int, text: str) -> None:
    """Отправка текста частями, с ретраями при сетевых сбоях."""
    for chunk in split_telegram_chunks(text):
        with_retries(
            lambda c=chunk: bot.send_message(chat_id, c),
            "send_message (Telegram)",
        )


class TypingSignal:
    """Показывает «печатает…» (chat action) на время долгих операций.

    Одно действие Telegram живёт ~5 секунд — фоновый поток повторяет его
    каждые 4 секунды, пока идёт обдумывание/индексация.
    """

    ACTION_INTERVAL = 4.0

    def __init__(self, bot: telebot.TeleBot, chat_id: int, action: str = "typing") -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._action = action
        self._stop = threading.Event()

    def start(self) -> None:
        self._stop.clear()
        self._show()
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _show(self) -> None:
        try:
            self._bot.send_chat_action(self._chat_id, self._action)
        except Exception:  # noqa: BLE001 — индикатор не важнее ответа
            pass

    def _loop(self) -> None:
        while not self._stop.wait(self.ACTION_INTERVAL):
            self._show()


def push_history(chat_id: int, role: str, content: str) -> None:
    q = dialog_history.setdefault(chat_id, deque(maxlen=DIALOG_HISTORY_TURNS * 2))
    q.append({"role": role, "content": content})


def history_messages(chat_id: int) -> list[dict[str, str]]:
    return list(dialog_history.get(chat_id, ()))


def user_memory_filter(chat_id: int) -> dict[str, Any]:
    return {"user_id": {"$eq": str(chat_id)}}


MAX_MEMORY_ITEMS = 8  # сколько фактов памяти попадает в промпт
MAX_DOC_ITEMS = 6  # сколько чанков документов попадает в промпт


def assistant_summary(answer: str) -> str:
    """Компактное резюме ответа бота: до ASSISTANT_SUMMARY_CHARS, по границе предложения."""
    answer = answer.strip()
    if len(answer) <= ASSISTANT_SUMMARY_CHARS:
        return answer
    window = answer[: ASSISTANT_SUMMARY_CHARS]
    for sep in (". ", "? ", "! ", "\n"):
        cut = window.rfind(sep)
        if cut > ASSISTANT_SUMMARY_CHARS // 2:
            return window[: cut + len(sep)].strip()
    return window.rstrip() + "…"


def wants_past_answers(user_text: str) -> bool:
    """Считает ли пользователь, что просит процитировать прошлые ответы бота."""
    lowered = user_text.lower()
    return any(trigger in lowered for trigger in PAST_ANSWERS_TRIGGERS)


def context_from_query(response: Any) -> tuple[str, str]:
    """Разделяет найденные фрагменты на память диалога и чанки документов.

    Возвращает (память, документы); пустые строки — если раздела нет.
    Ответы ассистента (type=assistant_message) в память не включаются:
    это «эхо» прошлых ответов бота, они затапливают выдачу и могут
    пересказывать собственные ошибки («я не знаю») как факт.
    Старые записи без metadata.type остаются в памяти — совместимость
    с векторами, записанными до появления документов.
    """
    memory_parts: list[str] = []
    doc_parts: list[str] = []
    for m in getattr(response, "matches", None) or []:
        meta = getattr(m, "metadata", None) or {}
        if not isinstance(meta, dict):
            continue
        text = meta.get("text")
        if not text:
            continue
        mtype = meta.get("type")
        if mtype == "file_chunk":
            name = str(meta.get("file_name") or "файл")
            doc_parts.append(f"[{name}] {str(text).strip()}")
        elif mtype not in ("assistant_message", "assistant_summary"):
            memory_parts.append(str(text).strip())
    mem = "\n".join(f"• {p}" for p in memory_parts[:MAX_MEMORY_ITEMS] if p)
    docs = "\n\n".join(doc_parts[:MAX_DOC_ITEMS])
    return mem, docs


def stores_summary(answer: str) -> bool:
    """Не сохранять резюме ответов-признаний («я не помню/не знаю») — иначе бот
    эхоит собственное незнание вместо фактов."""
    head = answer.strip().lower()[:60]
    if any(
        phrase in head
        for phrase in ("не знаю", "не помню", "не смог", "не удалось", "не имею")
    ):
        return False
    return answer != FALLBACK_ANSWER and bool(answer.strip())


def split_telegram_chunks(text: str, limit: int = TELEGRAM_MAX_MESSAGE) -> list[str]:
    if len(text) <= limit:
        return [text]
    return [text[i : i + limit] for i in range(0, len(text), limit)]


FALLBACK_ANSWER = "Не смог сформулировать ответ, попробуй ещё раз."


def make_fact_arbitrator(memory: PineconeManager, chat_model: str) -> Callable[[str, str], str]:
    """LLM-арбитр для «серой зоны» сходства (0.65–0.85).

    Сравнивает новый факт со старым и возвращает same/contradiction/unrelated.
    Пример: «хочу в Новгород» vs «не хочу в Новгород, хочу в Тагил» — для векторов
    это 0.76 (ниже порога перезаписи), а арбитр распознаёт противоречие.
    Неизвестный ответ трактуется как unrelated — не перезаписываем.
    """
    client = memory.openai_client

    def arbitrate(new_text: str, old_text: str) -> str:
        completion = client.chat.completions.create(
            model=chat_model,
            temperature=0,
            max_tokens=8,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты классификатор фактов в личной памяти ассистента. "
                        "Даны старое и новое утверждение пользователя. "
                        "Ответь ровно одним словом: "
                        "same — это повтор/перефраз/уточнение того же факта; "
                        "contradiction — новое противоречит или исправляет старое; "
                        "unrelated — независимые факты."
                    ),
                },
                {"role": "user", "content": f"Старое: «{old_text}»\nНовое: «{new_text}»"},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip().lower()
        return raw if raw in ("same", "contradiction", "unrelated") else "unrelated"

    return arbitrate


def store_memory(
    memory: PineconeManager,
    uid: int,
    text: str,
    meta: dict[str, Any],
    flt: dict[str, Any],
    *,
    mode: str,
    arbitrator: Callable[[str, str], str] | None = None,
) -> Any:
    """Запись фрагмента в долговременную память с проверкой сходства, с ретраями."""
    result = with_retries(
        lambda: memory.remember_document(
            {"text": text, "metadata": meta},
            generate_ids=True,
            filter=flt,
            on_high_similarity="update" if mode == "update" else "skip",
            arbitrator=arbitrator,
        ),
        "remember_document (Pinecone)",
    )
    if result.verdict:
        score = result.max_similarity
        preview = (result.best_match_text or "")[:80]
        logger.info(
            "Память: арбитр=%s user_id=%s сходство=%s старый текст=%r",
            result.verdict,
            uid,
            f"{score:.4f}" if score is not None else "n/a",
            preview,
        )
    if result.action == "stored":
        preview = text[:80] + ("…" if len(text) > 80 else "")
        logger.info(
            "Память: сохранено user_id=%s type=%s vector_id=%s превью=%r",
            uid,
            meta.get("type"),
            result.vector_id,
            preview,
        )
    elif result.action == "skipped":
        score = result.max_similarity
        score_s = f"{score:.4f}" if score is not None else "n/a"
        logger.info(
            "Память: пропуск (похоже на существующее) user_id=%s "
            "similar_to_id=%s score=%s порог=%.4f",
            uid,
            result.similar_to_id,
            score_s,
            result.threshold_used,
        )
    else:
        logger.info(
            "Память: обновлён слот user_id=%s vector_id=%s",
            uid,
            result.vector_id,
        )
    return result


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Задайте TELEGRAM_BOT_TOKEN в .env")

    memory = PineconeManager()
    if not memory.openai_client:
        raise SystemExit("Нужен OPENAI_API_KEY для чата и эмбеддингов.")

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    bot = telebot.TeleBot(token, parse_mode=None)
    arbitrate = make_fact_arbitrator(memory, chat_model)

    def report_memory_write(
        uid: int,
        meta: dict[str, Any],
        text: str,
        flt: dict[str, Any],
        mode: str,
        arbitrator: Callable[[str, str], str] | None = None,
    ) -> None:
        try:
            store_memory(memory, uid, text, meta, flt, mode=mode, arbitrator=arbitrator)
        except Exception:  # noqa: BLE001 — сбои записи памяти не роняют ответ пользователю
            logger.exception("Не удалось записать в память user_id=%s", uid)

    @bot.message_handler(commands=["start", "help"])
    def send_welcome(message: telebot.types.Message) -> None:
        bot.reply_to(
            message,
            "Привет! Я помню наши разговоры и факты о тебе в долговременной памяти.\n\n"
            "Пиши текстом — отвечу с учётом истории диалога и того, что уже сохранено.\n"
            "Скинь документ (.txt, .md, .pdf, .docx) — я индексирую его в векторную "
            "базу и смогу отвечать по содержимому.\n\n"
            "Что я знаю о тебе — «что ты мне говорил?».\n"
            "Очистить память диалогов — /forget (с подтверждением).",
        )

    @bot.message_handler(commands=["forget"])
    def ask_forget_confirmation(message: telebot.types.Message) -> None:
        bot.reply_to(
            message,
            "Удалю ВСЮ диалоговую память: твои сообщения, резюме моих ответов и дайджесты. "
            "Проиндексированные документы останутся.\n\n"
            "Подтверди командой /forget_yes.",
        )

    @bot.message_handler(commands=["forget_yes"])
    def do_forget(message: telebot.types.Message) -> None:
        chat_id = message.chat.id
        uid = message.from_user.id if message.from_user else chat_id
        try:
            deleted = with_retries(
                lambda: memory_consolidation.forget_dialogue_memory(memory, uid),
                "forget (Pinecone)",
            )
            dialog_history.pop(chat_id, None)
            send_text(
                bot,
                chat_id,
                f"Память диалогов очищена: удалено записей — {deleted}. Документы остались.",
            )
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось очистить память user_id=%s", uid)
            send_text(bot, chat_id, "Не смог очистить память — сервисы недоступны. Попробуй позже.")

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
        known_users.add(uid)
        think = TypingSignal(bot, chat_id)
        think.start()

        try:
            # Запрашиваем с запасом: мусорные записи (эхо-ответы, резюме) отфильтровываются
            # ПОСЛЕ выборки, поэтому реальный размер выдачи меньше top_k.
            q = with_retries(
                lambda: memory.query_by_text(
                    user_text,
                    top_k=25,
                    filter=flt,
                    include_metadata=True,
                ),
                "query_by_text (Pinecone)",
            )
            mem_ctx, doc_ctx = context_from_query(q)

            blocks = [f"Память:\n{mem_ctx or '(записей пока нет)'}"]
            if doc_ctx:
                blocks.append("Из загруженных документов:\n" + doc_ctx)

            # Явный вопрос «что ты мне говорил/отвечал» — отдельный запрос по резюме ответов бота
            if wants_past_answers(user_text):
                try:
                    qs = with_retries(
                        lambda: memory.query_by_text(
                            user_text,
                            top_k=6,
                            filter={
                                "user_id": {"$eq": str(uid)},
                                "type": {"$eq": "assistant_summary"},
                            },
                            include_metadata=True,
                        ),
                        "query assistant_summary (Pinecone)",
                    )
                    past_parts: list[str] = []
                    for r in getattr(qs, "matches", None) or []:
                        rm = getattr(r, "metadata", None) or {}
                        if isinstance(rm, dict) and rm.get("text"):
                            past_parts.append(str(rm["text"]).strip())
                    if past_parts:
                        blocks.append(
                            "Мои прошлые ответы на похожие темы:\n"
                            + "\n".join(f"• {p}" for p in past_parts)
                        )
                except Exception:  # noqa: BLE001 — не хватает резюме, идёт основной ответ
                    logger.exception("Не удалось получить резюме прошлых ответов user_id=%s", uid)
            system = (
                "Ты дружелюбный помощник в Telegram. Отвечай по-русски, кратко и по делу, "
                "если пользователь не просит иначе.\n\n"
                "Продолжай диалог: учитывай предшествующие реплики, ссылайся на сказанное "
                "раньше, задавай уточняющие вопросы, если не хватает деталей.\n\n"
                "Ниже — выдержки из долговременной памяти и загруженных документов этого "
                "пользователя. Используй их, если релевантно; не выдумывай факты, которых "
                "там нет.\n\n" + "\n\n".join(blocks)
            )
            completion = memory.openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "system", "content": system}]
                + history_messages(chat_id)
                + [{"role": "user", "content": user_text}],
            )
            answer = (completion.choices[0].message.content or "").strip()
            if not answer:
                answer = FALLBACK_ANSWER

            push_history(chat_id, "user", user_text)
            push_history(chat_id, "assistant", answer)

            meta = {
                "user_id": str(uid),
                "telegram_chat_id": str(chat_id),
                "type": "user_message",
                "created_at": int(time.time()),
            }
            if message.from_user:
                if message.from_user.username:
                    meta["username"] = message.from_user.username
                if message.from_user.first_name:
                    meta["first_name"] = message.from_user.first_name
            report_memory_write(uid, meta, user_text, flt, mode="update", arbitrator=arbitrate)

            # Память об ответах: пишем компактное резюме (не полный текст),
            # чтобы не затапливать выдачу; подмешивается в контекст только
            # на явный вопрос «что ты мне говорил/отвечал» (wants_past_answers).
            if stores_summary(answer):
                summary = assistant_summary(answer)
                if summary:
                    summary_meta = {
                        "user_id": str(uid),
                        "telegram_chat_id": str(chat_id),
                        "type": "assistant_summary",
                        "created_at": int(time.time()),
                    }
                    report_memory_write(uid, summary_meta, summary, flt, mode="skip")

            send_text(bot, chat_id, answer)
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка обработки сообщения chat_id=%s", chat_id)
            try:
                bot.reply_to(message, ERROR_REPLY)
            except Exception:  # noqa: BLE001
                logger.warning("Не удалось отправить сообщение об ошибке chat_id=%s", chat_id)
        finally:
            think.stop()

    @bot.message_handler(content_types=["document"])
    def on_document(message: telebot.types.Message) -> None:
        doc = message.document
        if not doc:
            return
        chat_id = message.chat.id
        uid = message.from_user.id if message.from_user else chat_id
        known_users.add(uid)
        name = doc.file_name or "файл"
        ext = document_ingest.normalize_ext(name)

        if ext not in document_ingest.SUPPORTED_EXTENSIONS:
            send_text(
                bot,
                chat_id,
                f"Формат `{ext or name}` не поддерживаю. Пришли файл .txt, .md, .pdf или "
                ".docx — старый .doc пересохрани в .docx.",
            )
            return
        limit = document_ingest.DOC_FILE_SIZE_LIMIT_MB * 1024 * 1024
        if doc.file_size and doc.file_size > limit:
            send_text(
                bot,
                chat_id,
                f"Файл весит {doc.file_size / (1024 * 1024):.1f} МБ — больше лимита "
                f"{document_ingest.DOC_FILE_SIZE_LIMIT_MB} МБ (ограничение Telegram Bot API).",
            )
            return

        send_text(
            bot,
            chat_id,
            f"Принял 📄 {name}: скачиваю и индексирую в Pinecone. Отвечу, когда закончу.",
        )
        think_doc = TypingSignal(bot, chat_id)
        think_doc.start()
        try:
            file_info = with_retries(
                lambda: bot.get_file(doc.file_id), "get_file (Telegram)"
            )
            data = with_retries(
                lambda: bot.download_file(file_info.file_path),
                "download_file (Telegram)",
            )
            text = document_ingest.extract_text(data, name)
            if not text.strip():
                send_text(bot, chat_id, "Не нашёл в файле текста — похоже, он отсканирован с картинками или пуст.")
                return

            chunks = document_ingest.chunk_text(text)
            if not chunks:
                send_text(bot, chat_id, "Файл слишком короткий для индексации.")
                return

            meta = {
                "user_id": str(uid),
                "telegram_chat_id": str(chat_id),
                "type": "file_chunk",
                "file_name": name,
                "mime_type": doc.mime_type or document_ingest.SUPPORTED_EXTENSIONS[ext],
                "created_at": int(time.time()),
            }
            records = [
                {"text": chunk, "metadata": dict(meta, chunk_index=i)}
                for i, chunk in enumerate(chunks)
            ]
            with_retries(
                lambda: memory.upsert_documents(
                    records, generate_ids=True, batch_size=96
                ),
                "upsert_documents (Pinecone)",
            )
            logger.info(
                "Документ проиндексирован user_id=%s file=%r chunks=%d chars=%d",
                uid,
                name,
                len(chunks),
                len(text),
            )
            preview = text[:MAX_DOC_REPLY_TEXT].strip()
            ellipsis = "…" if len(text) > MAX_DOC_REPLY_TEXT else ""
            send_text(
                bot,
                chat_id,
                f"Готово: {name} проиндексирован.\n"
                f"Извлечено символов: {len(text)}; записано фрагментов: {len(chunks)}.\n"
                f"Теперь спрашивай — отвечу по содержимому документа.\n\n"
                f"Начало текста:\n“{preview}{ellipsis}”",
            )
        except ValueError as exc:
            send_text(bot, chat_id, f"Не смог обработать файл: {exc}")
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка индексации документа user_id=%s file=%r", uid, name)
            send_text(
                bot,
                chat_id,
                "Не смог обработать файл — сеть или сервисы недоступны. "
                "Попробуй отправить ещё раз.",
            )
        finally:
            think_doc.stop()

    @bot.message_handler(
        func=lambda m: getattr(m, "content_type", "")
        not in ("text", "document")
    )
    def fallback(message: telebot.types.Message) -> None:
        bot.reply_to(message, "Понимаю текст и файлы (.txt, .md, .pdf, .docx).")

    def consolidation_worker() -> None:
        """Раз в сутки сжимает старую память известных пользователей (как сон)."""
        time.sleep(90)  # дать боту стартовать
        while True:
            try:
                for uid in list(known_users):
                    memory_consolidation.consolidate_user(memory, chat_model, uid)
            except Exception:  # noqa: BLE001 — фон не должен падать
                logger.exception("Ошибка фоновой консолидации памяти")
            time.sleep(MEMORY_CONSOLIDATION_INTERVAL_HOURS * 3600)

    threading.Thread(target=consolidation_worker, daemon=True).start()

    logger.info(
        "Бот запущен (long polling): история %d пар, ретраи x%d, консолидация каждые %.0f ч",
        DIALOG_HISTORY_TURNS,
        NETWORK_RETRY_ATTEMPTS,
        MEMORY_CONSOLIDATION_INTERVAL_HOURS,
    )
    bot.infinity_polling(timeout=90, long_polling_timeout=90, skip_pending=True)


if __name__ == "__main__":
    main()
