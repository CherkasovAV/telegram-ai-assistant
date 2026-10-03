"""
Фоновая «консолидация памяти как сон».

Раз в сутки старые записи диалоговой памяти пользователя (user_message и
assistant_summary с created_at старше MEMORY_CONSOLIDATION_DAYS, а также все
легаси-записи без created_at) сжимаются LLM в устойчивые факты (memory_digest),
после чего оригиналы удаляются. Документы (file_chunk) и прочие записи
не трогаются. Память перестаёт неограниченно расти, факты остаются.

Порядок безопасный: сначала upsert дайджестов, потом удаление оригиналов.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)

# Возраст записи, с которого она считается «старой» и подлежит консолидации
MEMORY_CONSOLIDATION_DAYS: float = float(os.getenv("MEMORY_CONSOLIDATION_DAYS", "7"))
# Не гонять LLM ради пары записей
MEMORY_CONSOLIDATION_MIN_RECORDS: int = int(os.getenv("MEMORY_CONSOLIDATION_MIN_RECORDS", "10"))

LISTING_TOP_K: int = 1000  # верхняя граница выборки памяти одного пользователя
DELETE_BATCH: int = 100

# Типы, которые консолидация собирает (входные тексты) и удаляет после дайджеста
CONSOLIDATED_TYPES: tuple[str, ...] = ("user_message", "assistant_summary")

DIGEST_MAX_CHARS: int = 15000  # предел текста, отдаваемого LLM на компрессию


def collect_dialog_records(memory: PineconeManager, uid: int) -> tuple[list[str], list[str]]:
    """Сбор старых диалоговых записей пользователя.

    Возвращает (ids для удаления, помеченные тексты для компрессии).
    Записи без created_at (легаси) считаются старыми.
    Листовка через query с фильтром: для объёмов личного бота выборки
    LISTING_TOP_K достаточно.
    """
    cutoff = time.time() - MEMORY_CONSOLIDATION_DAYS * 86400
    vec = memory.create_embedding("факты и сообщения пользователя из истории диалогов")
    q = memory.query_by_vector(
        vec,
        top_k=LISTING_TOP_K,
        filter={"user_id": {"$eq": str(uid)}},
        include_metadata=True,
    )
    ids: list[str] = []
    texts: list[str] = []
    for match in getattr(q, "matches", None) or []:
        meta = getattr(match, "metadata", None) or {}
        if not isinstance(meta, dict):
            continue
        mtype = meta.get("type")
        text = str(meta.get("text") or "").strip()
        if not text:
            continue
        if mtype == "memory_digest":
            continue  # свежие дайджесты пропускаем
        if mtype not in CONSOLIDATED_TYPES:
            # только легаси-записи без типа считаем фактами пользователя;
            # file_chunk (документы) и прочие типы не трогаем
            if mtype is None:
                mtype = "user_message"
            else:
                continue
        created = meta.get("created_at")
        if created is not None and float(created) > cutoff:
            continue  # свежая запись — ещё живёт в полной памяти
        ids.append(str(match.id))
        label = "пользователь" if mtype == "user_message" else "я (ассистент)"
        texts.append(f"[{label}] {text}")
    return ids, texts


def make_digests(memory: PineconeManager, chat_model: str, texts: list[str]) -> list[str]:
    """Сжатие коллекции старых фактов в устойчивые формулировки (LLM)."""
    if not memory.openai_client:
        raise ValueError("OpenAI клиент не инициализирован — консолидация невозможна")
    payload = "\n".join(texts)[:DIGEST_MAX_CHARS]
    completion = memory.openai_client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты архивариус личной памяти. Ниже — старые сообщения пользователя "
                    "(и ответы ассистента, помечены «я (ассистент)»). Сожми их в список "
                    "УСТОЙЧИВЫХ фактов о пользователе: по одному факту на строку, начиная "
                    "каждую строку с «- ». Объединяй повторы, отбрасывай вопросы, болтовню "
                    "и ссылки на прошедший момент («я хотел узнать х» → сам факт х). "
                    "Пиши по-русски, только проверяемые факты, без интерпретации."
                ),
            },
            {"role": "user", "content": payload},
        ],
    )
    raw = (completion.choices[0].message.content or "").strip()
    return [line.lstrip("-• ").strip() for line in raw.splitlines() if line.strip()]


def consolidate_user(memory: PineconeManager, chat_model: str, uid: int) -> int:
    """Консолидация памяти одного пользователя. Возвращает число удалённых записей (0 = нечего делать)."""
    ids, texts = collect_dialog_records(memory, uid)
    if len(texts) < MEMORY_CONSOLIDATION_MIN_RECORDS:
        return 0

    digests = [d for d in make_digests(memory, chat_model, texts) if d]
    if not digests:
        logger.warning("Консолидация: LLM вернул пустой дайджест user_id=%s — пропускаю", uid)
        return 0

    meta_base = {"user_id": str(uid), "type": "memory_digest"}
    records = [{"text": d, "metadata": dict(meta_base)} for d in digests]
    memory.upsert_documents(records, generate_ids=True, batch_size=96)

    deleted = 0
    for i in range(0, len(ids), DELETE_BATCH):
        memory.delete(ids[i : i + DELETE_BATCH])
        deleted += len(ids[i : i + DELETE_BATCH])

    logger.info(
        "Консолидация: user_id=%s — %d записей сжаты в %d фактов",
        uid,
        deleted,
        len(digests),
    )
    return deleted


def forget_dialogue_memory(memory: PineconeManager, uid: int) -> int:
    """Ручная очистка диалоговой памяти (/forget): удаляет всё, кроме file_chunk.

    Возвращает число удалённых записей. Дайджесты, сообщения, резюме и эхо — удаляются.
    """
    vec = memory.create_embedding("все сообщения и факты пользователя")
    q = memory.query_by_vector(
        vec,
        top_k=LISTING_TOP_K,
        filter={"user_id": {"$eq": str(uid)}},
        include_metadata=True,
    )
    ids = [
        str(m.id)
        for m in getattr(q, "matches", None) or []
        if (getattr(m, "metadata", None) or {}).get("type") != "file_chunk"
    ]
    deleted = 0
    for i in range(0, len(ids), DELETE_BATCH):
        memory.delete(ids[i : i + DELETE_BATCH])
        deleted += len(ids[i : i + DELETE_BATCH])
    logger.info("/forget: user_id=%s удалено записей=%d (документы не тронуты)", uid, deleted)
    return deleted


def integration_hint() -> dict[str, Any]:
    """Мелкая отладочная сводка конфигурации (для логов старта)."""
    return {
        "days": MEMORY_CONSOLIDATION_DAYS,
        "min_records": MEMORY_CONSOLIDATION_MIN_RECORDS,
    }