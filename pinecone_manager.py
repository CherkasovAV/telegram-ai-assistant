"""
Менеджер Pinecone.

Методы: create_embedding, upsert_vector(s), upsert_document(s), query_by_vector/text,
fetch_vectors, delete / delete_by_filter / delete_all, describe_index_stats, update_metadata,
assess_memory_similarity, remember_document (запись в память с проверкой сходства).
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, MutableMapping, Optional, Sequence

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# --- Долговременная память чат-бота: порог косинусного сходства ---
# Score из ответа Pinecone query (индекс с metric="cosine"): чем выше, тем ближе векторы.
# Значения >= порога — «высокое» сходство (дубликат / вариация той же мысли).
# Значения < порога — «низкое» сходство (новая информация → обычно записываем).
MEMORY_COSINE_SIMILARITY_HIGH_THRESHOLD: float = 0.85

Vector = Sequence[float]
Metadata = Mapping[str, Any]
VectorRecord = MutableMapping[str, Any]
EmbedFn = Callable[[str], Vector]
MemoryDuplicateMode = Literal["skip", "update"]


@dataclass(frozen=True)
class MemorySimilarityAssessment:
    """Результат сравнения текста с уже сохранёнными фрагментами."""

    max_score: float | None
    best_match_id: str | None
    best_match_metadata: dict[str, Any] | None
    is_high_similarity: bool
    threshold_used: float


@dataclass(frozen=True)
class MemoryWriteResult:
    """Итог попытки записи фрагмента в память с учётом порога сходства."""

    action: Literal["stored", "skipped", "updated"]
    vector_id: str | None
    max_similarity: float | None
    similar_to_id: str | None
    threshold_used: float


class PineconeManager:
    """Класс для управления операциями с векторной базой данных Pinecone."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        openai_base_url: Optional[str] = None,
        *,
        namespace: str = "",
        host: Optional[str] = None,
    ) -> None:
        """
        Инициализация менеджера Pinecone.

        Args:
            api_key: API ключ Pinecone (если None, загружается из .env)
            environment: Окружение/регион Pinecone (если None, из .env)
            index_name: Имя индекса (если None, загружается из .env)
            openai_api_key: API ключ OpenAI для эмбеддингов (если None, из .env)
            openai_model: Модель OpenAI для создания эмбеддингов
            openai_base_url: Базовый URL API (если None, из OPENAI_BASE_URL в .env; для прокси / совместимых эндпоинтов)
            namespace: Namespace в индексе (пустая строка — default namespace)
            host: Явный host индекса (опционально)
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.namespace = namespace

        if not self.api_key:
            raise ValueError(
                "PINECONE_API_KEY не найден. Укажите в параметрах или в .env файле."
            )
        if not self.index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME не найден. Укажите в параметрах или в .env файле."
            )

        self.pc = Pinecone(api_key=self.api_key)

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_model = openai_model
        self.openai_base_url = (openai_base_url or os.getenv("OPENAI_BASE_URL") or "").strip() or None
        self.openai_client: OpenAI | None = None
        if self.openai_api_key:
            client_kwargs: dict[str, Any] = {"api_key": self.openai_api_key}
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            self.openai_client = OpenAI(**client_kwargs)

        if host:
            self.index = self.pc.Index(self.index_name, host=host)
        else:
            self.index = self.pc.Index(self.index_name)

    def _ns_kwargs(self) -> dict[str, str]:
        return {"namespace": self.namespace} if self.namespace else {}

    def create_embedding(self, text: str) -> list[float]:
        """Создание эмбеддинга для текста через модель OpenAI."""
        if not self.openai_client:
            raise ValueError(
                "OpenAI клиент не инициализирован. Передайте openai_api_key или задайте OPENAI_API_KEY."
            )
        response = self.openai_client.embeddings.create(
            model=self.openai_model,
            input=text,
        )
        return list(response.data[0].embedding)

    def _resolve_embed_fn(self, embed_fn: Optional[EmbedFn]) -> EmbedFn:
        if embed_fn is not None:
            return embed_fn
        return self.create_embedding

    # --- Запись векторов ---

    def upsert_vectors(
        self,
        vectors: Sequence[VectorRecord],
        *,
        batch_size: int = 100,
    ) -> None:
        """Запись нескольких векторов: элементы {"id", "values", опционально "metadata"}."""
        batch: list[VectorRecord] = []
        for rec in vectors:
            batch.append(dict(rec))
            if len(batch) >= batch_size:
                self.index.upsert(vectors=batch, **self._ns_kwargs())
                batch.clear()
        if batch:
            self.index.upsert(vectors=batch, **self._ns_kwargs())

    def upsert_vector(
        self,
        vector_id: str,
        values: Vector,
        metadata: Metadata | None = None,
    ) -> None:
        """Запись одного вектора: id, значения и опциональные метаданные."""
        row: VectorRecord = {"id": vector_id, "values": list(values)}
        if metadata is not None:
            row["metadata"] = dict(metadata)
        self.index.upsert(vectors=[row], **self._ns_kwargs())

    # --- Запись документов (текст → эмбеддинг) ---

    def upsert_document(
        self,
        document: Mapping[str, Any],
        embed_fn: Optional[EmbedFn] = None,
        *,
        text_key: str = "text",
        id_key: str = "id",
        metadata_key: str = "metadata",
        generate_ids: bool = False,
    ) -> str:
        """
        Запись одного документа. Текст из поля text_key преобразуется в вектор
        через embed_fn или create_embedding (OpenAI).
        """
        ids = self.upsert_documents(
            [document],
            embed_fn=embed_fn,
            text_key=text_key,
            id_key=id_key,
            metadata_key=metadata_key,
            generate_ids=generate_ids,
        )
        return ids[0]

    def upsert_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        embed_fn: Optional[EmbedFn] = None,
        *,
        text_key: str = "text",
        id_key: str = "id",
        metadata_key: str = "metadata",
        generate_ids: bool = False,
        batch_size: int = 100,
    ) -> list[str]:
        """
        Запись документов: для каждого текста строится эмбеддинг (OpenAI или embed_fn).
        Возвращает список id записанных векторов.
        """
        fn = self._resolve_embed_fn(embed_fn)
        rows: list[VectorRecord] = []
        used_ids: list[str] = []

        for doc in documents:
            text = doc[text_key]
            if not isinstance(text, str):
                raise TypeError(f"Поле {text_key!r} должно быть строкой")

            if generate_ids or id_key not in doc:
                vid = str(uuid.uuid4())
            else:
                vid = str(doc[id_key])

            meta: dict[str, Any] = {}
            if metadata_key in doc and doc[metadata_key] is not None:
                meta = dict(doc[metadata_key])
            meta.setdefault(text_key, text)

            vec = fn(text)
            row: VectorRecord = {"id": vid, "values": list(vec), "metadata": meta}
            rows.append(row)
            used_ids.append(vid)

        self.upsert_vectors(rows, batch_size=batch_size)
        return used_ids

    # --- Поиск ---

    def query_by_vector(
        self,
        vector: Vector,
        *,
        top_k: int = 10,
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Any:
        """Поиск ближайших записей по вектору запроса."""
        kwargs: dict[str, Any] = {
            "vector": list(vector),
            "top_k": top_k,
            "include_values": include_values,
            "include_metadata": include_metadata,
            **self._ns_kwargs(),
        }
        if filter is not None:
            kwargs["filter"] = dict(filter)
        return self.index.query(**kwargs)

    def query_by_text(
        self,
        text: str,
        embed_fn: Optional[EmbedFn] = None,
        *,
        top_k: int = 10,
        filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> Any:
        """Поиск по тексту: сначала эмбеддинг (create_embedding или embed_fn), затем query_by_vector."""
        q = self._resolve_embed_fn(embed_fn)(text)
        return self.query_by_vector(
            q,
            top_k=top_k,
            filter=filter,
            include_values=include_values,
            include_metadata=include_metadata,
        )

    @staticmethod
    def _effective_memory_threshold(override: float | None) -> float:
        return (
            MEMORY_COSINE_SIMILARITY_HIGH_THRESHOLD
            if override is None
            else override
        )

    @staticmethod
    def _best_match_from_query(
        response: Any,
    ) -> tuple[float | None, str | None, dict[str, Any] | None]:
        matches = getattr(response, "matches", None) or []
        if not matches:
            return None, None, None
        m0 = matches[0]
        score = getattr(m0, "score", None)
        vid = getattr(m0, "id", None)
        raw_meta = getattr(m0, "metadata", None) or {}
        meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        max_score = float(score) if score is not None else None
        best_id = str(vid) if vid is not None else None
        return max_score, best_id, meta if meta else None

    def assess_memory_similarity(
        self,
        text: str,
        embed_fn: Optional[EmbedFn] = None,
        *,
        top_k: int = 5,
        filter: Mapping[str, Any] | None = None,
        similarity_threshold: float | None = None,
    ) -> MemorySimilarityAssessment:
        """
        Сравнивает текст с уже сохранёнными векторами (query по эмбеддингу).
        Индекс должен использовать metric=\"cosine\", иначе интерпретация score другая.
        """
        threshold = self._effective_memory_threshold(similarity_threshold)
        vec = self._resolve_embed_fn(embed_fn)(text)
        response = self.query_by_vector(
            vec,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
        )
        max_score, best_id, best_meta = self._best_match_from_query(response)
        is_high = max_score is not None and max_score >= threshold
        return MemorySimilarityAssessment(
            max_score=max_score,
            best_match_id=best_id,
            best_match_metadata=best_meta,
            is_high_similarity=is_high,
            threshold_used=threshold,
        )

    def remember_document(
        self,
        document: Mapping[str, Any],
        embed_fn: Optional[EmbedFn] = None,
        *,
        text_key: str = "text",
        id_key: str = "id",
        metadata_key: str = "metadata",
        generate_ids: bool = False,
        on_high_similarity: MemoryDuplicateMode = "skip",
        similarity_threshold: float | None = None,
        top_k: int = 5,
        filter: Mapping[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """
        Запись фрагмента в долговременную память с проверкой сходства.

        При низком сходстве с ближайшими соседями (< порога) — upsert новой записи.
        При высоком (>= порога): при on_high_similarity=\"skip\" ничего не пишем;
        при \"update\" — обновляем тот же id (новый эмбеддинг и merge метаданных).
        Порог по умолчанию — MEMORY_COSINE_SIMILARITY_HIGH_THRESHOLD.
        """
        threshold = self._effective_memory_threshold(similarity_threshold)
        fn = self._resolve_embed_fn(embed_fn)
        text = document[text_key]
        if not isinstance(text, str):
            raise TypeError(f"Поле {text_key!r} должно быть строкой")

        vec = fn(text)
        response = self.query_by_vector(
            vec,
            top_k=top_k,
            filter=filter,
            include_metadata=True,
        )
        max_score, best_id, best_meta = self._best_match_from_query(response)

        new_meta: dict[str, Any] = {}
        if metadata_key in document and document[metadata_key] is not None:
            new_meta = dict(document[metadata_key])
        new_meta.setdefault(text_key, text)

        if max_score is None or max_score < threshold:
            if generate_ids or id_key not in document:
                vid = str(uuid.uuid4())
            else:
                vid = str(document[id_key])
            self.upsert_vector(vid, vec, new_meta)
            return MemoryWriteResult(
                action="stored",
                vector_id=vid,
                max_similarity=max_score,
                similar_to_id=None,
                threshold_used=threshold,
            )

        if not best_id:
            vid = str(uuid.uuid4()) if (generate_ids or id_key not in document) else str(
                document[id_key]
            )
            self.upsert_vector(vid, vec, new_meta)
            return MemoryWriteResult(
                action="stored",
                vector_id=vid,
                max_similarity=max_score,
                similar_to_id=None,
                threshold_used=threshold,
            )

        if on_high_similarity == "skip":
            return MemoryWriteResult(
                action="skipped",
                vector_id=None,
                max_similarity=max_score,
                similar_to_id=best_id,
                threshold_used=threshold,
            )

        merged: dict[str, Any] = dict(best_meta) if best_meta else {}
        merged.update(new_meta)
        merged[text_key] = text
        self.upsert_vector(best_id, vec, merged)
        return MemoryWriteResult(
            action="updated",
            vector_id=best_id,
            max_similarity=max_score,
            similar_to_id=best_id,
            threshold_used=threshold,
        )

    def fetch_vectors(self, ids: Sequence[str]) -> Any:
        """Получение векторов (и метаданных) по списку id."""
        return self.index.fetch(ids=list(ids), **self._ns_kwargs())

    def describe_index_stats(self) -> Any:
        """Статистика индекса."""
        return self.index.describe_index_stats()

    # --- Удаление ---

    def delete(self, ids: Sequence[str]) -> None:
        """Удаление векторов по списку id."""
        self.index.delete(ids=list(ids), **self._ns_kwargs())

    def delete_by_filter(self, filter: Mapping[str, Any]) -> None:
        """Удаление векторов, подходящих под фильтр метаданных."""
        self.index.delete(filter=dict(filter), **self._ns_kwargs())

    def delete_all(self) -> None:
        """Удаление всех векторов в текущем namespace."""
        self.index.delete(delete_all=True, **self._ns_kwargs())

    # --- Обновление метаданных ---

    def update_metadata(self, vector_id: str, metadata: Metadata) -> Any:
        """
        Обновление метаданных вектора по id. Поля из metadata перезаписывают
        одноимённые поля; остальные ключи у записи сохраняются (поведение Pinecone: merge).
        """
        kwargs: dict[str, Any] = {
            "id": vector_id,
            "set_metadata": dict(metadata),
            **self._ns_kwargs(),
        }
        return self.index.update(**kwargs)
