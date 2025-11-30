import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:
    raise RuntimeError("sentence-transformers is required for TemporaryRAGChatService") from exc

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
except ImportError:
    from requests.packages.urllib3.util.retry import Retry  # type: ignore


class TemporaryRAGChatService:
    """In-memory service that turns the latest transcript into a temporary RAG store."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size: int = 420,
        chunk_overlap: int = 60,
    ):
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedder: Optional[SentenceTransformer] = None
        self._sessions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def ensure_session(self, payload: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Prepare cached embeddings for a temporary knowledge base payload.

        Args:
            payload: Dict coming from `transcribe_file` with `session_id` & `combined_text`.

        Returns:
            session_id if prepared successfully, otherwise None.
        """
        if not payload:
            return None

        combined_text = (payload.get("combined_text") or "").strip()
        files: Sequence[Dict[str, Any]] = payload.get("files") or []
        if not combined_text and not files:
            return None

        session_id = payload.get("session_id") or str(uuid4())

        if session_id in self._sessions:
            return session_id

        chunks: List[str] = []
        for file_entry in files:
            file_text = (file_entry or {}).get("text") or ""
            chunks.extend(self._split_text(file_text))

        if not chunks and combined_text:
            chunks = self._split_text(combined_text)

        if not chunks:
            # Keep placeholder session to indicate "empty knowledge base"
            self._sessions[session_id] = {"chunks": [], "embeddings": np.zeros((0, 0)), "created_at": time.time()}
            return session_id

        embeddings = self._embed(chunks)
        self._sessions[session_id] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "created_at": time.time(),
        }
        return session_id

    def clear_session(self, session_id: Optional[str]):
        if session_id and session_id in self._sessions:
            self._sessions.pop(session_id, None)

    def generate_reply(
        self,
        payload: Optional[Dict[str, Any]],
        user_message: str,
        history: Optional[List[List[str]]] = None,
        base_url: Optional[str] = None,
        model: str = "qwen2.5:3b",
        top_k: int = 4,
        similarity_threshold: float = 0.75,
    ) -> Tuple[str, bool]:
        """
        Generate a reply using Ollama with optional temporary context.

        Returns:
            (answer, used_context_flag)
        """
        history = history or []
        session_id = self.ensure_session(payload)
        context_chunks: List[str] = []
        used_context = False

        if session_id and self._has_chunks(session_id):
            context_chunks = self._retrieve_context(session_id, user_message, top_k, similarity_threshold)
            used_context = len(context_chunks) > 0

        context_block = (
            "\n\n".join([f"[片段{i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)])
            if context_chunks
            else "（当前没有可用的访谈文本上下文。）"
        )

        system_prompt = (
            "你是一个访谈分析助手。优先参考“访谈片段”回答问题；"
            "当片段不足以支撑答案时，请基于常识回答，但明确指出猜测成分。"
            "\n\n访谈片段：\n"
            f"{context_block}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history:
            if not isinstance(turn, (list, tuple)) or len(turn) != 2:
                continue
            user_turn, assistant_turn = turn
            if user_turn:
                messages.append({"role": "user", "content": user_turn})
            if assistant_turn:
                messages.append({"role": "assistant", "content": assistant_turn})

        messages.append({"role": "user", "content": user_message})

        response_text = self._call_ollama_chat(messages, base_url or "http://localhost:11434", model)
        return response_text, used_context

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _has_chunks(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        return bool(session and session.get("chunks"))

    def _split_text(self, text: str) -> List[str]:
        normalized = (text or "").replace("\r\n", "\n")
        paragraphs = [p.strip() for p in normalized.split("\n") if p.strip()]
        chunks: List[str] = []
        for paragraph in paragraphs:
            start = 0
            while start < len(paragraph):
                end = start + self.chunk_size
                chunks.append(paragraph[start:end])
                if end >= len(paragraph):
                    break
                start = end - self.chunk_overlap
        return [c for c in chunks if c.strip()]

    def _get_embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name, device="cpu")
        return self._embedder

    def _embed(self, chunks: List[str]) -> np.ndarray:
        if not chunks:
            return np.zeros((0, 0))
        embedder = self._get_embedder()
        vectors = embedder.encode(chunks, batch_size=16, convert_to_numpy=True, show_progress_bar=False)
        return vectors

    def _retrieve_context(
        self,
        session_id: str,
        query: str,
        top_k: int,
        similarity_threshold: float,
    ) -> List[str]:
        session = self._sessions.get(session_id)
        if not session or not session.get("chunks"):
            return []
        embeddings = session["embeddings"]
        chunks = session["chunks"]
        embedder = self._get_embedder()
        query_vec = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]

        if embeddings.size == 0:
            return []

        sims = self._cosine_similarity(query_vec, embeddings)
        ranked_indices = np.argsort(-sims)

        results = []
        for idx in ranked_indices[: max(1, int(top_k))]:
            if sims[idx] >= similarity_threshold:
                results.append(chunks[idx])
        return results

    @staticmethod
    def _cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        vec_norm = np.linalg.norm(vec) + 1e-8
        matrix_norm = np.linalg.norm(matrix, axis=1) + 1e-8
        return (matrix @ vec) / (matrix_norm * vec_norm)

    def _call_ollama_chat(self, messages: List[Dict[str, str]], base_url: str, model: str) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        session = requests.Session()
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        resp = session.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message") or {}
        return (message.get("content") or "").strip() or "（未从模型获得有效回复）"

