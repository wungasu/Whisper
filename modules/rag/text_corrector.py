import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import chromadb
from chromadb.config import Settings
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

from modules.utils.logger import get_logger
from modules.whisper.data_classes import Segment

logger = get_logger()

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,12}|[A-Za-z][A-Za-z0-9\-\+]{1,31}")


@dataclass
class CorrectionRecord:
    original: str
    corrected: str
    score: float
    reference: str


class TextCorrectionRAG:
    """
    轻量级文本纠错服务：
    1. 使用 SentenceTransformer 对 txt 知识库进行向量化，并存入 Chroma 持久化集合。
    2. 针对 Whisper 的段落输出，检索最相关的知识片段，提取高频术语并做模糊比对，仅替换疑似错别字。
    3. 支持基于 LLM 的智能纠错，通过上下文理解进行更准确的词语替换。
    """

    def __init__(
        self,
        knowledge_dir: str,
        persist_dir: str,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "rag_correction",
        chunk_size: int = 220,
        chunk_overlap: int = 40,
    ):
        self.knowledge_dir = knowledge_dir
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.persist_dir, exist_ok=True)

        self._embedder: Optional[SentenceTransformer] = None
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

        self.meta_path = os.path.join(self.persist_dir, "meta.json")
        self.token_stats_path = os.path.join(self.persist_dir, "token_stats.json")
        self.token_frequency = self._load_token_stats()

    # ---------- Public APIs ----------
    def configure(self, knowledge_dir: Optional[str] = None):
        if knowledge_dir:
            self.knowledge_dir = knowledge_dir

    def rebuild_index(self, knowledge_dir: Optional[str] = None) -> str:
        self.configure(knowledge_dir)
        kb_path = os.path.abspath(self.knowledge_dir)
        if not os.path.isdir(kb_path):
            return f"知识库目录不存在：{kb_path}"

        supported_ext = (".txt", ".md", ".markdown")
        kb_files = [
            os.path.join(kb_path, f)
            for f in os.listdir(kb_path)
            if f.lower().endswith(supported_ext) and os.path.isfile(os.path.join(kb_path, f))
        ]
        if not kb_files:
            return f"目录 {kb_path} 中未找到支持的文本文件（支持扩展名：txt、md）。"

        self._reset_collection()
        self.token_frequency = {}

        documents: List[str] = []
        metadatas: List[Dict[str, str]] = []
        ids: List[str] = []
        total_chunks = 0

        for file_path in kb_files:
            content = self._read_text_file(file_path)
            if content is None:
                continue

            for chunk_idx, chunk in enumerate(self._split_document(content, file_path)):
                chunk_id = f"{os.path.basename(file_path)}::{chunk_idx}"
                documents.append(chunk)
                metadatas.append({"source": file_path, "chunk_index": chunk_idx})
                ids.append(chunk_id)
                total_chunks += 1

                for token in set(self._tokenize(chunk)):
                    self.token_frequency[token] = self.token_frequency.get(token, 0) + 1

        if not documents:
            return "未能从知识库文件中切分出有效文本，请检查文件内容。"

        embeddings = self._embed(documents)
        try:
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        except Exception as exc:
            logger.error(f"写入 Chroma 失败: {exc}", exc_info=True)
            return f"写入向量数据库失败：{exc}"

        self._save_token_stats()
        self._save_meta(
            {
                "knowledge_dir": kb_path,
                "total_files": len(kb_files),
                "total_chunks": total_chunks,
                "total_terms": len(self.token_frequency),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        return (
            f"索引完成：{len(kb_files)} 个文本文件，{total_chunks} 个文本块，"
            f"{len(self.token_frequency)} 个术语。"
        )

    def get_status(self) -> str:
        meta = self._load_meta()
        if not meta:
            return "知识库尚未建立索引。"
        return (
            f"知识库目录：{meta.get('knowledge_dir', 'N/A')}\n"
            f"文件数：{meta.get('total_files', 0)}\n"
            f"文本块：{meta.get('total_chunks', 0)}\n"
            f"术语数：{meta.get('total_terms', 0)}\n"
            f"更新时间：{meta.get('updated_at', 'N/A')}"
        )

    def correct_segments(
        self,
        segments: Sequence[Segment],
        knowledge_dir: Optional[str] = None,
        top_k: int = 4,
        similarity_threshold: float = 0.8,
        max_corrections: int = 3,
    ) -> Tuple[List[Segment], List[CorrectionRecord]]:
        if not segments:
            return [], []

        self.configure(knowledge_dir)
        if not self._is_index_ready():
            logger.warning("RAG 纠错被启用，但知识库尚未建立索引。")
            return list(segments), []

        corrected_segments: List[Segment] = []
        applied_records: List[CorrectionRecord] = []

        for seg in segments:
            if not seg.text:
                corrected_segments.append(seg)
                continue

            new_text, records = self._correct_text(
                seg.text,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                max_corrections=max_corrections,
            )
            if records:
                applied_records.extend(records)
                corrected_segments.append(seg.model_copy(update={"text": new_text}))
            else:
                corrected_segments.append(seg)

        return corrected_segments, applied_records

    def correct_segments_with_llm(
        self,
        segments: Sequence[Segment],
        knowledge_dir: Optional[str] = None,
        llm_api_type: str = "openai",
        llm_api_key: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_model: str = "gpt-3.5-turbo",
        top_k: int = 4,
        similarity_threshold: float = 0.8,
    ) -> Tuple[List[Segment], List[CorrectionRecord]]:
        """
        使用 LLM 进行智能文本纠错。
        
        Args:
            segments: 要纠错的文本片段列表
            knowledge_dir: 知识库目录
            llm_api_type: LLM API 类型 ("openai", "ollama", "custom")
            llm_api_key: API 密钥（如 OpenAI API key）
            llm_base_url: API 基础 URL（如 Ollama 的本地 URL）
            llm_model: 使用的模型名称
            top_k: 检索相关知识片段的数量
            similarity_threshold: 相似度阈值（用于筛选检索结果）
            
        Returns:
            (纠错后的片段列表, 纠错记录列表)
        """
        if not segments:
            return [], []

        self.configure(knowledge_dir)
        if not self._is_index_ready():
            logger.warning("RAG 纠错被启用，但知识库尚未建立索引。")
            return list(segments), []

        # 检索相关知识库内容作为上下文
        knowledge_context = self._retrieve_knowledge_context(segments, top_k, similarity_threshold)
        
        corrected_segments: List[Segment] = []
        applied_records: List[CorrectionRecord] = []

        # 批量处理以提高效率
        texts = [seg.text for seg in segments if seg.text]
        if not texts:
            return list(segments), []
        
        corrected_texts = self._correct_texts_with_llm(
            texts=texts,
            knowledge_context=knowledge_context,
            llm_api_type=llm_api_type,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            llm_model=llm_model,
        )
        
        # 构建结果
        text_idx = 0
        for seg in segments:
            if seg.text:
                corrected_text = corrected_texts[text_idx] if text_idx < len(corrected_texts) else seg.text
                if corrected_text != seg.text:
                    # 提取替换记录（简化版，实际可以更详细）
                    records = self._extract_correction_records(seg.text, corrected_text)
                    applied_records.extend(records)
                    corrected_segments.append(seg.model_copy(update={"text": corrected_text}))
                else:
                    corrected_segments.append(seg)
                text_idx += 1
            else:
                corrected_segments.append(seg)

        return corrected_segments, applied_records

    def _retrieve_knowledge_context(
        self,
        segments: Sequence[Segment],
        top_k: int,
        similarity_threshold: float,
    ) -> str:
        """检索相关知识库内容作为 LLM 上下文"""
        if not segments:
            return ""
        
        # 合并所有文本作为查询
        query_text = " ".join([seg.text for seg in segments if seg.text])
        if not query_text:
            return ""
        
        query_embeddings = self._embed([query_text])
        self._ensure_collection()
        try:
            total_entries = self._collection.count()
        except Exception:
            total_entries = 0
        if total_entries == 0:
            return ""
        
        try:
            query = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=min(top_k, total_entries),
                include=["documents", "distances"],
            )
            documents = (query.get("documents") or [[]])[0]
            distances = (query.get("distances") or [[]])[0]
            
            # 过滤低相似度的结果
            filtered_docs = []
            for doc, dist in zip(documents, distances):
                similarity = 1 - dist  # cosine distance to similarity
                if similarity >= similarity_threshold:
                    filtered_docs.append(doc)
            
            return "\n\n".join(filtered_docs[:top_k]) if filtered_docs else ""
        except Exception as exc:
            logger.error(f"检索知识库失败: {exc}", exc_info=True)
            return ""

    def _correct_texts_with_llm(
        self,
        texts: List[str],
        knowledge_context: str,
        llm_api_type: str,
        llm_api_key: Optional[str],
        llm_base_url: Optional[str],
        llm_model: str,
    ) -> List[str]:
        """使用 LLM 对文本进行纠错"""
        # 获取 prompt 模板
        prompt_template = self._build_knowledge_prompt(knowledge_context)
        
        # 如果知识库上下文为空，使用默认值
        if not knowledge_context:
            knowledge_context = "（暂无相关知识库内容）"
        
        corrected_texts = []
        for text in texts:
            try:
                # 填充 prompt 模板
                prompt = prompt_template.format(context=knowledge_context, text_to_correct=text)
                
                corrected = self._call_llm_api(
                    text=text,
                    knowledge_prompt=prompt,
                    llm_api_type=llm_api_type,
                    llm_api_key=llm_api_key,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                )
                corrected_texts.append(corrected if corrected else text)
            except Exception as exc:
                logger.error(f"LLM 纠错失败: {exc}", exc_info=True)
                corrected_texts.append(text)
        
        return corrected_texts

    def _build_knowledge_prompt(self, knowledge_context: str) -> str:
        """构建包含知识库内容的 prompt"""
        prompt_template = """你是一位专业的文本校对专家。你的任务是根据我提供的【参考知识】来纠正【待纠错文本】中的错误词语、术语或事实。

请严格遵守以下规则：

1. 只修正与【参考知识】明显冲突的错误。

2. 保持原文的句子结构和风格不变，只替换错误的词语。

​3.将替换后的文本输出



---

【参考知识】:

{context}

---

【待纠错文本】:

{text_to_correct}

---

请根据以上规则，返回你的纠错结果。"""
        
        # 返回模板，实际使用时会在 _correct_texts_with_llm 中填充
        return prompt_template

    def _call_llm_api(
        self,
        text: str,
        knowledge_prompt: str,
        llm_api_type: str,
        llm_api_key: Optional[str],
        llm_base_url: Optional[str],
        llm_model: str,
    ) -> Optional[str]:
        """调用 LLM API
        
        注意：knowledge_prompt 已经是完整的 prompt，包含了参考知识和待纠错文本
        """
        # knowledge_prompt 已经是完整的 prompt，直接使用
        prompt = knowledge_prompt
        
        if llm_api_type == "openai":
            return self._call_openai_api(prompt, llm_api_key, llm_model)
        elif llm_api_type == "ollama":
            return self._call_ollama_api(prompt, llm_base_url or "http://localhost:11434", llm_model)
        elif llm_api_type == "custom":
            return self._call_custom_api(prompt, llm_base_url, llm_api_key, llm_model)
        else:
            logger.warning(f"未知的 LLM API 类型: {llm_api_type}")
            return None

    def _call_openai_api(self, prompt: str, api_key: Optional[str], model: str) -> Optional[str]:
        """调用 OpenAI API"""
        try:
            import openai
            if not api_key:
                logger.warning("OpenAI API key 未提供")
                return None
            
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专业的文本纠错助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000,
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            logger.error("openai 库未安装，请运行: pip install openai")
            return None
        except Exception as exc:
            logger.error(f"OpenAI API 调用失败: {exc}", exc_info=True)
            return None

    def _call_ollama_api(self, prompt: str, base_url: str, model: str) -> Optional[str]:
        """调用 Ollama API
        
        注意：Ollama 会在首次使用模型时自动下载，但需要：
        1. Ollama 服务已启动（运行 `ollama serve` 或通过桌面应用启动）
        2. 如果模型不存在，Ollama 会自动下载（可能需要一些时间）
        """
        try:
            import requests
            from requests.adapters import HTTPAdapter
            try:
                from urllib3.util.retry import Retry
            except ImportError:
                from requests.packages.urllib3.util.retry import Retry
            
            url = f"{base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                }
            }
            
            # 创建带有重试策略的 session
            session = requests.Session()
            retry_strategy = Retry(
                total=2,  # 最多重试2次
                backoff_factor=1,  # 重试间隔
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["POST"]
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # 增加超时时间，因为首次使用时 Ollama 需要下载模型
            response = session.post(url, json=payload, timeout=300)  # 5分钟超时
            response.raise_for_status()
            result = response.json()
            response_text = result.get("response", "").strip()
            
            # 检查是否有错误信息
            if "error" in result:
                error_msg = result.get("error", "")
                logger.error(f"Ollama API 返回错误: {error_msg}")
                if "model" in error_msg.lower() and "not found" in error_msg.lower():
                    logger.warning(
                        f"模型 '{model}' 未找到。Ollama 会在首次使用时自动下载模型。\n"
                        f"请确保：\n"
                        f"1. Ollama 服务正在运行（运行 `ollama serve` 或通过桌面应用启动）\n"
                        f"2. 模型名称正确（例如：qwen2.5:3b, llama3.2:3b 等）\n"
                        f"3. 或者手动运行 `ollama pull {model}` 来下载模型"
                    )
                return None
            
            return response_text
        except ImportError:
            logger.error("requests 库未安装，请运行: pip install requests")
            return None
        except requests.exceptions.ConnectionError as e:
            error_msg = str(e)
            if "10054" in error_msg or "ConnectionResetError" in str(type(e).__name__):
                logger.error(
                    f"连接被重置，无法连接到 Ollama 服务（{base_url}）。\n"
                    f"可能的原因：\n"
                    f"1. Ollama 服务未启动或已崩溃\n"
                    f"2. 防火墙或安全软件阻止了连接\n"
                    f"3. 端口 {base_url.split(':')[-1] if ':' in base_url else '11434'} 被占用\n"
                    f"解决方案：\n"
                    f"1. 检查 Ollama 服务是否运行（运行 `ollama serve` 或重启桌面应用）\n"
                    f"2. 检查 Base URL 是否正确（默认: http://localhost:11434）\n"
                    f"3. 尝试在命令行运行 `ollama run {model}` 测试服务是否正常"
                )
            else:
                logger.error(
                    f"无法连接到 Ollama 服务（{base_url}）。\n"
                    f"请确保：\n"
                    f"1. Ollama 已安装（从 https://ollama.ai 下载）\n"
                    f"2. Ollama 服务正在运行（运行 `ollama serve` 或通过桌面应用启动）\n"
                    f"3. Base URL 正确（默认: http://localhost:11434）"
                )
            return None
        except requests.exceptions.Timeout:
            logger.warning(
                f"Ollama API 调用超时。这可能是因为：\n"
                f"1. 模型正在首次下载中（可能需要几分钟）\n"
                f"2. 模型太大，推理时间较长\n"
                f"请稍后重试或手动运行 `ollama pull {model}` 来预先下载模型"
            )
            return None
        except requests.exceptions.ChunkedEncodingError:
            logger.warning(
                f"Ollama API 响应不完整（连接中断）。这可能是因为：\n"
                f"1. 网络连接不稳定\n"
                f"2. Ollama 服务在处理长文本时连接中断\n"
                f"3. 模型推理过程中连接被重置\n"
                f"请检查网络连接或稍后重试"
            )
            return None
        except requests.exceptions.RequestException as e:
            logger.error(
                f"Ollama API 请求失败: {str(e)}\n"
                f"请检查：\n"
                f"1. Ollama 服务是否正常运行\n"
                f"2. 网络连接是否正常\n"
                f"3. Base URL 和模型名称是否正确"
            )
            return None
        except Exception as exc:
            logger.error(f"Ollama API 调用失败: {exc}", exc_info=True)
            return None

    def _call_custom_api(
        self,
        prompt: str,
        base_url: Optional[str],
        api_key: Optional[str],
        model: str,
    ) -> Optional[str]:
        """调用自定义 API（兼容 OpenAI 格式）"""
        try:
            import requests
            if not base_url:
                logger.warning("自定义 API base_url 未提供")
                return None
            
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个专业的文本纠错助手。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000,
            }
            
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except ImportError:
            logger.error("requests 库未安装，请运行: pip install requests")
            return None
        except Exception as exc:
            logger.error(f"自定义 API 调用失败: {exc}", exc_info=True)
            return None

    def _extract_correction_records(self, original: str, corrected: str) -> List[CorrectionRecord]:
        """从原始文本和纠正后的文本中提取替换记录（简化版）"""
        records = []
        if original == corrected:
            return records
        
        # 简单的差异检测（实际可以使用更复杂的算法）
        orig_tokens = self._tokenize(original)
        corr_tokens = self._tokenize(corrected)
        
        # 找出不同的 token
        orig_set = set(orig_tokens)
        corr_set = set(corr_tokens)
        added = corr_set - orig_set
        removed = orig_set - corr_set
        
        # 简单的匹配逻辑
        for removed_token in removed:
            best_match = None
            best_score = 0.0
            for added_token in added:
                score = self._similarity(removed_token, added_token)
                if score > best_score and score > 0.8:
                    best_match = added_token
                    best_score = score
            
            if best_match:
                records.append(CorrectionRecord(
                    original=removed_token,
                    corrected=best_match,
                    score=best_score,
                    reference="LLM 纠错"
                ))
        
        return records

    # ---------- Internal helpers ----------
    def _ensure_collection(self):
        if self._client is not None and self._collection is not None:
            return
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _reset_collection(self):
        self._ensure_collection()
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _ensure_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model_name, device="cpu")

    def _embed(self, documents: List[str]) -> List[List[float]]:
        self._ensure_embedder()
        embeddings = self._embedder.encode(documents, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()

    def _read_text_file(self, file_path: str) -> Optional[str]:
        encodings = ("utf-8", "utf-8-sig", "gbk")
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as exc:
                logger.warning(f"读取文件失败 {file_path}: {exc}")
                return None
        logger.warning(f"无法使用常见编码读取文件：{file_path}")
        return None

    def _split_document(self, text: str, file_path: str) -> List[str]:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in (".md", ".markdown"):
            return self._split_markdown(text)
        return self._split_text(text)

    def _split_text(self, text: str) -> List[str]:
        normalized = text.replace("\r\n", "\n")
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized) if p.strip()]
        chunks: List[str] = []

        for paragraph in paragraphs:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + self.chunk_size)
                chunk = paragraph[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(paragraph):
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks

    def _split_markdown(self, text: str) -> List[str]:
        cleaned = self._normalize_markdown(text)
        sections: List[Tuple[Optional[str], str]] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []

        def flush_section():
            if not current_lines:
                return
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_title, body))

        for line in cleaned.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                flush_section()
                current_title = stripped.lstrip("#").strip() or None
                current_lines = []
                continue
            if stripped.startswith(("-", "*", "+")):
                stripped = stripped[1:].strip()
                if stripped:
                    current_lines.append(f"- {stripped}")
            else:
                current_lines.append(line)

        flush_section()
        if not sections and cleaned.strip():
            sections.append((None, cleaned.strip()))

        chunks: List[str] = []
        for title, body in sections:
            prefix = f"{title}\n" if title else ""
            if not body:
                if prefix:
                    chunks.append(prefix.strip())
                continue
            start = 0
            while start < len(body):
                end = min(len(body), start + self.chunk_size)
                chunk_body = body[start:end].strip()
                if chunk_body:
                    if prefix:
                        chunks.append(f"{prefix}{chunk_body}")
                    else:
                        chunks.append(chunk_body)
                if end == len(body):
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks

    def _normalize_markdown(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n")
        normalized = re.sub(r"^---\s*\n.*?\n---\s*\n", "", normalized, flags=re.S)
        normalized = self._strip_code_blocks(normalized)
        normalized = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", normalized)
        normalized = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", normalized)
        normalized = re.sub(r"<[^>]+>", " ", normalized)
        normalized = re.sub(r"\*\*(.+?)\*\*", r"\1", normalized)
        normalized = re.sub(r"__(.+?)__", r"\1", normalized)
        normalized = re.sub(r"\*(.+?)\*", r"\1", normalized)
        normalized = re.sub(r"_(.+?)_", r"\1", normalized)

        processed_lines: List[str] = []
        for line in normalized.split("\n"):
            stripped = line.strip()
            if not stripped:
                processed_lines.append("")
                continue

            heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if heading:
                processed_lines.append("")
                processed_lines.append(heading.group(2).strip())
                processed_lines.append("")
                continue

            if stripped.startswith(">"):
                stripped = stripped.lstrip(">").strip()

            if re.match(r"^(\d+\.)\s+", stripped):
                stripped = re.sub(r"^\d+\.\s+", "", stripped)
                processed_lines.append(f"- {stripped}")
                continue

            if re.match(r"^[-*+]\s+", stripped):
                stripped = re.sub(r"^[-*+]\s+", "", stripped)
                processed_lines.append(f"- {stripped}")
                continue

            if stripped.count("|") >= 2:
                processed_lines.append(self._flatten_table_row(stripped))
                continue

            processed_lines.append(stripped)

        collapsed = "\n".join(processed_lines)
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    @staticmethod
    def _flatten_table_row(row: str) -> str:
        cells = [cell.strip() for cell in row.strip("|").split("|") if cell.strip()]
        return " | ".join(cells)

    @staticmethod
    def _strip_code_blocks(text: str) -> str:
        without_fences = re.sub(r"```.*?```", "", text, flags=re.S)
        return re.sub(r"`([^`]+)`", r"\1", without_fences)

    def _tokenize(self, text: str) -> List[str]:
        return TOKEN_PATTERN.findall(text)

    def _load_meta(self) -> Optional[Dict]:
        if not os.path.exists(self.meta_path):
            return None
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_meta(self, meta: Dict):
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning(f"写入 meta 信息失败: {exc}")

    def _load_token_stats(self) -> Dict[str, int]:
        if not os.path.exists(self.token_stats_path):
            return {}
        try:
            with open(self.token_stats_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: int(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_token_stats(self):
        try:
            with open(self.token_stats_path, "w", encoding="utf-8") as f:
                json.dump(self.token_frequency, f, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"写入术语统计失败: {exc}")

    def _is_index_ready(self) -> bool:
        meta = self._load_meta()
        if not meta:
            return False
        kb_path = os.path.abspath(self.knowledge_dir)
        return meta.get("knowledge_dir") == kb_path and meta.get("total_chunks", 0) > 0

    def _correct_text(
        self,
        text: str,
        top_k: int,
        similarity_threshold: float,
        max_corrections: int,
    ) -> Tuple[str, List[CorrectionRecord]]:
        query_embeddings = self._embed([text])
        self._ensure_collection()
        try:
            total_entries = self._collection.count()
        except Exception:
            total_entries = 0
        if total_entries == 0:
            return text, []
        try:
            query = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=min(top_k, total_entries),
                include=["documents"],
            )
        except Exception as exc:
            logger.error(f"向量检索失败: {exc}", exc_info=True)
            return text, []

        documents = (query.get("documents") or [[]])[0]
        candidate_terms = self._collect_candidate_terms(documents)
        if not candidate_terms:
            return text, []

        replacements: List[CorrectionRecord] = []
        tokens = self._tokenize(text)
        updated_text = text

        for token in tokens:
            if token in candidate_terms:
                continue
            best_candidate, best_score = self._select_best_candidate(token, candidate_terms)
            if best_candidate and best_score >= similarity_threshold and best_candidate != token:
                replacements.append(
                    CorrectionRecord(original=token, corrected=best_candidate, score=best_score, reference="; ".join(documents[:2]))
                )
                updated_text = self._replace_token_once(updated_text, token, best_candidate)
                if len(replacements) >= max_corrections:
                    break

        return updated_text, replacements

    def _collect_candidate_terms(self, documents: Sequence[str]) -> Dict[str, int]:
        candidates: Dict[str, int] = {}
        for doc in documents:
            for token in self._tokenize(doc):
                freq = self.token_frequency.get(token, 1)
                candidates[token] = max(candidates.get(token, 0), freq)
        return candidates

    def _select_best_candidate(self, token: str, candidates: Dict[str, int]) -> Tuple[Optional[str], float]:
        best_term = None
        best_score = 0.0
        for term, freq in candidates.items():
            score = self._similarity(token, term)
            # 高频术语略微加权
            score *= 1.0 + min(freq / 100, 0.2)
            if score > best_score:
                best_term = term
                best_score = score
        return best_term, best_score

    def _similarity(self, token: str, term: str) -> float:
        if not token or not term:
            return 0.0

        lev_score = 1.0 - (self._edit_distance(token, term) / max(len(token), len(term)))
        fuzz_score = fuzz.WRatio(token, term) / 100.0
        score = max(lev_score, fuzz_score)

        if len(token) <= 3 and len(term) <= 3:
            dist = self._edit_distance(token, term)
            if dist == 1:
                score = max(score, 0.92)
        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _replace_token_once(text: str, old: str, new: str) -> str:
        idx = text.find(old)
        if idx == -1:
            return text
        return text[:idx] + new + text[idx + len(old):]

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, lb + 1):
                temp = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
        return dp[lb]

