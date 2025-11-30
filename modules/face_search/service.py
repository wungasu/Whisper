import os
import json
import hashlib
import uuid
import shutil
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import cv2
from PIL import Image

from modules.utils.logger import get_logger

logger = get_logger()

# 支持的图像格式
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tiff", ".tif"}
# 最大文件大小（100MB）
MAX_FILE_SIZE = 100 * 1024 * 1024
# 最小图像尺寸
MIN_IMAGE_SIZE = 32


class FaceSearchService:
    """
    人脸搜索服务类，使用 InsightFace 进行人脸检测和特征提取，使用 ChromaDB 进行向量存储和检索。
    
    特性：
    - 支持批量添加图片到数据库
    - 基于人脸相似度的图像搜索
    - MD5 去重，避免重复索引
    - 支持 GPU/CPU 自动切换
    - 完整的错误处理和输入验证
    """
    
    def __init__(self, db_dir: str = "face_db", det_size: Tuple[int, int] = (512, 512),
                 max_faces_per_image: int = 3, flush_batch_size: int = 1024, max_image_side: int = 1600):
        """
        初始化人脸搜索服务。
        
        Args:
            db_dir: 数据库存储目录
            det_size: 人脸检测尺寸，影响检测精度和速度
            max_faces_per_image: 每张图片最多处理的人脸数量
            flush_batch_size: 批量写入数据库的批次大小
            max_image_side: 图片最大边长，超过此尺寸会自动缩放
        """
        self.db_dir = db_dir
        os.makedirs(self.db_dir, exist_ok=True)
        self.images_dir = os.path.join(self.db_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.md5_index_path = os.path.join(self.db_dir, "md5_index.json")

        # Lazy-loaded components
        self._insight_app = None
        self._chroma_client = None
        self._collection = None
        self.collection_name = "faces"
        # Performance-related settings
        self.det_size = det_size
        self.max_faces_per_image = max_faces_per_image
        self.flush_batch_size = max(64, int(flush_batch_size))
        self.max_image_side = max_image_side
        # In-memory MD5 set
        self._seen_md5s = self._load_md5_index()

    # ---------- Lazy initializers ----------
    def _ensure_insightface(self):
        if self._insight_app is not None:
            return
        from insightface.app import FaceAnalysis
        # Prefer GPU if available, fallback to CPU
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        except Exception:
            providers = ['CPUExecutionProvider']
        self._insight_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self._insight_app.prepare(ctx_id=0, det_size=self.det_size)
        logger.info("InsightFace initialized.")

    def _ensure_chromadb(self):
        if self._chroma_client is not None and self._collection is not None:
            return
        import chromadb
        from chromadb.config import Settings
        self._chroma_client = chromadb.PersistentClient(
            path=os.path.join(self.db_dir, "chroma"),
            settings=Settings(anonymized_telemetry=False)
        )
        # No embedding function: we will push precomputed embeddings
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("ChromaDB initialized.")

    # ---------- Utilities ----------
    def _load_md5_index(self) -> set:
        try:
            if os.path.exists(self.md5_index_path):
                with open(self.md5_index_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data if isinstance(data, list) else [])
        except Exception:
            pass
        return set()

    def _save_md5_index(self) -> None:
        try:
            with open(self.md5_index_path, "w", encoding="utf-8") as f:
                json.dump(list(self._seen_md5s), f, ensure_ascii=False)
        except Exception:
            # 不因索引文件写失败而影响主流程
            pass

    @staticmethod
    def _safe_remove_file(path: Optional[str]) -> None:
        if not path:
            return
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as exc:
            logger.warning(f"删除临时文件失败 {path}: {exc}")

    def _store_image_file(self, src_path: str, file_md5: Optional[str]) -> Optional[str]:
        """
        将源图像复制到持久化目录，返回新的存储路径。
        """
        try:
            os.makedirs(self.images_dir, exist_ok=True)
            ext = os.path.splitext(src_path)[1].lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                ext = ".png"
            base_name = os.path.basename(src_path) or f"{uuid.uuid4().hex}{ext}"
            stem = os.path.splitext(base_name)[0]
            dest_path = os.path.join(self.images_dir, base_name)
            counter = 1
            while os.path.exists(dest_path):
                candidate = f"{stem}_{counter}{ext}"
                dest_path = os.path.join(self.images_dir, candidate)
                counter += 1
            shutil.copy2(src_path, dest_path)
            return dest_path
        except Exception as exc:
            logger.error(f"复制文件到存储目录失败 {src_path}: {exc}", exc_info=True)
            return None

    def _reset_image_store(self):
        try:
            if os.path.exists(self.images_dir):
                shutil.rmtree(self.images_dir)
            os.makedirs(self.images_dir, exist_ok=True)
        except Exception as exc:
            logger.error(f"重置 images 目录失败: {exc}", exc_info=True)

    @staticmethod
    def _validate_image_file(image_path: str) -> Tuple[bool, Optional[str]]:
        """
        验证图像文件是否有效。
        
        Returns:
            (is_valid, error_message)
        """
        if not image_path or not isinstance(image_path, str):
            return False, "无效的文件路径"
        
        if not os.path.exists(image_path):
            return False, f"文件不存在: {image_path}"
        
        if not os.path.isfile(image_path):
            return False, f"不是文件: {image_path}"
        
        # 检查文件扩展名
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_IMAGE_EXTENSIONS:
            return False, f"不支持的图像格式: {ext}。支持的格式: {', '.join(SUPPORTED_IMAGE_EXTENSIONS)}"
        
        # 检查文件大小
        try:
            file_size = os.path.getsize(image_path)
            if file_size > MAX_FILE_SIZE:
                return False, f"文件过大: {file_size / 1024 / 1024:.2f}MB，最大支持 {MAX_FILE_SIZE / 1024 / 1024:.2f}MB"
            if file_size == 0:
                return False, "文件为空"
        except OSError as e:
            return False, f"无法读取文件大小: {e}"
        
        # 尝试读取图像以验证完整性
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                # 尝试用 PIL 读取
                try:
                    with Image.open(image_path) as pil_img:
                        pil_img.verify()
                except Exception:
                    return False, "图像文件损坏或格式不正确"
            else:
                h, w = img.shape[:2]
                if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
                    return False, f"图像尺寸过小: {w}x{h}，最小支持 {MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}"
        except Exception as e:
            return False, f"无法读取图像: {e}"
        
        return True, None

    @staticmethod
    def _compute_file_md5(path: str, chunk_size: int = 1024 * 1024) -> Optional[str]:
        """
        计算文件的 MD5 哈希值。
        
        Args:
            path: 文件路径
            chunk_size: 读取块大小
            
        Returns:
            MD5 哈希值，失败返回 None
        """
        try:
            md5 = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    md5.update(chunk)
            return md5.hexdigest()
        except Exception as e:
            logger.warning(f"计算 MD5 失败 {path}: {e}")
            return None

    @staticmethod
    def _read_image(image_path: str) -> Optional[np.ndarray]:
        """
        读取图像文件为 numpy 数组。
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            BGR 格式的图像数组，失败返回 None
        """
        if not image_path or not os.path.exists(image_path):
            logger.warning(f"图像文件不存在: {image_path}")
            return None
        
        try:
            # 使用 cv2 读取（支持中文路径）
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                # 使用 PIL 作为备选方案
                try:
                    pil = Image.open(image_path).convert("RGB")
                    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    logger.debug(f"使用 PIL 读取图像: {image_path}")
                except Exception as e:
                    logger.error(f"无法读取图像 {image_path}: {e}")
                    return None
            return img
        except Exception as e:
            logger.error(f"读取图像失败 {image_path}: {e}")
            return None

    def _extract_face_embeddings(self, image_path: str) -> List[np.ndarray]:
        """
        从图像中提取人脸特征向量。
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            人脸特征向量列表，每个向量是一个归一化的 numpy 数组
        """
        self._ensure_insightface()
        img = self._read_image(image_path)
        if img is None:
            return []
        
        # 可选缩放以提升速度（保持宽高比）
        if self.max_image_side and max(img.shape[0], img.shape[1]) > self.max_image_side:
            h, w = img.shape[:2]
            scale = self.max_image_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.debug(f"图像已缩放: {w}x{h} -> {new_w}x{new_h}")
        
        try:
            faces = self._insight_app.get(img)
        except Exception as e:
            logger.error(f"人脸检测失败 {image_path}: {e}")
            return []
        
        # 限制每张图片的人脸数量以减少索引成本
        selected = faces[: self.max_faces_per_image] if (self.max_faces_per_image and self.max_faces_per_image > 0) else faces
        
        embs: List[np.ndarray] = []
        for f in selected:
            try:
                if getattr(f, "normed_embedding", None) is not None:
                    embs.append(f.normed_embedding.astype(np.float32))
                elif getattr(f, "embedding", None) is not None:
                    # 归一化处理
                    emb = f.embedding.astype(np.float32)
                    norm = np.linalg.norm(emb) + 1e-12
                    embs.append((emb / norm).astype(np.float32))
            except Exception as e:
                logger.warning(f"提取人脸特征失败: {e}")
                continue
        
        return embs

    # ---------- Public APIs ----------
    def add_images(self, image_paths: List[str], progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[int, int, List[str]]:
        """
        将图像添加到数据库进行索引。每张检测到的人脸都会成为一个向量条目。
        
        Args:
            image_paths: 图像文件路径列表
            progress_callback: 进度回调函数，参数为 (当前索引, 总数, 当前文件路径)
            
        Returns:
            (处理的图像数量, 索引的人脸数量, 错误信息列表)
        """
        if not image_paths:
            return 0, 0, []
        
        self._ensure_chromadb()
        total_faces = 0
        processed = 0
        errors: List[str] = []
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict] = []
        documents: List[str] = []

        total = len(image_paths)
        for idx, p in enumerate(image_paths):
            try:
                # 进度回调
                if progress_callback:
                    progress_callback(idx + 1, total, p)
                
                # 验证文件
                is_valid, error_msg = self._validate_image_file(p)
                if not is_valid:
                    errors.append(f"{p}: {error_msg}")
                    logger.warning(f"跳过无效文件 {p}: {error_msg}")
                    continue
                
                # MD5 去重：若同一字节内容已导入过，则跳过
                file_md5 = self._compute_file_md5(p)
                if file_md5 and file_md5 in self._seen_md5s:
                    logger.debug(f"跳过重复文件: {p}")
                    continue

                stored_path = self._store_image_file(p, file_md5)
                if not stored_path:
                    errors.append(f"{p}: 无法复制到存储目录。")
                    continue

                embs = self._extract_face_embeddings(stored_path)
                if not embs:
                    errors.append(f"{p}: 未检测到人脸")
                    logger.debug(f"未检测到人脸: {p}")
                    self._safe_remove_file(stored_path)
                    continue

                processed += 1
                for emb in embs:
                    total_faces += 1
                    face_id = str(uuid.uuid4())
                    ids.append(face_id)
                    embeddings.append(emb.tolist())
                    metadatas.append({
                        "path": stored_path,
                        "original_path": p,
                        "md5": file_md5 or ""
                    })
                    documents.append(os.path.basename(stored_path))

                # 索引成功（有嵌入）才记录 MD5，避免把无脸图片标记为已导入
                if file_md5:
                    self._seen_md5s.add(file_md5)

                # 批量写入以控制内存
                if len(ids) >= self.flush_batch_size:
                    try:
                        self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
                        ids, embeddings, metadatas, documents = [], [], [], []
                    except Exception as e:
                        logger.error(f"批量写入数据库失败: {e}")
                        errors.append(f"批量写入失败: {e}")
                        # 清空当前批次，继续处理
                        ids, embeddings, metadatas, documents = [], [], [], []
                        
            except Exception as e:
                error_msg = f"{p}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"处理图像失败 {p}: {e}", exc_info=True)
                continue

        # 写入剩余数据
        if ids:
            try:
                self._collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            except Exception as e:
                logger.error(f"最终批量写入失败: {e}")
                errors.append(f"最终批量写入失败: {e}")

        # 持久化 MD5 索引
        self._save_md5_index()

        return processed, total_faces, errors

    def search_by_image_with_scores(self, query_image_path: str, top_k: int = 50, max_distance: float = 0.75) -> List[Tuple[str, float]]:
        """
        通过人脸相似度搜索图像。返回 (图像路径, 距离) 列表，最多 top_k 个结果。
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回的最大结果数量
            max_distance: 距离阈值（余弦空间，越小越相似），距离大于此值的结果将被过滤
            
        Returns:
            (图像路径, 距离) 列表，按距离升序排列
            
        Raises:
            ValueError: 查询图像无效或未检测到人脸
        """
        # 验证查询图像
        is_valid, error_msg = self._validate_image_file(query_image_path)
        if not is_valid:
            raise ValueError(f"查询图像无效: {error_msg}")
        
        self._ensure_chromadb()
        
        # 检查数据库是否为空
        try:
            count = self._collection.count()
            if count == 0:
                logger.warning("数据库为空，无法搜索")
                return []
        except Exception as e:
            logger.error(f"检查数据库状态失败: {e}")
            raise RuntimeError(f"数据库错误: {e}")
        
        embs = self._extract_face_embeddings(query_image_path)
        if not embs:
            raise ValueError("查询图像中未检测到人脸")
        
        # 为每张人脸查询并合并结果（每个路径保留最小距离）
        path_to_best_distance: Dict[str, float] = {}
        per_face_k = max(10, min(top_k * 2, 100))  # 每张人脸查询更多结果，然后合并去重
        
        try:
            for e in embs:
                q = self._collection.query(
                    query_embeddings=[e.tolist()],
                    n_results=per_face_k,
                    include=["metadatas", "distances"]
                )
                metas = (q.get("metadatas") or [[]])[0]
                dists = (q.get("distances") or [[]])[0]
                
                for m, d in zip(metas, dists):
                    p = (m or {}).get("path")
                    if not p or not os.path.exists(p):
                        continue
                    # 按距离阈值过滤（距离越小越相似）
                    if max_distance is not None and d is not None and d > max_distance:
                        continue
                    # 保留该路径的最佳（最小）距离
                    prev = path_to_best_distance.get(p)
                    if prev is None or (d is not None and d < prev):
                        path_to_best_distance[p] = float(d) if d is not None else float("inf")
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise RuntimeError(f"搜索过程出错: {e}")

        if not path_to_best_distance:
            return []

        # 按距离升序排序，然后截取 top_k
        ranked = sorted(path_to_best_distance.items(), key=lambda kv: kv[1])
        return ranked[:max(0, int(top_k))]

    def search_by_image(self, query_image_path: str, top_k: int = 50, max_distance: float = 0.8) -> List[str]:
        """
        向后兼容的 API，仅返回图像路径列表，使用相同的过滤和排序逻辑。
        
        Args:
            query_image_path: 查询图像路径
            top_k: 返回的最大结果数量
            max_distance: 距离阈值
            
        Returns:
            图像路径列表，按相似度降序排列
        """
        ranked = self.search_by_image_with_scores(query_image_path, top_k=top_k, max_distance=max_distance)
        return [p for p, _ in ranked]
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取数据库统计信息。
        
        Returns:
            包含统计信息的字典：
            - total_faces: 总人脸数量
            - total_images: 唯一图像数量（基于路径）
            - total_indexed_files: 已索引的文件数量（基于 MD5）
        """
        self._ensure_chromadb()
        stats = {
            "total_faces": 0,
            "total_images": 0,
            "total_indexed_files": len(self._seen_md5s)
        }
        
        try:
            count = self._collection.count()
            stats["total_faces"] = count
            
            # 获取所有元数据以统计唯一图像
            if count > 0:
                # 使用 get 方法获取所有数据（对于大数据集可能较慢，但这里用于统计）
                try:
                    all_data = self._collection.get(include=["metadatas"])
                    paths = set()
                    for meta in (all_data.get("metadatas") or []):
                        if meta and "path" in meta:
                            paths.add(meta["path"])
                    stats["total_images"] = len(paths)
                except Exception as e:
                    logger.warning(f"获取图像统计信息失败，使用估算值: {e}")
                    # 如果获取失败，使用估算：假设平均每张图有 1-2 张人脸
                    stats["total_images"] = max(1, count // 2)
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
        
        return stats
    
    def rename_indexed_image(self, old_path: str, new_path: str) -> Tuple[int, List[str]]:
        """
        将数据库中引用 old_path 的记录更新为 new_path。用于文件重命名后的同步。
        """
        self._ensure_chromadb()
        errors: List[str] = []
        try:
            data = self._collection.get(include=["metadatas", "documents", "embeddings"])
        except Exception as e:
            error_msg = f"获取数据库记录失败: {e}"
            logger.error(error_msg)
            return 0, [error_msg]
        
        normalized_old = os.path.normpath(old_path)
        ids_to_update: List[str] = []
        new_metas: List[Dict] = []
        new_docs: List[str] = []
        metadatas = data.get("metadatas") or []
        ids = data.get("ids") or []
        documents = data.get("documents") or []
        raw_embeddings = data.get("embeddings")
        embeddings = raw_embeddings if raw_embeddings is not None else []
        collected_embeddings: List[List[float]] = []
        can_use_embeddings = raw_embeddings is not None
        
        for idx, meta in enumerate(metadatas):
            if not meta or idx >= len(ids):
                continue
            path = os.path.normpath(meta.get("path", ""))
            if path != normalized_old:
                continue
            new_meta = dict(meta)
            new_meta["path"] = new_path
            ids_to_update.append(ids[idx])
            new_metas.append(new_meta)
            new_docs.append(os.path.basename(new_path))
            if can_use_embeddings and idx < len(embeddings) and embeddings[idx] is not None:
                embedding_value = embeddings[idx]
                if isinstance(embedding_value, np.ndarray):
                    embedding_value = embedding_value.tolist()
                elif isinstance(embedding_value, (list, tuple)):
                    embedding_value = list(embedding_value)
                if embedding_value is not None:
                    collected_embeddings.append(embedding_value)
                else:
                    can_use_embeddings = False
            else:
                can_use_embeddings = False
        
        if not ids_to_update:
            return 0, []
        
        try:
            update_kwargs = {
                "ids": ids_to_update,
                "metadatas": new_metas,
                "documents": new_docs,
            }
            if can_use_embeddings and len(collected_embeddings) == len(ids_to_update):
                update_kwargs["embeddings"] = collected_embeddings
            self._collection.update(**update_kwargs)
            logger.info(f"更新了 {len(ids_to_update)} 条与 {old_path} 相关的记录路径。")
            return len(ids_to_update), []
        except Exception as e:
            error_msg = f"更新数据库记录失败: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
            return 0, errors
    
    def delete_images(self, image_paths: List[str]) -> Tuple[int, List[str]]:
        """
        从数据库中删除指定图像的所有人脸记录。
        
        Args:
            image_paths: 要删除的图像路径列表
            
        Returns:
            (删除的人脸数量, 错误信息列表)
        """
        if not image_paths:
            return 0, []
        
        self._ensure_chromadb()
        deleted_count = 0
        errors: List[str] = []
        
        paths_to_remove: set = set()

        try:
            # 获取所有数据
            all_data = self._collection.get(include=["metadatas", "ids"])
            ids_to_delete: List[str] = []
            
            image_paths_set = set(os.path.normpath(p) for p in image_paths)
            
            for idx, meta in enumerate(all_data.get("metadatas", [])):
                if meta and "path" in meta:
                    stored_path = os.path.normpath(meta["path"])
                    original_path = os.path.normpath(meta.get("original_path", "")) if meta.get("original_path") else None
                    if stored_path in image_paths_set or (original_path and original_path in image_paths_set):
                        face_id = all_data.get("ids", [])[idx]
                        if face_id:
                            ids_to_delete.append(face_id)
                            paths_to_remove.add(stored_path)
            
            if ids_to_delete:
                try:
                    self._collection.delete(ids=ids_to_delete)
                    deleted_count = len(ids_to_delete)
                    logger.info(f"删除了 {deleted_count} 个人脸记录")
                    for path in paths_to_remove:
                        self._safe_remove_file(path)
                except Exception as e:
                    error_msg = f"删除操作失败: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        except Exception as e:
            error_msg = f"获取数据失败: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        return deleted_count, errors
    
    def clear_database(self) -> bool:
        """
        清空整个数据库。
        
        Returns:
            是否成功
        """
        try:
            self._ensure_chromadb()
            # 删除集合
            self._chroma_client.delete_collection(name=self.collection_name)
            # 重新创建集合
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            # 清空 MD5 索引
            self._seen_md5s.clear()
            self._save_md5_index()
            # 清理同时存储的图像
            self._reset_image_store()
            logger.info("数据库已清空")
            return True
        except Exception as e:
            logger.error(f"清空数据库失败: {e}")
            return False
    
    def remove_orphaned_entries(self) -> Tuple[int, List[str]]:
        """
        删除数据库中指向不存在文件的人脸记录（孤儿记录）。
        
        Returns:
            (删除的记录数量, 错误信息列表)
        """
        self._ensure_chromadb()
        deleted_count = 0
        errors: List[str] = []
        
        try:
            all_data = self._collection.get(include=["metadatas", "ids"])
            ids_to_delete: List[str] = []
            
            for idx, meta in enumerate(all_data.get("metadatas", [])):
                if meta and "path" in meta:
                    path = meta["path"]
                    if not os.path.exists(path):
                        face_id = all_data.get("ids", [])[idx]
                        if face_id:
                            ids_to_delete.append(face_id)
            
            if ids_to_delete:
                try:
                    self._collection.delete(ids=ids_to_delete)
                    deleted_count = len(ids_to_delete)
                    logger.info(f"删除了 {deleted_count} 个孤儿记录")
                except Exception as e:
                    error_msg = f"删除孤儿记录失败: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
        except Exception as e:
            error_msg = f"清理孤儿记录失败: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        return deleted_count, errors


