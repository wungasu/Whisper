"""
人脸搜索模块

提供基于 InsightFace 和 ChromaDB 的人脸搜索功能。
"""

from modules.face_search.service import FaceSearchService, SUPPORTED_IMAGE_EXTENSIONS

__all__ = ['FaceSearchService', 'SUPPORTED_IMAGE_EXTENSIONS']

