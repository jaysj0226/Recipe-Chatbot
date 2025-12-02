"""Hybrid Retriever - Combining Dense (Vector) + Sparse (BM25) Search with RRF"""
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
import re
import pickle
import os
from pathlib import Path

from rank_bm25 import BM25Okapi
from konlpy.tag import Okt

from utils.vectorstore import get_vectorstore
from config.settings import DEBUG_RAW, BASE_DIR


class KoreanTokenizer:
    """한국어 형태소 분석 기반 토크나이저"""
    def __init__(self):
        self.okt = Okt()

    def tokenize(self, text: str) -> List[str]:
        """
        한국어 텍스트를 형태소 단위로 분리

        Args:
            text: 분리할 텍스트

        Returns:
            형태소 리스트
        """
        if not text or not isinstance(text, str):
            return []

        # 소문자 변환 및 정규화
        text = text.lower().strip()

        # 형태소 분석 (명사, 동사, 형용사만 추출)
        try:
            morphs = self.okt.morphs(text, stem=True)  # stem=True로 어간 추출
            return morphs
        except Exception as e:
            if DEBUG_RAW:
                print(f"Tokenization error: {e}")
            # Fallback: 단순 공백 분리
            return text.split()


class HybridRetriever:
    """
    Hybrid Retrieval: Dense (Vector) + Sparse (BM25) 검색 결합

    - Dense: OpenAI Embeddings로 의미적 유사도 검색
    - Sparse: BM25로 키워드 정확도 검색
    - Fusion: Reciprocal Rank Fusion (RRF)으로 결과 병합
    """

    def __init__(self, vectorstore=None, cache_dir=None):
        """
        Args:
            vectorstore: Chroma vectorstore instance (optional, will use get_vectorstore if None)
            cache_dir: Directory to cache BM25 index (optional, defaults to BASE_DIR / "bm25_cache")
        """
        self.vectorstore = vectorstore or get_vectorstore()
        self.tokenizer = KoreanTokenizer()

        # 캐시 디렉토리 설정
        if cache_dir is None:
            self.cache_dir = Path(BASE_DIR) / "bm25_cache"
        else:
            self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.cache_file = self.cache_dir / "bm25_index.pkl"

        # BM25 인덱스 (lazy loading)
        self._bm25_index = None
        self._bm25_docs = None
        self._bm25_metas = None

    def _load_from_cache(self) -> bool:
        """캐시에서 BM25 인덱스 로드"""
        if not self.cache_file.exists():
            return False

        try:
            if DEBUG_RAW:
                print(f"Loading BM25 index from cache: {self.cache_file}")

            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)

            self._bm25_index = cached_data['index']
            self._bm25_docs = cached_data['docs']
            self._bm25_metas = cached_data['metas']

            if DEBUG_RAW:
                print(f"BM25 index loaded from cache ({len(self._bm25_docs)} documents)")

            return True

        except Exception as e:
            if DEBUG_RAW:
                print(f"Failed to load cache: {e}")
            return False

    def _save_to_cache(self):
        """BM25 인덱스를 캐시에 저장"""
        try:
            if DEBUG_RAW:
                print(f"Saving BM25 index to cache: {self.cache_file}")

            cached_data = {
                'index': self._bm25_index,
                'docs': self._bm25_docs,
                'metas': self._bm25_metas
            }

            with open(self.cache_file, 'wb') as f:
                pickle.dump(cached_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            if DEBUG_RAW:
                print("BM25 index saved to cache")

        except Exception as e:
            if DEBUG_RAW:
                print(f"Failed to save cache: {e}")

    def _build_bm25_index(self):
        """BM25 인덱스 구축 (캐시 사용, 없으면 새로 생성)"""
        if self._bm25_index is not None:
            return  # Already built

        # 1. 캐시에서 로드 시도
        if self._load_from_cache():
            return

        # 2. 캐시 없으면 새로 빌드
        if DEBUG_RAW:
            print("Building BM25 index from scratch...")

        try:
            # Vector DB에서 모든 문서 가져오기
            collection = getattr(self.vectorstore, "_collection", None)
            if collection is None:
                if DEBUG_RAW:
                    print("No collection found, BM25 disabled")
                self._bm25_index = None
                return

            # Chroma에서 모든 문서 가져오기
            all_data = collection.get(include=["documents", "metadatas"])

            if not all_data or not all_data.get("documents"):
                if DEBUG_RAW:
                    print("No documents found, BM25 disabled")
                self._bm25_index = None
                return

            self._bm25_docs = all_data["documents"]
            self._bm25_metas = all_data.get("metadatas", [{}] * len(self._bm25_docs))

            # 토크나이징
            if DEBUG_RAW:
                print(f"Tokenizing {len(self._bm25_docs)} documents...")

            tokenized_corpus = [self.tokenizer.tokenize(doc) for doc in self._bm25_docs]

            # BM25 인덱스 생성
            self._bm25_index = BM25Okapi(tokenized_corpus)

            if DEBUG_RAW:
                print(f"BM25 index built with {len(self._bm25_docs)} documents")

            # 3. 캐시에 저장
            self._save_to_cache()

        except Exception as e:
            if DEBUG_RAW:
                print(f"BM25 index build error: {e}")
            self._bm25_index = None

    def _bm25_search(self, query: str, k: int = 10) -> List[Tuple[str, Dict, float]]:
        """
        BM25 검색

        Args:
            query: 검색 쿼리
            k: 반환할 상위 k개 문서

        Returns:
            List of (document_text, metadata, bm25_score)
        """
        # BM25 인덱스 빌드 (최초 1회)
        self._build_bm25_index()

        if self._bm25_index is None:
            return []

        # 쿼리 토크나이징
        tokenized_query = self.tokenizer.tokenize(query)

        if not tokenized_query:
            return []

        # BM25 점수 계산
        scores = self._bm25_index.get_scores(tokenized_query)

        # Top-k 인덱스 추출
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_k_indices:
            doc = self._bm25_docs[idx]
            meta = self._bm25_metas[idx] if idx < len(self._bm25_metas) else {}
            score = float(scores[idx])
            results.append((doc, meta, score))

        return results

    def _vector_search(self, query: str, k: int = 10) -> List[Tuple[str, Dict, float]]:
        """
        Vector 검색

        Args:
            query: 검색 쿼리
            k: 반환할 상위 k개 문서

        Returns:
            List of (document_text, metadata, similarity_score)
        """
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # 결과 변환: (Document, distance) -> (text, metadata, similarity)
            formatted = []
            for doc, distance in results:
                text = doc.page_content
                meta = getattr(doc, "metadata", {}) or {}
                similarity = 1.0 - float(distance)  # distance를 similarity로 변환
                formatted.append((text, meta, similarity))

            return formatted

        except Exception as e:
            if DEBUG_RAW:
                print(f"Vector search error: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[str, Dict, float]],
        sparse_results: List[Tuple[str, Dict, float]],
        k: int = 10,
        alpha: float = 0.5,
        k_rrf: int = 60
    ) -> List[Tuple[str, Dict, float]]:
        """
        Reciprocal Rank Fusion (RRF) 알고리즘

        RRF Score = alpha * (1 / (k_rrf + rank_dense)) + (1-alpha) * (1 / (k_rrf + rank_sparse))

        Args:
            dense_results: Vector 검색 결과
            sparse_results: BM25 검색 결과
            k: 최종 반환할 문서 수
            alpha: Dense/Sparse 가중치 (0.5 = 동등, 1.0 = dense only)
            k_rrf: RRF 상수 (일반적으로 60)

        Returns:
            Fused results: List of (document_text, metadata, rrf_score)
        """
        # 문서 식별을 위한 키 생성 함수
        def _doc_key(text: str, meta: Dict) -> str:
            """문서를 고유하게 식별하는 키 생성"""
            # 텍스트 일부 + URL/제목 조합
            text_sig = text[:200] if text else ""
            url = meta.get("url", "") or meta.get("source", "")
            title = meta.get("title", "") or meta.get("name", "")
            return f"{url}|{title}|{hash(text_sig)}"

        # 각 문서의 랭킹 저장
        doc_ranks = {}  # key -> {"dense_rank": int, "sparse_rank": int, "text": str, "meta": dict}

        # Dense 결과 처리
        for rank, (text, meta, score) in enumerate(dense_results, start=1):
            key = _doc_key(text, meta)
            if key not in doc_ranks:
                doc_ranks[key] = {"text": text, "meta": meta, "dense_rank": None, "sparse_rank": None}
            doc_ranks[key]["dense_rank"] = rank

        # Sparse 결과 처리
        for rank, (text, meta, score) in enumerate(sparse_results, start=1):
            key = _doc_key(text, meta)
            if key not in doc_ranks:
                doc_ranks[key] = {"text": text, "meta": meta, "dense_rank": None, "sparse_rank": None}
            doc_ranks[key]["sparse_rank"] = rank

        # RRF 점수 계산
        fused = []
        for key, info in doc_ranks.items():
            dense_rank = info["dense_rank"] or 1000  # 없으면 낮은 순위
            sparse_rank = info["sparse_rank"] or 1000

            # RRF 공식
            rrf_score = (
                alpha * (1.0 / (k_rrf + dense_rank)) +
                (1.0 - alpha) * (1.0 / (k_rrf + sparse_rank))
            )

            fused.append((info["text"], info["meta"], rrf_score))

        # RRF 점수로 정렬
        fused.sort(key=lambda x: x[2], reverse=True)

        # Top-k 반환
        return fused[:k]

    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5,
        k_rrf: int = 60,
        fetch_k: int = None
    ) -> List[Tuple[str, Dict, float]]:
        """
        Hybrid Search: Dense + Sparse 검색 결합

        Args:
            query: 검색 쿼리
            k: 최종 반환할 문서 수
            alpha: Dense/Sparse 가중치 (0.5 = 동등 가중)
            k_rrf: RRF 상수
            fetch_k: Dense/Sparse 각각에서 가져올 문서 수 (default: k * 2)

        Returns:
            List of (document_text, metadata, rrf_score)
        """
        if not query or not query.strip():
            return []

        # Fetch more candidates for fusion
        if fetch_k is None:
            fetch_k = k * 2

        # Dense 검색
        dense_results = self._vector_search(query, k=fetch_k)

        # Sparse 검색
        sparse_results = self._bm25_search(query, k=fetch_k)

        # RRF Fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, k=k, alpha=alpha, k_rrf=k_rrf
        )

        if DEBUG_RAW:
            print(f"Hybrid search: Dense={len(dense_results)}, Sparse={len(sparse_results)}, Fused={len(fused_results)}")

        return fused_results


@lru_cache(maxsize=1)
def get_hybrid_retriever():
    """캐싱된 Hybrid Retriever 인스턴스 반환"""
    return HybridRetriever()
