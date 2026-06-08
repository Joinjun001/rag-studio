import os
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class MemoryVectorDB:
    def __init__(self):
        # Documents 구조: {"text": str, "source": str, "vector": np.ndarray} 리스트
        self.documents = []
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )

    def add_document(self, text: str, source: str = "User Upload"):
        if not text.strip():
            return None
        # 임베딩 벡터 생성
        vector = self.embeddings_model.embed_query(text)
        doc = {
            "text": text,
            "source": source,
            "vector": np.array(vector)
        }
        self.documents.append(doc)
        return doc

    def clear(self):
        self.documents = []

    def similarity_search(self, query: str, k: int = 3):
        if not self.documents:
            return [], None
        
        # 질문 임베딩 생성
        query_vector = np.array(self.embeddings_model.embed_query(query))
        
        results = []
        for doc in self.documents:
            doc_vector = doc["vector"]
            # 코사인 유사도 계산
            dot_product = np.dot(query_vector, doc_vector)
            norm_q = np.linalg.norm(query_vector)
            norm_d = np.linalg.norm(doc_vector)
            similarity = dot_product / (norm_q * norm_d) if (norm_q * norm_d) > 0 else 0.0
            
            results.append({
                "text": doc["text"],
                "source": doc["source"],
                "vector": doc_vector,
                "similarity": float(similarity)
            })
            
        # 유사도 기준 내림차순 정렬
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k], query_vector

    def get_visualization_coordinates(self, query_vector: np.ndarray = None):
        """
        768차원 고차원 벡터들을 PCA(SVD) 알고리즘을 사용해 2차원 평면 좌표(x, y)로 압축합니다.
        문서 좌표 리스트와 질문 좌표를 함께 반환합니다.
        """
        if not self.documents:
            return [], None

        vectors = [doc["vector"] for doc in self.documents]
        has_query = query_vector is not None
        
        if has_query:
            vectors.append(query_vector)
            
        X = np.array(vectors)
        
        # 데이터가 1개뿐인 경우 PCA 수행 불가하므로 원점에 배치
        if X.shape[0] < 2:
            single_coord = {"x": 0.0, "y": 0.0}
            doc_coords = [{
                "text": self.documents[0]["text"],
                "source": self.documents[0]["source"],
                "x": 0.0,
                "y": 0.0
            }]
            return doc_coords, (single_coord if has_query else None)
            
        # 데이터 센터링 (평균을 0으로 맞춤)
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        
        # SVD(특이값 분해)를 활용한 차원 축소
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 상위 2개의 주성분(Principal Components)으로 투영
        X_2d = np.dot(X_centered, Vt[:2].T)
        
        # 시각화 격자(-10에서 10)에 맞게 좌표 스케일링
        max_val = np.max(np.abs(X_2d))
        if max_val > 0:
            X_2d = (X_2d / max_val) * 8.0  # 여백을 위해 8.0 곱함
            
        doc_coords = []
        for i in range(len(self.documents)):
            doc_coords.append({
                "text": self.documents[i]["text"],
                "source": self.documents[i]["source"],
                "x": round(float(X_2d[i, 0]), 3),
                "y": round(float(X_2d[i, 1]), 3)
            })
            
        query_coord = None
        if has_query:
            query_coord = {
                "x": round(float(X_2d[-1, 0]), 3),
                "y": round(float(X_2d[-1, 1]), 3)
            }
            
        return doc_coords, query_coord

# 글로벌 싱글톤 벡터 DB 인스턴스 생성
vdb = MemoryVectorDB()
