"""
检索引擎 - 基于TF-IDF的文档检索（纯numpy实现，无需额外API）
"""
import os
import json
import re
import math
import numpy as np
from config import DOCS_DIR, VECTOR_STORE_DIR


class EmbeddingManager:
    """TF-IDF文档检索引擎"""

    def __init__(self):
        self.documents = []
        self.metadata = []
        self.vocabulary = {}
        self.idf = None
        self.tfidf_matrix = None
        self.store_path = os.path.join(VECTOR_STORE_DIR, "tfidf_store.npz")

    def _tokenize(self, text):
        """中文分词：按字+按词混合"""
        text = text.lower()
        chars = list(re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text))
        bigrams = []
        for i in range(len(chars) - 1):
            bigrams.append(chars[i] + chars[i+1])
        return chars + bigrams

    def build_index(self):
        """构建TF-IDF索引"""
        print("[1] 加载知识库文档...")
        docs = self._load_documents()

        print(f"[2] 共加载 {len(docs)} 个文档片段，开始构建TF-IDF索引...")
        self.documents = [d["content"] for d in docs]
        self.metadata = [{"source": d["source"], "title": d["title"]} for d in docs]

        tokenized_docs = [self._tokenize(doc) for doc in self.documents]

        all_tokens = set()
        for tokens in tokenized_docs:
            all_tokens.update(tokens)
        self.vocabulary = {token: idx for idx, token in enumerate(sorted(all_tokens))}

        vocab_size = len(self.vocabulary)
        num_docs = len(tokenized_docs)

        tf_matrix = np.zeros((num_docs, vocab_size))
        for i, tokens in enumerate(tokenized_docs):
            for token in tokens:
                if token in self.vocabulary:
                    tf_matrix[i][self.vocabulary[token]] += 1

        tf_matrix = tf_matrix / (tf_matrix.sum(axis=1, keepdims=True) + 1e-10)

        df = np.sum(tf_matrix > 0, axis=0)
        self.idf = np.log((num_docs + 1) / (df + 1)) + 1

        self.tfidf_matrix = tf_matrix * self.idf

        norms = np.linalg.norm(self.tfidf_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.tfidf_matrix = self.tfidf_matrix / norms

        print(f"[3] TF-IDF索引构建完成: {num_docs}文档 × {vocab_size}词汇")
        self._save_store()
        return num_docs

    def search(self, query, top_k=5):
        """检索最相关的文档片段"""
        if self.tfidf_matrix is None:
            self._load_store()

        query_tokens = self._tokenize(query)
        query_vec = np.zeros(len(self.vocabulary))
        for token in query_tokens:
            if token in self.vocabulary:
                query_vec[self.vocabulary[token]] += 1

        query_vec = query_vec / (query_vec.sum() + 1e-10)
        query_vec = query_vec * self.idf
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        similarities = np.dot(self.tfidf_matrix, query_vec)

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                results.append({
                    "content": self.documents[idx],
                    "source": self.metadata[idx]["source"],
                    "title": self.metadata[idx]["title"],
                    "score": float(similarities[idx])
                })
        return results

    def _load_documents(self):
        """加载知识库文档，按段落切分"""
        docs = []
        for filename in sorted(os.listdir(DOCS_DIR)):
            if not filename.endswith(".md"):
                continue
            filepath = os.path.join(DOCS_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = self._split_text(content, max_length=500)
            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 20:
                    continue
                docs.append({
                    "content": chunk,
                    "source": filename,
                    "title": filename.replace(".md", "")
                })
        return docs

    def _split_text(self, text, max_length=500):
        """按段落切分文本"""
        paragraphs = text.split("\n## ")
        chunks = []
        for para in paragraphs:
            if len(para) > max_length:
                lines = para.split("\n")
                current = ""
                for line in lines:
                    if len(current) + len(line) > max_length and current:
                        chunks.append(current.strip())
                        current = line
                    else:
                        current = current + "\n" + line if current else line
                if current.strip():
                    chunks.append(current.strip())
            else:
                if para.strip():
                    chunks.append(para.strip())
        return chunks

    def _save_store(self):
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
        np.savez(self.store_path,
                 tfidf_matrix=self.tfidf_matrix,
                 idf=self.idf,
                 documents=json.dumps(self.documents, ensure_ascii=False),
                 metadata=json.dumps(self.metadata, ensure_ascii=False),
                 vocabulary=json.dumps(self.vocabulary, ensure_ascii=False))

    def _load_store(self):
        if not os.path.exists(self.store_path):
            raise FileNotFoundError("TF-IDF索引不存在，请先运行 build_index()")
        data = np.load(self.store_path, allow_pickle=True)
        self.tfidf_matrix = data["tfidf_matrix"]
        self.idf = data["idf"]
        self.documents = json.loads(str(data["documents"]))
        self.metadata = json.loads(str(data["metadata"]))
        self.vocabulary = json.loads(str(data["vocabulary"]))


if __name__ == "__main__":
    em = EmbeddingManager()
    count = em.build_index()
    print(f"\n知识库索引完成，共 {count} 个文档片段")

    print("\n测试检索:")
    queries = [
        "Ty1基因和Ty3基因同时携带的材料抗病性如何",
        "糖度大于10的品种怎么评价",
        "杂交组合选配原则",
    ]
    for q in queries:
        print(f"\n查询: {q}")
        results = em.search(q, top_k=3)
        for r in results:
            print(f"  [{r['score']:.4f}] {r['source']}")
            print(f"  {r['content'][:100]}...")
            print()
