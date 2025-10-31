import numpy as np
from config import (DEFAULT_TOPK,
                    DEFAULT_THRESHOLD,
                    DEFAULT_MODEL_PATH,
                    DEFAULT_MODEL_NAME)
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def initialize_model(model_name=DEFAULT_MODEL_NAME,model_path=DEFAULT_MODEL_PATH):
    model = SentenceTransformer(model_name, cache_folder=model_path)
    return model


def get_topk_similar_sklearn(A, B, topk=DEFAULT_TOPK, threshold=DEFAULT_THRESHOLD):
    # 转换为numpy数组（A需要是二维数组，B需要是二维数组）
    A_np = np.array(A)  # 形状：(n, d)，n是向量数量，d是维度
    B_np = np.array([B])  # 形状：(1, d)，转为二维数组以适配sklearn
    
    # 批量计算A中所有向量与B的相似度（结果是形状为(n, 1)的数组）
    similarities = cosine_similarity(A_np, B_np).flatten()  # 展平为一维数组
    
    # 过滤并记录索引
    valid_pairs = [
        (idx, sim) for idx, sim in enumerate(similarities)
        if sim >= threshold
    ]
    
    # 排序并取TopK
    valid_pairs.sort(key=lambda x: x[1], reverse=True)
    return valid_pairs[:topk]

def get_paper_by_embedding(model, all_papers_list, target_title, target_summary, topk=DEFAULT_TOPK, threshold=DEFAULT_THRESHOLD):
    # 计算所有论文的嵌入
    all_embeddings = model.encode(all_papers_list)
    target_paper = paper_cat = str('Title: ' + target_title + '\n Summary: ' + target_summary)
    target_embedding = model.encode(target_paper)
            
    # 进行相似度搜索
    result = get_topk_similar_sklearn(
        all_embeddings, 
        target_embedding, 
        topk=topk,  
        threshold=threshold)
    return result