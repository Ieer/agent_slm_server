import requests
import platform
import psutil
from typing import List, Dict, Union, Optional, Literal, Tuple
from pydantic import BaseModel
import numpy as np

# 設定 API 端點
API_BASE_URL = "http://127.0.0.1:8001/v1"  # 恢復為端口 8001

# --- 系統環境檢測和資源估算函數 ---
def get_system_memory_info() -> Dict[str, float]:
    """
    獲取系統記憶體信息（單位：GB）
    
    Returns:
        Dict[str, float]: 包含總記憶體、可用記憶體、總虛擬記憶體、可用虛擬記憶體的字典
    """
    try:
        # 獲取物理記憶體信息
        mem = psutil.virtual_memory()
        total_memory = mem.total / (1024**3)  # 轉換為GB
        available_memory = mem.available / (1024**3)  # 轉換為GB
        
        # 獲取虛擬記憶體（頁面文件）信息
        swap = psutil.swap_memory()
        total_swap = swap.total / (1024**3)  # 轉換為GB
        available_swap = swap.free / (1024**3)  # 轉換為GB
        
        return {
            "total_memory": total_memory,
            "available_memory": available_memory,
            "total_swap": total_swap,
            "available_swap": available_swap,
            "total_available": available_memory + available_swap,
        }
    except Exception as e:
        print(f"獲取系統記憶體信息失敗: {str(e)}")
        # 返回一個保守的默認值
        return {
            "total_memory": 8.0,
            "available_memory": 4.0,
            "total_swap": 4.0,
            "available_swap": 2.0,
            "total_available": 6.0,
        }

def get_model_memory_requirements() -> Dict[str, float]:
    """
    獲取不同嵌入模型的預估記憶體需求（單位：GB）
    
    Returns:
        Dict[str, float]: 模型ID到記憶體需求（GB）的映射
    """
    # 基於模型大小和結構的估算值
    return {
        "embeddinggemma-300m": 5,  # 較大模型
        "paraphrase-MiniLM-L6-v2": 0.5,  # 較小模型
        # 可以添加更多模型的估計值
    }

def estimate_batch_size(model_id: str, text_length: float = 100.0) -> int:
    """
    根據模型和輸入文本長度估算合適的批次大小
    
    Args:
        model_id: 模型ID
        text_length: 每條文本的平均字符長度
        
    Returns:
        int: 建議的批次大小
    """
    # 獲取系統資源信息和模型記憶體需求
    sys_mem = get_system_memory_info()
    model_mem = get_model_memory_requirements().get(model_id, 1.0)  # 默認1GB
    
    # 計算每批次文本的記憶體需求（模型本身 + 每條文本的開銷）
    mem_per_text = 0.001 * (text_length / 100)  # 粗略估計，每100字符約1MB
    
    # 可用記憶體的60%可用於處理批次
    usable_memory = sys_mem["available_memory"] * 0.6
    
    # 估算批次大小
    if usable_memory <= model_mem:
        # 記憶體不足以載入模型，返回最小批次
        return 1
    else:
        # 可用於批次處理的記憶體
        batch_memory = usable_memory - model_mem
        # 計算批次大小
        batch_size = int(batch_memory / mem_per_text)
        # 設定合理範圍
        return max(1, min(batch_size, 128))

def select_suitable_model() -> Tuple[str, bool]:
    """
    根據系統環境自動選擇合適的模型
    
    Returns:
        Tuple[str, bool]: (模型ID, 是否使用批次處理模式)
    """
    # 獲取系統資源信息
    sys_mem = get_system_memory_info()
    model_reqs = get_model_memory_requirements()
    
    # 可用於模型的記憶體（考慮保留一些給系統）
    available_for_model = sys_mem["total_available"] * 0.7
    
    # 決定使用哪個模型和處理模式
    if available_for_model >= model_reqs.get("embeddinggemma-300m", 2.0):
        # 足夠運行大模型，可以批次處理
        return "embeddinggemma-300m", True
    elif available_for_model >= model_reqs.get("paraphrase-MiniLM-L6-v2", 0.5):
        # 足夠運行小模型，可以批次處理
        return "paraphrase-MiniLM-L6-v2", True
    else:
        # 記憶體非常有限，使用小模型並且逐個處理
        return "paraphrase-MiniLM-L6-v2", False

# --- Embedding 請求和響應模型 ---
class EmbeddingsRequest(BaseModel):
    model: str = "paraphrase-MiniLM-L6-v2"  # 預設使用較小模型
    input: Union[str, List[str]]
    encoding_format: Optional[Literal["float", "base64"]] = "float"
    user: Optional[str] = None

class EmbeddingDataItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: Union[List[float], str]  # 若 encoding_format=base64 則為字串
    model: Optional[str] = None
    dimensions: Optional[int] = None

class EmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0

class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingDataItem]
    model: str
    usage: EmbeddingsUsage = EmbeddingsUsage()

# --- Embedding API 客戶端 ---
class EmbeddingAPIClient:
    """用於調用 embedding API 的客戶端類"""
    
    def __init__(self, base_url: str = API_BASE_URL, auto_detect: bool = True):
        self.base_url = base_url
        
        if auto_detect:
            # 自動根據系統環境選擇合適的模型
            model, self.use_batching = select_suitable_model()
            self.default_model = model
            
            # 打印系統環境信息
            mem_info = get_system_memory_info()
            print(f"[系統環境] 總記憶體: {mem_info['total_memory']:.1f}GB, "
                  f"可用記憶體: {mem_info['available_memory']:.1f}GB, "
                  f"虛擬記憶體: {mem_info['total_swap']:.1f}GB")
            
            print(f"[自動選擇] 使用模型: {self.default_model}, "
                  f"批次處理: {'是' if self.use_batching else '否'}")
        else:
            # 默認使用較小模型作為安全選擇
            self.default_model = "paraphrase-MiniLM-L6-v2"
            self.use_batching = True
            
        # 設定後備模型（用於在出錯時降級）
        self.fallback_model = "paraphrase-MiniLM-L6-v2"
        # 當前批次大小
        self.current_batch_size = 64
    
    def get_available_models(self) -> List[str]:
        """獲取可用的 embedding 模型列表"""
        try:
            response = requests.get(f"{self.base_url}/models")
            if response.status_code == 200:
                data = response.json()
                # 過濾出 embedding 模型（這裡假設非 qwen 模型都是 embedding 模型）
                models = [model["id"] for model in data.get("data", []) 
                        if not model["id"].startswith("qwen")]
                return models
            else:
                print(f"獲取模型列表失敗: {response.status_code} {response.text}")
                return []
        except Exception as e:
            print(f"獲取模型列表異常: {str(e)}")
            return []
    
    def create_embeddings(
        self, 
        texts: Union[str, List[str]], 
        model: Optional[str] = None, 
        encoding_format: str = "float",
        force_full_batch: bool = False
    ) -> EmbeddingsResponse:
        """
        生成文本的 embedding
        
        Args:
            texts: 單個文本或文本列表
            model: 指定使用的模型，若不指定則使用默認模型
            encoding_format: 返回格式，"float" 或 "base64"
            force_full_batch: 是否強制使用完整批次（不分批）
            
        Returns:
            EmbeddingsResponse: 包含 embedding 結果的響應對象
        """
        if model is None:
            model = self.default_model
        
        # 將輸入轉換為列表形式
        input_texts = [texts] if isinstance(texts, str) else texts
        
        # 如果是空列表，直接返回空結果
        if not input_texts:
            return EmbeddingsResponse(data=[], model=model)
        
        # 計算輸入文本的平均長度
        avg_text_length = sum(len(t) for t in input_texts) / len(input_texts)
        
        # 不再進行批次處理，直接發送請求
        # 僅保留日誌
        if not force_full_batch and len(input_texts) > 1:
            estimated_batch_size = estimate_batch_size(model, avg_text_length)
            print(f"估計批次大小: {estimated_batch_size}，但不再使用批次處理")
        
        # 標準情況：一次性處理所有文本
        return self._call_embedding_api(input_texts, model, encoding_format)
    
    def _call_embedding_api(self, texts: List[str], model: str, encoding_format: str) -> EmbeddingsResponse:
        """內部方法：直接調用 API 生成 embeddings"""
        req_data = {
            "model": model,
            "input": texts,
            "encoding_format": encoding_format
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=req_data
            )
            
            if response.status_code == 200:
                return EmbeddingsResponse.model_validate(response.json())
            elif response.status_code == 500:
                # 檢查是否是頁面文件太小錯誤或其他記憶體相關錯誤
                error_text = response.text.lower()
                memory_error_indicators = [
                    "页面文件太小", "page file", "osError", "1455", 
                    "memory", "out of memory", "insufficient memory", 
                    "cannot allocate", "allocation failed"
                ]
                
                is_memory_error = any(indicator in error_text for indicator in memory_error_indicators)
                
                if is_memory_error:
                    # 內存錯誤的處理策略
                    print("警告: 檢測到內存錯誤。正在進行降級處理...")
                    
                    # 降級策略：如果不是後備模型，切換到後備模型
                    if model != self.fallback_model:
                        print(f"降級策略: 從 {model} 切換到較小的模型 {self.fallback_model}")
                        return self._call_embedding_api(texts, self.fallback_model, encoding_format)
                    
                    # 如果已經是後備模型，直接拋出錯誤
                    raise Exception(f"使用後備模型 {self.fallback_model} 仍然遇到記憶體錯誤")
                else:
                    # 非內存錯誤
                    error_msg = f"Embedding 請求失敗: {response.status_code} {response.text}"
                    print(error_msg)
                    raise Exception(error_msg)
            else:
                error_msg = f"Embedding 請求失敗: {response.status_code} {response.text}"
                print(error_msg)
                raise Exception(error_msg)
                
        except requests.exceptions.RequestException as e:
            print(f"請求異常: {str(e)}")
            raise
        except Exception as e:
            if isinstance(e, Exception) and str(e).startswith("所有降級策略都失敗"):
                # 已經嘗試了所有降級策略
                raise
            
            print(f"Embedding 請求異常: {str(e)}")
            error_str = str(e).lower()
            
            # 檢查是否為內存錯誤
            memory_error_indicators = [
                "页面文件太小", "page file", "osError", "1455", 
                "memory", "out of memory", "insufficient memory", 
                "cannot allocate", "allocation failed"
            ]
            
            if any(indicator in error_str for indicator in memory_error_indicators):
                # 降級邏輯 - 僅切換模型
                if model != self.fallback_model:
                    print(f"降級: 切換到較小的模型 {self.fallback_model}")
                    return self._call_embedding_api(texts, self.fallback_model, encoding_format)
                else:
                    # 如果已經是後備模型，直接拋出錯誤
                    print(f"使用後備模型 {self.fallback_model} 仍然遇到記憶體錯誤")
                    raise
            
            # 其他類型的錯誤
            raise
    
    # 已移除 _process_embeddings_in_batches 和 _process_embeddings_one_by_one 方法
    # 不再支援批次處理和逐個處理模式
    
    def calculate_similarity(self, query: str, documents: List[str], model: Optional[str] = None) -> List[float]:
        """
        計算查詢和文檔之間的餘弦相似度
        
        Args:
            query: 查詢文本
            documents: 文檔列表
            model: 使用的模型，若不指定則使用默認模型
            
        Returns:
            List[float]: 相似度列表
        """
        # 獲取 query 的 embedding
        query_resp = self.create_embeddings(query, model)
        if not query_resp.data:
            raise Exception("獲取查詢 embedding 失敗")
        query_embedding = np.array(query_resp.data[0].embedding)
        
        # 獲取 documents 的 embeddings
        docs_resp = self.create_embeddings(documents, model)
        if not docs_resp.data:
            raise Exception("獲取文檔 embedding 失敗")
        
        # 計算相似度
        similarities = []
        for item in docs_resp.data:
            doc_embedding = np.array(item.embedding)
            # 計算餘弦相似度
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(float(similarity))
        
        return similarities

# --- 主測試函數 ---
def main():
    # 顯示運行環境信息
    print("===== 自適應 Embedding API 測試 =====")
    print(f"操作系統: {platform.system()} {platform.version()}")
    print(f"Python 版本: {platform.python_version()}")
    
    # 建立自適應客戶端
    print("\n初始化自適應 Embedding API 客戶端...")
    client = EmbeddingAPIClient(auto_detect=True)
    
    # 1. 獲取可用模型
    print("\n獲取可用的 embedding 模型...")
    models = client.get_available_models()
    print(f"可用模型: {models}")
    
    # 2. 測試單個文本 embedding
    print("\n測試單個文本 embedding...")
    single_text = "這是一個測試文本，用於生成 embedding。"
    try:
        single_resp = client.create_embeddings(single_text)
        print(f"單個文本 embedding 維度: {len(single_resp.data[0].embedding)}")
        print(f"前 5 個值: {single_resp.data[0].embedding[:5]}")
    except Exception as e:
        print(f"單個文本測試失敗: {str(e)}")

# 簡化的模型切換測試
def test_model_switch():
    """簡化版：測試模型切換功能"""
    print("\n=== 測試模型切換功能 ===")
    
    # 建立自適應客戶端
    client = EmbeddingAPIClient(auto_detect=True)
    print(f"默認使用模型: {client.default_model}")
    
    # 獲取可用模型
    models = client.get_available_models()
    print(f"可用模型列表: {models}")
    
    # 測試錯誤降級功能
    print("\n測試模型切換和錯誤降級功能...")
    
    # 保存原來的方法
    original_call_api = client._call_embedding_api
    
    # 模擬 API 調用，大模型拋出錯誤
    def mock_call_api(texts, model, encoding_format):
        if model == "embeddinggemma-300m":
            print(f"模擬使用 {model} 時發生記憶體錯誤...")
            raise OSError("页面文件太小，无法完成操作。 (os error 1455)")
        else:
            # 模擬成功的返回結果
            print(f"成功使用 {model} 模型")
            data_items = []
            for i, _ in enumerate(texts if isinstance(texts, list) else [texts]):
                embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 20  # 假設的 embedding 向量
                data_items.append(
                    EmbeddingDataItem(index=i, embedding=embedding, model=model, dimensions=len(embedding))
                )
            return EmbeddingsResponse(data=data_items, model=model)
    
    try:
        # 替換方法
        client._call_embedding_api = mock_call_api
        
        # 測試文本
        test_text = "這是一個用來測試模型切換的文本"
        
        # 強制使用大模型，預期會自動降級到小模型
        print("\n1. 強制使用大模型，預期自動降級:")
        client.default_model = "embeddinggemma-300m"
        resp = client.create_embeddings(test_text)
        print(f"結果使用模型: {resp.model}")
        print(f"向量前5個值: {resp.data[0].embedding[:5]}")
        
        # 直接使用小模型，預期成功
        print("\n2. 直接使用小模型:")
        client.default_model = "paraphrase-MiniLM-L6-v2"
        resp = client.create_embeddings(test_text)
        print(f"結果使用模型: {resp.model}")
        print(f"向量前5個值: {resp.data[0].embedding[:5]}")
        
    finally:
        # 恢復原方法
        client._call_embedding_api = original_call_api

if __name__ == "__main__":
    # 簡化的主要測試
    def simplified_test():
        print("===== 簡化版 Embedding API 測試 =====")
        
        # 建立自適應客戶端
        print("\n初始化自適應 Embedding API 客戶端...")
        client = EmbeddingAPIClient(auto_detect=True)
        
        # 測試模型切換
        print("\n測試模型切換功能...")
        
        # 獲取可用模型
        models = client.get_available_models()
        print(f"可用模型: {models}")
        
        # 分別使用不同的模型生成 embedding
        test_text = "這是一個簡單的測試文本。"
        
        # 預設模型
        default_model = client.default_model
        print(f"\n使用預設模型 {default_model} 生成 embedding:")
        default_resp = client.create_embeddings(test_text)
        print(f"向量維度: {len(default_resp.data[0].embedding)}")
        print(f"前 5 個值: {default_resp.data[0].embedding[:5]}")
        
        # 嘗試使用其他模型（如果有的話）
        if len(models) > 1 and models[0] != client.default_model:
            other_model = models[0]
            print(f"\n切換到模型 {other_model} 生成 embedding:")
            other_resp = client.create_embeddings(test_text, model=other_model)
            print(f"向量維度: {len(other_resp.data[0].embedding)}")
            print(f"前 5 個值: {other_resp.data[0].embedding[:5]}")
        
        # 測試相似度計算
        print("\n測試相似度計算（簡化版）...")
        query = "火星是紅色星球。"
        docs = ["火星表面呈現紅色。", "地球是藍色星球。"]
        
        try:
            sims = client.calculate_similarity(query, docs)
            for i, (doc, sim) in enumerate(zip(docs, sims)):
                print(f"文本 {i+1} 相似度: {sim:.4f} - {doc}")
        except Exception as e:
            print(f"相似度計算失敗: {str(e)}")
    
    simplified_test()