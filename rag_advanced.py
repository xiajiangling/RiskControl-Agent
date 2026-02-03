import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- 标准导入 (基于 LangChain 0.3.x) ---
# 如果这里报错，说明 pip install langchain==0.3.7 没有成功
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# --- Community 组件导入 ---
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



class AdvancedRAG:
    def __init__(self, vector_db_path="./chroma_db", llm_client=None):
        self.llm = llm_client
        
        # 1. 初始化 Embedding
        print("⚙️ [Init] 加载 Embedding 模型 (BGE-M3)...")
        self.embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        self.vector_db = Chroma(persist_directory=vector_db_path, embedding_function=self.embedding)
        
        # 2. 初始化混合检索 (Hybrid Search)
        print("⚙️ [Init] 构建混合检索器 (BM25 + Vector)...")
        try:
            # 从向量库中提取所有文档用于构建 BM25 索引
            data = self.vector_db.get()
            all_docs = data['documents']
            all_metadatas = data['metadatas']
            
            if not all_docs:
                raise ValueError("向量库为空，请先运行 1_build_rag.py 构建知识库！")
                
            docs_obj = [Document(page_content=c, metadata=m) for c, m in zip(all_docs, all_metadatas)]
            
            # A. 关键词检索 (BM25)
            self.bm25_retriever = BM25Retriever.from_documents(docs_obj)
            self.bm25_retriever.k = 10
            
            # B. 向量检索 (Vector)
            self.vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
            
            # C. 混合检索 (Ensemble)
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.vector_retriever],
                weights=[0.4, 0.6] # 向量检索权重略高
            )
        except Exception as e:
            print(f"⚠️ BM25 初始化失败: {e}，将降级为纯向量检索。")
            self.ensemble_retriever = self.vector_db.as_retriever(search_kwargs={"k": 10})
        
        # 3. 初始化重排序 (Re-ranking)
        print("⚙️ [Init] 加载重排序模型 (BGE-Reranker)...")
        try:
            self.rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
            self.compressor = CrossEncoderReranker(model=self.rerank_model, top_n=3)
            
            # D. 最终检索器
            self.final_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=self.ensemble_retriever
            )
        except Exception as e:
            print(f"⚠️ 重排序模型加载失败: {e}，将降级为混合检索。")
            self.final_retriever = self.ensemble_retriever

    def rewrite_query(self, query: str) -> str:
        """
        Query Rewrite: 把用户口语转化为风控术语
        """
        if not self.llm:
            return query
            
        prompt = ChatPromptTemplate.from_template("""
        你是一个风控搜索专家。请将用户的口语化问题改写为更精准的术语查询语句。
        
        示例:
        输入: "我付不了款了" -> 输出: 支付拦截解除流程
        输入: "借个号测试" -> 输出: 内部测试账号加白申请
        
        用户输入: {input}
        仅输出改写后的查询语句，不要包含其他文字。
        """)
        try:
            chain = prompt | self.llm | StrOutputParser()
            rewritten = chain.invoke({"input": query})
            # 简单清洗，防止模型输出 "输出: xxx"
            rewritten = rewritten.replace("输出:", "").strip()
            print(f"🔄 [Rewrite] '{query}' -> '{rewritten}'")
            return rewritten
        except Exception:
            return query

    def search(self, query: str):
        # 1. 改写
        final_query = self.rewrite_query(query)
        # 2. 检索 (Hybrid -> Rerank)
        print(f"🔍 [Search] 正在检索: {final_query}")
        docs = self.final_retriever.invoke(final_query)
        return docs