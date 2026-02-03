import os
import glob

# --- 【新增】设置 Hugging Face 镜像地址，解决连接超时问题 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- 配置路径 ---
SOP_DIR = "./data/sops"
PERSIST_DIRECTORY = "./chroma_db"

def load_and_split_sops():
    # ... (这部分代码不用变) ...
    print(f"📂 开始扫描目录: {SOP_DIR}")
    files = glob.glob(os.path.join(SOP_DIR, "*.md"))
    
    if not files:
        print("❌ 错误: data/sops/ 目录下没有找到 .md 文件！")
        return []

    all_splits = []
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for file_path in files:
        print(f"   - 处理文档: {os.path.basename(file_path)}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                splits = markdown_splitter.split_text(text)
                for split in splits:
                    split.metadata["source"] = os.path.basename(file_path)
                all_splits.extend(splits)
                print(f"     -> 切分为 {len(splits)} 个语义块")
        except Exception as e:
            print(f"❌ 读取文件失败 {file_path}: {e}")

    return all_splits

def build_vector_store(splits):
    print("\n🧠 正在加载 Embedding 模型 (BAAI/bge-m3)... (第一次运行会下载模型，约需几分钟)")
    
    # 【注意】这里会自动使用上面设置的 hf-mirror.com 镜像
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            # 如果显存不够，可以取消下面这行的注释，强制用CPU跑embedding
            # model_kwargs={'device': 'cpu'} 
        )
    except Exception as e:
        print(f"❌ 模型下载失败，请检查网络设置: {e}")
        return None

    print(f"💾 正在构建向量索引，共 {len(splits)} 条数据...")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"✅ 向量库已保存至: {PERSIST_DIRECTORY}")
    return vectorstore

def test_query(vectorstore):
    if not vectorstore:
        return
    print("\n🔎 --- 开始检索测试 ---")
    
    # 测试案例 1: 内部测试加白
    query1 = "我是测试，想申请个加白账号跑流程"
    print(f"\n❓ 问题: {query1}")
    results = vectorstore.similarity_search(query1, k=2)
    for i, res in enumerate(results):
        print(f"   [结果{i+1}] (来源: {res.metadata['source']})\n   {res.page_content[:100].replace(chr(10), ' ')}...") # 把换行符替换为空格显示

    # 测试案例 2: 用户被拦截
    query2 = "支付提示风险拦截怎么办？"
    print(f"\n❓ 问题: {query2}")
    results = vectorstore.similarity_search(query2, k=2)
    for i, res in enumerate(results):
        print(f"   [结果{i+1}] (来源: {res.metadata['source']})\n   {res.page_content[:100].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    splits = load_and_split_sops()
    if splits:
        vector_db = build_vector_store(splits)
        test_query(vector_db)