import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from typing import TypedDict, Literal, List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from rag_advanced import AdvancedRAG 

# --- 1. 配置与初始化 ---

# 设置 Embedding 模型 (需与构建知识库时一致)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# # 加载向量数据库
# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ✅ 替换为: 初始化高级 RAG 引擎
rag_engine = AdvancedRAG(llm_client=llm)

# 加载本地大模型 (连接 vLLM)
# 【注意】如果没有本地模型，可以将 base_url 改为在线 API 地址
llm = ChatOpenAI(
    model="Qwen2.5-7B", 
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8002/v1", 
    temperature=0.1 # 风控场景需要低随机性
)

# 加载模拟数据库
with open("./data/db/mock_database.json", "r") as f:
    MOCK_DB = json.load(f)

# 加载 FAQ
with open("./data/faqs/risk_faq.json", "r") as f:
    FAQ_DATA = json.load(f)

# --- 2. 定义状态 (State) ---
class AgentState(TypedDict):
    """
    定义 Agent 在运行过程中的共享状态
    """
    user_id: str
    messages: List[str]      # 聊天记录
    intent: str              # 意图分类结果: 'customer_service' | 'internal_test' | 'handoff'
    is_internal: bool        # 是否内部人员
    final_response: str      # 最终回复给用户的内容

# --- 3. 定义节点逻辑 (Nodes) ---

def check_user_identity(state: AgentState):
    """
    【前置节点】查询 Mock DB，判断用户身份
    """
    uid = state['user_id']
    user_info = MOCK_DB['users_table'].get(uid)
    
    is_internal = False
    if user_info and user_info.get('role') == 'internal_qa':
        is_internal = True
        
    print(f"\n[System] 用户身份校验: UID={uid}, 内部人员={is_internal}")
    return {"is_internal": is_internal}

def intent_router(state: AgentState):
    """
    【路由节点】分析用户意图
    """
    last_message = state['messages'][-1]
    
    # 构造 Prompt
    router_prompt = ChatPromptTemplate.from_template("""
    你是一个京东风控系统的路由助手。请分析用户的输入，将其归类为以下三种意图之一：
    
    1. "internal_test": 用户暗示是内部测试人员，想申请加白、跑流程、借号测试、环境联调等。关键词：测试、加白、跑流程、环境、联调、借号。
    2. "handoff": 涉及转人工、投诉、极其紧急的个案、或用户明确要求转人工。
    3. "customer_service": 普通客诉问题，如支付拦截、账号被封、解封咨询、名词解释。
    
    用户输入: {input}
    
    请仅输出分类结果（不要输出其他文字）：internal_test 或 handoff 或 customer_service
    """)
    
    chain = router_prompt | llm
    response = chain.invoke({"input": last_message})
    intent = response.content.strip()
    
    print(f"[Router] 意图识别结果: {intent}")
    return {"intent": intent}

def rag_node(state: AgentState):
    """
    【客诉节点】RAG 检索回答
    """
    query = state['messages'][-1]
    
    # 1. 先查 FAQ (精确匹配)
    for faq in FAQ_DATA:
        if faq['question'] in query: # 简单匹配，实际可用向量匹配
            return {"final_response": f"【FAQ匹配】{faq['answer']}"}
    
    # 2. 查向量库
    # docs = retriever.invoke(query)
    # context = "\n\n".join([d.page_content for d in docs])

    print("🔍 执行 Advanced RAG 检索...")
    docs = rag_engine.search(query) # 这里会自动触发 Rewrite -> Hybrid -> Rerank
    context = "\n\n".join([d.page_content for d in docs])
    
    # 3. 生成回答
    rag_prompt = ChatPromptTemplate.from_template("""
    基于以下参考资料回答用户问题。
    约束：
    1. 严格按照【结论 -> 步骤 -> 注意事项】的格式输出。
    2. 不要编造资料中没有的信息。
    3. 如果资料不足以回答，请建议转人工。

    参考资料：
    {context}

    用户问题：{input}
    """)
    
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "input": query})
    return {"final_response": response.content}

def test_flow_node(state: AgentState):
    """
    【测试流程节点】处理加白逻辑
    """
    # 1. 资格校验
    if not state['is_internal']:
        return {"final_response": "⚠️ 权限拒绝：检测到您不是内部测试人员 (internal_qa)，无权申请加白。请按正常客诉流程申诉。"}
    
    # 2. 检索加白 SOP
    docs = retriever.invoke("内部测试账号加白流程")
    context = docs[0].page_content if docs else "未找到SOP"
    
    # 3. 模拟工具调用结果
    return {"final_response": f"""
✅ **资格校验通过**
检测到您的身份为：京东内部测试工程师 (internal_qa)。

为您检索到《内部测试账号加白SOP》核心流程：
{context[:200]}...

🚀 **已为您自动发起申请**
- 申请UID: {state['user_id']}
- 策略范围: 防刷单拦截
- 预计生效时间: 5分钟后

请在测试完成后及时通知我移除白名单。
"""}

def handoff_node(state: AgentState):
    """
    【转人工节点】强制结构化输出
    """
    # 这是一个硬约束的例子，强制模型输出 JSON
    return {"final_response": f"""
正在为您转接人工客服...
请提供以下信息以便我们快速处理：
--------------------------------
【转人工工单预填】
UID: {state['user_id']}
时间: 2024-05-20
问题描述: {state['messages'][-1]}
--------------------------------
"""}

# --- 4. 构建图 (Graph) ---

workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("check_identity", check_user_identity)
workflow.add_node("router", intent_router)
workflow.add_node("rag_service", rag_node)
workflow.add_node("test_process", test_flow_node)
workflow.add_node("handoff_service", handoff_node)

# 设置入口
workflow.set_entry_point("check_identity")

# 添加边 (Edges)
workflow.add_edge("check_identity", "router")

# 条件边：根据 intent 跳转不同节点
def route_decision(state):
    intent = state['intent']
    if intent == "internal_test":
        return "test_process"
    elif intent == "handoff":
        return "handoff_service"
    else:
        return "rag_service"

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "test_process": "test_process",
        "handoff_service": "handoff_service",
        "rag_service": "rag_service"
    }
)

# 结束边
workflow.add_edge("rag_service", END)
workflow.add_edge("test_process", END)
workflow.add_edge("handoff_service", END)

# 编译图
app = workflow.compile()

# --- 5. 运行测试 (CLI) ---

if __name__ == "__main__":
    print("🤖 京东风控智能体已启动... (输入 'q' 退出)")
    
    # 模拟登录用户 (可以在这里修改 UID 来测试不同身份)
    # user_001 = 内部测试员
    # user_002 = 被封号用户
    CURRENT_USER = "user_003" 
    
    while True:
        user_input = input(f"\nUser ({CURRENT_USER}): ")
        if user_input.lower() == 'q':
            break
            
        # 构造初始状态
        initial_state = {
            "user_id": CURRENT_USER,
            "messages": [user_input],
            "intent": "",
            "is_internal": False,
            "final_response": ""
        }
        
        # 运行图
        try:
            result = app.invoke(initial_state)
            print(f"\nAgent: {result['final_response']}")
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            print("提示：请确保 vLLM 服务已在 localhost:8002 启动")