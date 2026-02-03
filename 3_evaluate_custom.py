
import os
# --- 配置 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 引入你的高级 RAG 引擎
from rag_advanced import AdvancedRAG



# 1. 初始化本地 LLM
eval_llm = ChatOpenAI(
    model="Qwen2.5-7B",
    openai_api_base="http://localhost:8002/v1",
    openai_api_key="EMPTY",
    temperature=0.0 # 评测时温度设为0，保持稳定
)

# 2. 初始化被测对象 (RAG 引擎)
# 注意：这里我们只测 RAG 检索生成能力，暂不测 Router
rag_engine = AdvancedRAG(llm_client=eval_llm)

# 3. 定义评测 Prompt (LLM-as-a-Judge)
EVAL_PROMPT = """
你是一个公正的阅卷老师。请根据参考答案（Ground Truth），对考生的回答（Candidate Answer）进行打分。

【评分标准】
1. 忠实度 (Faithfulness): 考生回答是否与参考资料一致？没有编造信息？(0-1分)
2. 准确度 (Accuracy): 考生回答是否覆盖了参考答案的核心要点？(0-1分)

【输入数据】
问题: {question}
参考答案: {ground_truth}
考生回答: {answer}
检索到的资料: {context}

【输出格式】
请输出 JSON 格式，包含 faithfulness_score, accuracy_score, reason 三个字段。
例如: {{"faithfulness_score": 0.9, "accuracy_score": 0.8, "reason": "回答准确，但缺少了关于审批时效的说明。"}}
"""

def evaluate_one_case(case):
    question = case['question']
    gt = case['ground_truth']
    
    print(f"\n📝 正在评测: {question}")
    
    # 1. 让 Agent 生成回答
    print("   -> 检索中...")
    docs = rag_engine.search(question)
    context_text = "\n".join([d.page_content for d in docs])
    
    print("   -> 生成回答中...")
    # 简单模拟生成过程
    gen_prompt = f"基于以下资料回答用户问题：\n{context_text}\n\n问题：{question}"
    candidate_answer = eval_llm.invoke(gen_prompt).content
    
    # 2. 让 LLM 打分
    print("   -> 打分中...")
    eval_chain = ChatPromptTemplate.from_template(EVAL_PROMPT) | eval_llm | StrOutputParser()
    result_str = eval_chain.invoke({
        "question": question,
        "ground_truth": gt,
        "answer": candidate_answer,
        "context": context_text
    })
    
    # 3. 解析分数 (简单正则提取，防止 JSON 格式错误)
    try:
        # 尝试提取 JSON 部分
        match = re.search(r"\{.*\}", result_str, re.DOTALL)
        if match:
            score_dict = json.loads(match.group())
            return {
                "question": question,
                "answer": candidate_answer,
                "scores": score_dict
            }
    except:
        print(f"❌ 解析分数失败: {result_str}")
        return None

if __name__ == "__main__":
    # 加载测试集
    with open("3_evaluate_data.json", "r") as f:
        test_data = json.load(f)
        
    report = []
    for case in test_data:
        res = evaluate_one_case(case)
        if res:
            report.append(res)
            print(f"   🏆 得分: {res['scores']}")
            
    # 保存报告
    with open("evaluation_report_custom.json", "w", encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n✅ 评测完成！报告已保存至 evaluation_report_custom.json")