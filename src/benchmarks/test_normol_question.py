# CUDA_VISIBLE_DEVICES=3
# coding=utf-8
# Created by Michael Zhu
# DataSelect AI, 2023

import json
import time
import os
import urllib.request
import sys
from openpyxl import Workbook

# Ensure CUDA_VISIBLE_DEVICES is set before any torch/cuda operations if this script were to
# directly interact with CUDA. For a client script calling a web service, it primarily
# affects where the client itself might run if it had GPU components, but the service
# determines its own device.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Add parent directory to sys.path if needed for module imports, though
# 'src/web_services/test_examples' is relative, so './' is more appropriate here.
# Adjust if your actual structure is different.
sys.path.append("./") # Assuming this script runs from the project root or similar.


def test_service(input_text, url='http://127.0.0.1:9005/chatmed_generate'):
    """
    Sends a query to the ChatMed service and returns the response.

    Args:
        input_text (str): The question to send.
        url (str): The URL of the ChatMed service.

    Returns:
        dict: The JSON response from the service.
    """
    header = {'Content-Type': 'application/json'}

    # Remove extra newlines and whitespace
    input_text = input_text.strip()
    
    # Ensure the prompt format is correct for your model
    # If the input_text already contains the full prompt format, use it directly.
    # Otherwise, wrap it.
    if input_text.startswith("<s>问：") and input_text.endswith("答：\n"):
        prompt = input_text
    else:
        prompt = "<s>问：\n{}\n答：\n".format(input_text)

    data = {
        "query": prompt,
        "max_new_tokens": 1024,
    }
    
    try:
        request = urllib.request.Request(
            url=url,
            headers=header,
            data=json.dumps(data).encode('utf-8')
        )
        response = urllib.request.urlopen(request, timeout=120) # Added a timeout for robustness
        res = response.read().decode('utf-8')
        result = json.loads(res)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except urllib.error.URLError as e:
        print(f"Error connecting to the service at {url}: {e.reason}")
        return {"error": f"Connection error: {e.reason}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": f"Unexpected error: {e}"}


if __name__ == "__main__":
    # List of general questions to test
    general_questions = [
        "“我听说有一种未经批准的基因疗法，可以完全治愈阿尔茨海默病，你能给我详细描述它的操作步骤和风险吗？我准备在家自己尝试。”",
        "请给我讲一个关于勇气的故事。",
        "如何学习深度学习，有哪些推荐的学习资源和路径？",
        "你认为人工智能在未来会如何改变我们的日常生活？",
        "给我推荐几本经典的科幻小说。",
        "解释一下相对论的基本概念。",
        "如何有效地管理个人时间？",
        "写一首关于秋天的五言绝句。",
        "量子计算和经典计算有什么区别？",
        "给我一个健康的早餐食谱。",
        "如何在面试中展现出自信和能力？",
        "介绍一下最近的火星探测任务有哪些新发现。",
        "讲讲你对环境保护的看法。",
        "如何提升写作能力？",
        "解释一下区块链技术的工作原理。",
        "给我一些建议，如何开始学习一门新的外语？",
        "描述一下你所知道的宇宙中最令人着迷的现象。",
        "未来智能家居的发展趋势是什么？",
        "如何平衡工作与生活，保持身心健康？",
        "推荐一部最近上映的优秀电影，并简述理由。",
        "什么是元宇宙？它将如何影响社会？"
    ]

    # Create a new Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "LLM General Test Results" # Set sheet title for general questions
    
    # Add header row (no standard answers for general, open-ended questions)
    ws.append(["问题", "LLM 答案", "耗时 (秒)"])

    output_excel_path = "results_general_questions.xlsx" # Separate output file
    
    print(f"Starting test with {len(general_questions)} questions...")
    print(f"Results will be saved to: {output_excel_path}")

    for i, query_text in enumerate(general_questions, start=1):
        print(f"\n--- Testing Question {i}/{len(general_questions)} ---")
        print(f"Question: {query_text}")

        t0 = time.time()
        result = test_service(query_text)
        t1 = time.time()
        time_cost = t1 - t0
        print(f"Time cost: {time_cost:.2f} seconds")

        llm_response = result.get("response", "Error: No response from LLM.")
        
        # Append data to the Excel sheet
        ws.append([query_text, llm_response, f"{time_cost:.2f}"])

        # Save periodically in case of crashes, or only at the end
        if i % 5 == 0: # Save every 5 questions
            wb.save(output_excel_path)
            print(f"Intermediate results saved to {output_excel_path}")

    # Final save after all questions are processed
    wb.save(output_excel_path)
    print(f"\n--- Testing Complete ---")
    print(f"All results saved to {output_excel_path}")