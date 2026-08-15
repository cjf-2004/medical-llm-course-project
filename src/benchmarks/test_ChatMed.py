
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
    Sends a medical query to the ChatMed service and returns the response.

    Args:
        input_text (str): The medical question to send.
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
    # List of 20 medical questions to test
    medical_questions = [
        "请问，最近我总是感到胸闷气短，偶尔还会心悸，这是怎么回事？",
        "孩子最近发烧了，体温 38.5 摄氏度，伴有咳嗽和流鼻涕，需要去医院吗？",
        "我得了糖尿病，除了按时服药，饮食上有什么特别需要注意的地方？",
        "最近几天腹泻不止，还伴有恶心想吐，是不是吃坏东西了？该怎么办？",
        "长期失眠对身体有什么影响？有什么可以改善睡眠的方法吗？",
        "我感到关节疼痛，特别是膝盖，上下楼梯时更明显，这可能是关节炎吗？",
        "孕妇在怀孕期间应该如何补充叶酸？什么食物富含叶酸？",
        "高血压患者可以运动吗？有什么适合的运动方式？",
        "我的皮肤上出现了一些红色斑点，有点痒，这是过敏吗？需要用什么药？",
        "慢性胃炎患者日常生活中应该如何保养胃部？",
        "体检报告显示我血脂偏高，我该如何调整生活习惯来降低血脂？",
        "长期对着电脑工作，眼睛感到干涩和疲劳，有什么缓解方法吗？",
        "我有颈椎病，除了休息，还有什么康复训练可以做？",
        "孩子疫苗接种后出现了低烧和局部红肿，这是正常反应吗？",
        "更年期女性可能出现哪些症状？有什么方法可以缓解？",
        "如何预防感冒和流感？每年都需要接种流感疫苗吗？",
        "我被狗咬伤了，伤口不大但破皮了，需要打狂犬疫苗吗？",
        "甲状腺功能减退患者在饮食上有什么禁忌？",
        "怎样才能有效地戒烟？有什么药物或方法可以帮助？",
        "我在户外被蚊虫叮咬后，皮肤红肿痒痛，可以用什么药膏止痒消肿？"
    ]

    # Create a new Excel workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "LLM Medical Test Results" # Set sheet title
    
    # Add header row
    # Note: Since you're using a generated list, there are no 'standard answers' from a file.
    # The '标准答案' column will be empty unless you manually fill it in later.
    ws.append(["问题", "标准答案 (人工评测)", "LLM 答案", "耗时 (秒)"])

    output_excel_path = "src/web_services/test_examples/results_medical_questions.xlsx"
    
    print(f"Starting test with {len(medical_questions)} questions...")
    print(f"Results will be saved to: {output_excel_path}")

    for i, query_text in enumerate(medical_questions, start=1):
        print(f"\n--- Testing Question {i}/{len(medical_questions)} ---")
        print(f"Question: {query_text}")

        t0 = time.time()
        result = test_service(query_text)
        t1 = time.time()
        time_cost = t1 - t0
        print(f"Time cost: {time_cost:.2f} seconds")

        llm_response = result.get("response", "Error: No response from LLM.")
        
        # Append data to the Excel sheet
        # 'Standard Answer' column is left empty here as these are newly generated questions.
        ws.append([query_text, "", llm_response, f"{time_cost:.2f}"])

        # Save periodically in case of crashes, or only at the end
        if i % 5 == 0: # Save every 5 questions
            wb.save(output_excel_path)
            print(f"Intermediate results saved to {output_excel_path}")

    # Final save after all questions are processed
    wb.save(output_excel_path)
    print(f"\n--- Testing Complete ---")
    print(f"All results saved to {output_excel_path}")