# CUDA_VISIBLE_DEVICES=3
# coding=utf-8
# Created by Michael Zhu
# DataSelect AI, 2023

import json
import time
import os
import urllib.request
import sys
import pandas as pd # You'll need pandas to read CSV files
from openpyxl import Workbook

# Ensure CUDA_VISIBLE_DEVICES is set if this script were to
# directly interact with CUDA. For a client script calling a web service,
# it primarily affects where the client itself might run if it had GPU components,
# but the service determines its own device.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Add parent directory to sys.path if needed for module imports.
# Assuming this script runs from the project root or similar.
# Adjust if your actual structure is different.
sys.path.append("./") # Adjust this if your module path differs


def test_service(input_text, url='http://127.0.0.1:9005/chatmed_generate'):
    """
    Sends a medical query to the ChatMed service and returns the response.

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
    if not (input_text.startswith("<s>问：") and input_text.endswith("答：\n")):
        prompt = "<s>问：\n{}\n答：\n".format(input_text)
    else:
        prompt = input_text # Use as-is if it's already in the correct format

    data = {
        "query": prompt,
        "max_new_tokens": 1024,
        # You might want to experiment with `temperature` for TruthfulQA,
        # e.g., a lower temperature for more deterministic/factual answers.
        # "temperature": 0.1, 
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
    # --- Configuration ---
    TRUTHFULQA_DATA_PATH = "./TruthfulQA.csv"
    OUTPUT_EXCEL_PATH = "./results_truthfulqa.xlsx"
    NUM_QUESTIONS_TO_TEST = 100 # Adjust this to test more or fewer questions
    # --- End Configuration ---

    # Ensure pandas is installed
    try:
        import pandas as pd
    except ImportError:
        print("Pandas is not installed. Please install it using: pip install pandas openpyxl")
        sys.exit(1)

    print(f"Loading TruthfulQA dataset from: {TRUTHFULQA_DATA_PATH}")
    try:
        # Read the CSV file
        df = pd.read_csv(TRUTHFULQA_DATA_PATH)
        # Select relevant columns, assuming 'Question', 'Best Answer', 'Incorrect Answers' exist
        # TruthfulQA.csv has columns like 'Question', 'Best Answer', 'Correct Answers', 'Incorrect Answers'
        df = df[['Question', 'Best Answer', 'Correct Answers', 'Incorrect Answers']]
    except FileNotFoundError:
        print(f"Error: TruthfulQA.csv not found at {TRUTHFULQA_DATA_PATH}.")
        print("Please download it from https://github.com/sylinrl/TruthfulQA/blob/main/TruthfulQA.csv and place it there.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing expected column in TruthfulQA.csv: {e}.")
        print(f"Ensure the CSV has 'Question', 'Best Answer', 'Correct Answers', 'Incorrect Answers' columns. Available columns: {df.columns.tolist()}")
        sys.exit(1)
    
    # Limit the number of questions if desired
    questions_to_test = df.head(NUM_QUESTIONS_TO_TEST)

    wb = Workbook()
    ws = wb.active
    ws.title = "TruthfulQA Test Results" # Set sheet title
    
    # Add header row for the Excel file
    ws.append([
        "TruthfulQA问题", 
        "TruthfulQA标准最佳答案", 
        "TruthfulQA所有正确答案", 
        "TruthfulQA所有错误答案", 
        "LLM生成答案", 
        "耗时 (秒)"
    ])

    print(f"Starting TruthfulQA test with {len(questions_to_test)} questions...")
    print(f"Results will be saved to: {OUTPUT_EXCEL_PATH}")

    for i, row in questions_to_test.iterrows():
        question = row['Question']
        best_answer = row['Best Answer']
        correct_answers = row['Correct Answers'] # Can be multiple, separated by ';'
        incorrect_answers = row['Incorrect Answers'] # Can be multiple, separated by ';'

        print(f"\n--- Testing Question {i+1}/{len(questions_to_test)} ---")
        print(f"Question: {question}")
        print(f"Best Answer: {best_answer}")
        # print(f"Correct Answers: {correct_answers}") # Optional to print all
        # print(f"Incorrect Answers: {incorrect_answers}") # Optional to print all

        t0 = time.time()
        result = test_service(question)
        t1 = time.time()
        time_cost = t1 - t0
        print(f"Time cost: {time_cost:.2f} seconds")

        llm_response = result.get("response", "Error: No response from LLM.")
        print(f"LLM Response: {llm_response}")
        
        # Append data to the Excel sheet
        ws.append([
            question, 
            best_answer, 
            correct_answers, 
            incorrect_answers, 
            llm_response, 
            f"{time_cost:.2f}"
        ])

        # Save periodically in case of crashes
        if (i + 1) % 10 == 0: # Save every 10 questions
            wb.save(OUTPUT_EXCEL_PATH)
            print(f"Intermediate results saved to {OUTPUT_EXCEL_PATH}")

    # Final save after all questions are processed
    wb.save(OUTPUT_EXCEL_PATH)
    print(f"\n--- TruthfulQA Testing Complete ---")
    print(f"All results saved to {OUTPUT_EXCEL_PATH}")