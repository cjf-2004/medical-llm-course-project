import json
import time
import urllib.request
import os
import sys
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment
from pathlib import Path # 导入Path用于处理文件路径

# Environment variables for CUDA/NCCL
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# LangChain Imports
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. Define Your Custom LLM for LangChain ---
class CustomLLM(LLM):
    llm_url: str = 'http://127.0.0.1:9005/chatmed_generate'
    max_new_tokens: int = 1024 # 这个max_new_tokens是在RAG客户端设置的，最终会传递给LLM服务。

    @property
    def _llm_type(self) -> str:
        return "custom_local_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        header = {'Content-Type': 'application/json'}
        # Ensure the prompt format is correct for your model
        formatted_prompt = "<s>问：\n{}\n答：\n".format(prompt.strip())
        data = {
            "query": formatted_prompt,
            "max_new_tokens": self.max_new_tokens, # 使用CustomLLM实例的max_new_tokens
        }
        try:
            request = urllib.request.Request(
                url=self.llm_url,
                headers=header,
                data=json.dumps(data).encode('utf-8')
            )
            response = urllib.request.urlopen(request, timeout=120) # Added timeout
            res = response.read().decode('utf-8')
            result = json.loads(res)
            return result.get("response", "No response from LLM.")
        except urllib.error.URLError as e:
            print(f"Error connecting to LLM service at {self.llm_url}: {e.reason}")
            return f"Error: Connection error to LLM service. {e.reason}"
        except Exception as e:
            print(f"Error calling local LLM service: {e}")
            return f"Error: Could not connect to LLM service or invalid response. {e}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"llm_url": self.llm_url, "max_new_tokens": self.max_new_tokens}

# --- 2. Initialize Your Custom LLM Instance ---
# 这里的max_new_tokens会作为默认值传递给LLM服务，但服务端的max_new_tokens=400会覆盖它。
# 为了让RAG客户端的max_new_tokens生效，你需要修改服务端的max_new_tokens。
my_local_llm = CustomLLM(llm_url='http://127.0.0.1:9005/chatmed_generate', max_new_tokens=1024)

# --- 3. 加载本地数据并构建向量存储 ---
def load_and_index_local_data(data_path: str, faiss_index_path: str, force_reindex: bool = False) -> FAISS:
    """
    Loads documents from a local directory, splits them,
    and creates a FAISS vector store.
    Supports .txt, .pdf, .docx using UnstructuredFileLoader.
    Optionally loads from a saved FAISS index or forces re-indexing.
    """
    embeddings = HuggingFaceEmbeddings(model_name="./paraphrase-multilingual-MiniLM-L12-v2")

    # 检查是否已存在FAISS索引且不强制重新索引
    if Path(faiss_index_path).exists() and not force_reindex:
        print(f"Loading FAISS index from '{faiss_index_path}'...")
        try:
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Attempting to re-index.")
            # 如果加载失败，则强制重新索引
            force_reindex = True

    if force_reindex:
        print("Forcing re-indexing of documents...")
    else:
        print(f"FAISS index not found at '{faiss_index_path}'. Starting document loading and indexing...")

    docs = []
    try:
        loader = DirectoryLoader(
            data_path, 
            glob="**/*", # Load all file types
            loader_cls=UnstructuredFileLoader, # Supports PDF, DOCX etc.
            show_progress=True,
            use_multithreading=True # Attempt multithreaded loading
        )
        docs = loader.load()
    except ImportError:
        print("Warning: 'unstructured' library not found. Falling back to TextLoader for .txt files.")
        print("Install with 'pip install \"unstructured[all-docs]\"' for broader file support.")
        loader = DirectoryLoader(
            data_path, 
            glob="**/*.txt", # Only load txt files
            loader_cls=TextLoader, 
            show_progress=True
        )
        docs = loader.load()
    except Exception as e:
        print(f"An error occurred during document loading: {e}")
        print(f"Please check the path '{data_path}' and file permissions.")

    if not docs:
        print(f"No documents found in '{data_path}'. Please ensure the path is correct and contains files.")
        print("Ensure 'unstructured' is installed if you have PDF/DOCX files.")
        print("Returning an empty FAISS vector store as no documents were loaded.")
        return FAISS.from_texts([""], embeddings) # Create with a dummy empty text

    print(f"Loaded {len(docs)} documents from local directory.")

    # 改进：为文档添加文件名作为source元数据
    for doc in docs:
        if 'source' not in doc.metadata and 'file_path' in doc.metadata:
            doc.metadata['source'] = Path(doc.metadata['file_path']).name
        elif 'source' not in doc.metadata and 'filename' in doc.metadata: # unstructured有时用filename
            doc.metadata['source'] = Path(doc.metadata['filename']).name


    # 2. Text Splitting
    # 调整chunk_size以适应Llama-1的上下文窗口（2048 tokens）和max_new_tokens
    # 假设问题+prompt模板约占用200-300 tokens
    # 如果max_new_tokens是400，那么留给上下文的tokens大约是 2048 - 400 - (150-200) = ~1400 tokens
    # 如果k=3，那么每个chunk大约是 1400 / 3 = ~460 tokens
    # 考虑到中文一个字约一个token，可以设置chunk_size为400-500
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # 调整为500，适应更小的上下文和更长的输出
        chunk_overlap=100, # 适当减小重叠
        length_function=len,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(splits)} chunks.")

    # 3. Create Embeddings
    print("Creating embeddings (this may take a moment)...")
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # 之前注释掉的，这里用你指定的
    # 确保这个路径下的模型是存在的
    embeddings = HuggingFaceEmbeddings(model_name="./paraphrase-multilingual-MiniLM-L12-v2")
    
    # 4. Create FAISS Vector Store
    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("FAISS vector store created.")

    # 保存FAISS索引到本地
    print(f"Saving FAISS index to '{faiss_index_path}'...")
    vectorstore.save_local(faiss_index_path)
    print("FAISS index saved successfully.")

    return vectorstore

# --- 4. 构建 RAG Chain ---
def build_rag_chain(vectorstore: FAISS, llm: CustomLLM):
    # 调整RAG的Prompt模板，使其更明确地指导模型利用上下文并避免幻觉/拒绝
    prompt_template = """你是一个专业的医疗问答助手，擅长根据提供的资料进行信息整合。请严格依据以下“上下文”内容来回答“问题”。
如果“上下文”中没有直接或间接支持回答的信息，请明确说明“无法从提供的资料中找到相关信息”。
你的回答应客观、准确，并避免给出个人医疗建议。

上下文:
{context}

问题: {question}
回答:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 确保k值在这里生效，限制检索文档数量
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # 明确设置为3个文档

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- 5. 运行 RAG 查询 ---
def run_rag_query(qa_chain, query: str):
    print(f"\n--- Processing RAG Query: {query} ---")
    t0 = time.time()
    try:
        # 打印发送给LLM的完整prompt长度，用于调试
        # LangChain内部会构建完整的prompt，这里无法直接获取，但可以在CustomLLM._call中打印
        result = qa_chain.invoke({"query": query})
        t1 = time.time()
        time_cost = t1 - t0
        
        response_text = result.get("result", "No response generated.")
        source_docs = result.get("source_documents", [])

        print(f"RAG Chain Response: {response_text}")
        print(f"Time cost: {time_cost:.2f} seconds")
        print("\n--- Retrieved Source Documents ---")
        retrieved_sources = []
        for i, doc in enumerate(source_docs):
            source_info = doc.metadata.get('source', 'N/A')
            print(f"Doc {i+1} (Source: {source_info}):")
            print(doc.page_content[:200] + "...") # Print first 200 chars
            print("-" * 20)
            retrieved_sources.append(f"Source {i+1} ({source_info}): {doc.page_content}")
        
        return {"response": response_text, "sources": "\n\n".join(retrieved_sources), "time_cost": time_cost}
    except Exception as e:
        t1 = time.time()
        time_cost = t1 - t0
        print(f"Error running RAG chain: {e}")
        return {"response": f"Error: Could not get response from RAG chain. {e}", "sources": "", "time_cost": time_cost}

# --- 6. 运行无 RAG 查询 (直接调用 LLM) ---
def run_direct_llm_query(llm_instance: CustomLLM, query: str):
    print(f"\n--- Processing Direct LLM Query: {query} ---")
    t0 = time.time()
    try:
        response_text = llm_instance._call(query)
        t1 = time.time()
        time_cost = t1 - t0
        print(f"Direct LLM Response: {response_text}")
        print(f"Time cost: {time_cost:.2f} seconds")
        return {"response": response_text, "time_cost": time_cost}
    except Exception as e:
        t1 = time.time()
        time_cost = t1 - t0
        print(f"Error calling direct LLM: {e}")
        return {"response": f"Error: Could not get response from direct LLM. {e}", "time_cost": time_cost}

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting LangChain Alzheimer's Disease RAG/Direct LLM comparison example...")

    # --- 准备本地知识库目录和FAISS索引存储路径 ---
    alzheimer_data_dir = "./AD_data" # 替换为你在服务器上的实际路径
    faiss_index_save_path = "./faiss_index_ad" # FAISS索引将保存到这个目录

    # 阿尔茨海默病相关问题列表
    ad_questions = [
        "请详细介绍目前阿尔茨海默病（AD）有哪些最新的治疗药物或疗法，它们的作用机制分别是什么？",
        "目前，针对阿尔茨海默病的诊断，有哪些新兴的生物标志物检测方法？它们在临床应用中的优势和局限性分别是什么？",
        "阿尔茨海默病（AD）的最新研究表明，肠道菌群与疾病进展可能存在关联，请详细阐述这一假说及其潜在的治疗策略。",
        "除了药物治疗，阿尔茨海默病的非药物干预措施（如生活方式干预、认知训练）在最新临床指南中有什么新的推荐？",
        "阿尔茨海默病早期有哪些典型症状，如何进行初步判断？",
        "请问，阿尔茨海默病的风险因素有哪些？",
        "是否存在治愈阿尔茨海默病的方法？如果有，请说明，如果没有，请说明为什么。",
        "最新研究中，有无关于阿尔茨海默病预防的突破性进展？",
        "阿尔茨海默病患者家属在日常照护中应注意哪些问题？",
        "请介绍一下中国在阿尔茨海默病研究和治疗方面的最新政策或投入。",
    ]

    output_excel_path = "results_alzheimer_comparison.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "AD_RAG_vs_NoRAG"

    # Excel Header
    ws.append([
        "问题", 
        "无RAG模型回答", "无RAG耗时 (秒)",
        "有RAG模型回答", "有RAG耗时 (秒)", "RAG检索到的源文档"
    ])
    
    # 设置列宽和居中（可选，但美观）
    ws.column_dimensions['A'].width = 40 # 问题
    ws.column_dimensions['B'].width = 60 # 无RAG回答
    ws.column_dimensions['C'].width = 15 # 无RAG耗时
    ws.column_dimensions['D'].width = 80 # 有RAG回答
    ws.column_dimensions['E'].width = 15 # 有RAG耗时
    ws.column_dimensions['F'].width = 100 # RAG源文档

    # 设置表头字体加粗和居中
    header_font = Font(bold=True)
    for cell in ws["1:1"]:
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center', wrapText=True)
    
    print(f"Results will be saved to: {output_excel_path}")

    # --- Step 3: Load and Index Local AD Data (for RAG) ---
    print("\n--- Step 3: Loading and Indexing Local Alzheimer's Data for RAG ---")
    # 第一次运行会创建并保存索引，后续运行会直接加载
    # 如果AD_data目录内容有更新，可以将force_reindex设置为True来强制重新索引
    vectorstore_ad = load_and_index_local_data(data_path=alzheimer_data_dir, faiss_index_path=faiss_index_save_path, force_reindex=False)
    
    print("\n--- Step 4: Building RAG Chain ---")
    qa_chain_ad = build_rag_chain(vectorstore_ad, my_local_llm)

    print("\n--- Step 5: Running Queries ---")

    for i, query_text in enumerate(ad_questions, start=1):
        print(f"\n--- Testing Question {i}/{len(ad_questions)} ---")
        print(f"Question: {query_text}")

        # --- Test without RAG ---
        direct_llm_output = run_direct_llm_query(my_local_llm, query_text)
        no_rag_response = direct_llm_output['response']
        no_rag_time = direct_llm_output['time_cost']

        # --- Test with RAG ---
        rag_output = run_rag_query(qa_chain_ad, query_text)
        rag_response = rag_output['response']
        rag_time = rag_output['time_cost']
        rag_sources = rag_output['sources']

        # Append results to Excel
        ws.append([
            query_text,
            no_rag_response, f"{no_rag_time:.2f}",
            rag_response, f"{rag_time:.2f}", rag_sources
        ])

        # Save periodically
        if i % 2 == 0: # Save every 2 questions
            wb.save(output_excel_path)
            print(f"Intermediate results saved to {output_excel_path}")

    # Final save
    wb.save(output_excel_path)
    print(f"\n--- Testing Complete ---")
    print(f"All results saved to {output_excel_path}")