import os
import fitz  # PyMuPDF
import pyhwp
from tqdm import tqdm

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# HWP 파일에서 텍스트 추출
def extract_text_from_hwp(hwp_path):
    doc = pyhwp.HWPDocument(hwp_path)
    text = doc.body_text()
    return text

# 텍스트 데이터셋 생성
def create_text_file_from_documents(file_paths, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in tqdm(file_paths):
            if file_path.endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.endswith('.hwp'):
                text = extract_text_from_hwp(file_path)
            else:
                continue
            f.write(text + "\n")

if __name__ == "__main__":
    # 학습 데이터 파일 생성
    file_paths = ['/Users/hongseongmin/source/gpt_2_chatbot/data/create_dataset.py']
    output_file = 'training_data.txt'
    create_text_file_from_documents(file_paths, output_file)
