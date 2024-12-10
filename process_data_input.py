import streamlit as st
import PyPDF2
import pdfplumber
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from pdf2image import convert_from_bytes
from PIL import Image
import json
import openai
import base64

from detection import get_detector, get_textbox
from recognition import Predictor, get_text
from utils import group_text_box, get_image_list,printProgressBar,reformat_input,diff
# from bidi.algorithm import get_display
import numpy as np
import cv2
import torch
import os
import sys
from PIL import Image
from logging import getLogger
import time
from urllib.request import urlretrieve
from pathlib import Path
from tools.config import Cfg
import argparse
from pyvi import ViTokenizer, ViUtils
class Reader(object):

    def __init__(self, config, gpu=True, model_storage_directory=None,
                 download_enabled=True, detector=True, recognizer=True):
        self.config = config
        
        self.device = config['device']

        self.detector_path = config['detection']['model_path']
        if detector:
            self.detector = get_detector(self.detector_path, self.device)
        if recognizer:
            self.recognizer = Predictor(config)

    def detect(self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, reformat=True):

        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box = get_textbox(self.detector, img, canvas_size, mag_ratio,text_threshold, link_threshold, low_text,False, self.device)
        horizontal_list, free_list = group_text_box(text_box, slope_ths,ycenter_ths, height_ths,width_ths, add_margin)

        if min_size:
            horizontal_list = [i for i in horizontal_list if max(i[1]-i[0],i[3]-i[2]) > min_size]
            free_list = [i for i in free_list if max(diff([c[0] for c in i]), diff([c[1] for c in i]))>min_size]

        return horizontal_list, free_list

    def recognize(self, img, horizontal_list=None, free_list=None,reformat=True,imgH = 32):

        if reformat:
            img, img_cv_grey = reformat_input(img_cv_grey)

        if (horizontal_list==None) and (free_list==None):
            b,y_max, x_max = img.shape
            ratio = x_max/y_max
            max_width = int(imgH*ratio)
            crop_img = cv2.resize(img, (max_width, imgH), interpolation =  Image.ANTIALIAS)
            image_list = [([[0,0],[x_max,0],[x_max,y_max],[0,y_max]] ,crop_img)]
        else:
            image_list, max_width = get_image_list(horizontal_list, free_list, img, model_height = imgH)
        result = get_text(self.recognizer,image_list)
        return result

    def readtext(self, image,min_size = 20,\
                 text_threshold = 0.7, low_text = 0.4, link_threshold = 0.4,\
                 canvas_size = 2560, mag_ratio = 1.,\
                 slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
                 width_ths = 0.5, add_margin = 0.1):
        '''
        Parameters:
        image: file path or numpy-array or a byte stream object
        '''
        img, img_cv_grey = reformat_input(image)

        horizontal_list, free_list = self.detect(img, min_size, text_threshold,\
                                                 low_text, link_threshold,\
                                                 canvas_size, mag_ratio,\
                                                 slope_ths, ycenter_ths,\
                                                 height_ths,width_ths,\
                                                 add_margin, False)

        result = self.recognize(img, horizontal_list, free_list,False)
        return result

def correct_spelling(text):
    # Loại bỏ ký tự không mong muốn
    cleaned_text = ViUtils.remove_accents(text)  # Thêm dấu tùy ý
    return cleaned_text

# def extract_text_by_line(ocr_data, y_threshold=10):
#     from collections import defaultdict
    
#     # Dictionary để nhóm các dòng, key là y-center, value là các đoạn văn bản trên dòng đó
#     lines = defaultdict(list)

#     for box, text in ocr_data:
#         y_center = (box[0][1] + box[2][1]) // 2  # Tính trung tâm y của đoạn văn bản
#         lines[y_center].append((box, text))
    
#     # Sắp xếp các dòng theo tọa độ y
#     sorted_lines = sorted(lines.items())

#     result_lines = []
#     for y, segments in sorted_lines:
#         # Sắp xếp các đoạn trên cùng một dòng theo tọa độ x (ngang)
#         segments.sort(key=lambda seg: seg[0][0][0])
#         line_text = ' '.join([seg[1] for seg in segments])
#         result_lines.append(line_text)
    
#     # Ghép tất cả các dòng thành văn bản hoàn chỉnh
#     return '\n'.join(result_lines)
def extract_text_by_line(ocr_data, y_threshold=15):
    from collections import defaultdict
    
    lines = defaultdict(list)

    # Gom nhóm theo dòng dựa trên tọa độ y (với khoảng sai số)
    for box, text in ocr_data:
        y_center = (box[0][1] + box[2][1]) // 2
        added = False
        
        # Tìm dòng gần nhất để thêm đoạn vào
        for y in lines:
            if abs(y_center - y) <= y_threshold:
                lines[y].append((box, text))
                added = True
                break
        
        # Nếu không tìm thấy dòng gần, tạo dòng mới
        if not added:
            lines[y_center].append((box, text))
    
    # Sắp xếp các dòng theo y
    sorted_lines = sorted(lines.items())
    
    result_lines = []
    for y, segments in sorted_lines:
        # Sắp xếp các đoạn trên dòng theo x
        segments.sort(key=lambda seg: seg[0][0][0])
        line_text = ' '.join([seg[1] for seg in segments])
        result_lines.append(line_text.strip())  # Xóa khoảng trắng dư
    
    return '\n'.join(result_lines)

# Streamlit app title
st.title("Multi-PDF/Image to JSON Converter with Method Selection")

# Dropdown to select the extraction method
method = st.selectbox(
    "Select Extraction Method",
    ["PyPDF2", "pdfplumber", "OCR with pytesseract", "GPT-4o", "VietOCR"]
)

# Option to enter GPT API key if GPT-4o is selected
if method == "GPT-4o":
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if api_key:
        # openai.api_key = api_key
        client = openai.Client(api_key=api_key)
    else:
        st.warning("Please enter your OpenAI API Key to continue.")

# File uploader for PDFs or images
uploaded_files = st.file_uploader(
    "Upload PDFs or Images", 
    type=["pdf", "png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

# Process files if uploaded
if uploaded_files:
    data_list = []  # List to hold data from all files

    for uploaded_file in uploaded_files:
        # Show file name
        st.subheader(f"Preview: {uploaded_file.name}")

        if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            # Preview image files
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        
        elif uploaded_file.type == "application/pdf":
            # Preview PDF files
            st.text("Preview of first page:")
            try:
                reader = PyPDF2.PdfReader(uploaded_file)
                if reader.pages:
                    text_preview = reader.pages[0].extract_text()
                    st.text(text_preview[:1000])  # Show the first 1000 characters of text
                else:
                    st.warning("PDF file contains no text.")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
        full_text = ""
        
        if method == "PyPDF2":
            reader = PyPDF2.PdfReader(uploaded_file)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
        
        elif method == "pdfplumber":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
        
        elif method == "OCR with pytesseract":
            if uploaded_file.type == "application/pdf":
                images = convert_from_bytes(uploaded_file.read())
                custom_oem_psm_config = r'--oem 3 --psm 6'
                for image in images:
                    text = pytesseract.image_to_string(image, lang ='vie')
                    if text:
                        full_text += text + "\n"
            else:  # If it's an image file
                image = Image.open(uploaded_file)
                full_text = pytesseract.image_to_string(image, lang ='vie')
        
        elif method == "GPT-4o":
            file_content = uploaded_file.read()
            file_content_base64 = base64.b64encode(file_content).decode('utf-8')
            # OpenAI API Call using the new `openai.Completion.create` method
            response = client.chat.completions.create(
                model="gpt-4o",  # Use GPT-4o model identifier
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Trích xuất nội dung văn bản"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{file_content_base64}",
                                    "detail": "high"  # Adjust to "high" for detailed analysis
                                }
                            }
                        ]
                    }
                ],
                max_tokens=3000,
            )

            full_text = response.choices[0].message.content
        elif method == "VietOCR":
            config = Cfg.load_config_from_file('config/vgg-transformer.yml')
            start = time.time()
            reader = Reader(config)
            # print(uploaded_file)
            # result = reader.readtext(uploaded_file)
            # for rs in result:
            #     print(rs[1])
            # print(extract_text_by_line(result))
            if uploaded_file.type == "application/pdf":
                # Chuyển đổi PDF thành ảnh
                images = convert_from_bytes(uploaded_file.read())
                full_text = ""
                for img in images:
                    # Chuyển đổi ảnh thành mảng numpy
                    img_np = np.array(img)
                    result = reader.readtext(img_np)
                    # Draw bounding boxes on image
                    img_with_boxes = img_np.copy()
                    # for box in result[0]:  # Horizontal list
                    #     cv2.polylines(img_with_boxes, [np.array(box, np.int32).reshape((-1, 1, 2))],
                    #                 isClosed=True, color=(0, 255, 0), thickness=2)

                    # # Display the image with bounding boxes
                    # st.image(img_with_boxes, caption="Bounding Boxes", channels="BGR", use_container_width=True)
            else:  # Nếu là file ảnh
                img = Image.open(uploaded_file)
                img_np = np.array(img)  # Chuyển ảnh thành numpy array
                result = reader.readtext(img_np)
                # Draw bounding boxes
                # img_with_boxes = img_np.copy()
                # for box in result[0]:
                #     cv2.polylines(img_with_boxes, [np.array(box, np.int32).reshape((-1, 1, 2))],
                #                 isClosed=True, color=(0, 255, 0), thickness=2)

                # # Display the image with bounding boxes
                # st.image(img_with_boxes, caption="Bounding Boxes", channels="BGR", use_container_width=True)
            print(time.time()-start)
            full_text = extract_text_by_line(result)
            full_text = correct_spelling(full_text)

        # Extract title (first line) and remaining content
        title = full_text.split("\n")[0].strip() if full_text else "Untitled"
        content = full_text.strip()
        
        # Append extracted data to the list
        data_list.append({"title": title, "content": content})
    
    # Display and download JSON data
    st.json(data_list)
    st.download_button("Download JSON", json.dumps(data_list, ensure_ascii=False, indent=2), "output.json")


