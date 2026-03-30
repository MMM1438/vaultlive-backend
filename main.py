import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # อนุญาตให้ทุกที่เข้าถึงได้ (รวมถึงแอปเรา)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "VaultLive Backend is Running"}

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 1. เตรียมรูป (Preprocessing)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. หาขอบนอกสุดของการ์ด (Outer Edge)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"error": "No card detected"}
    
    card_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_cnt)
    
    # 3. ตัดเฉพาะตัวการ์ดออกมา (Crop) เพื่อหาขอบใน (Inner Artwork)
    card_img = img[y:y+h, x:x+w]
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Threshold เพื่อแยก "ขอบขาว" ออกจาก "รูปภาพ"
    _, thresh = cv2.threshold(card_gray, 200, 255, cv2.THRESH_BINARY) 
    
    # หาขอบของ Artwork ด้านใน
    inner_contours, _ = cv2.findContours(cv2.bitwise_not(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if inner_contours:
        # เลือก Contour ที่ใหญ่ที่สุดในตัวการ์ด (ซึ่งมักจะเป็น Artwork)
        inner_cnt = max(inner_contours, key=cv2.contourArea)
        ix, iy, iw, ih = cv2.boundingRect(inner_cnt)
        
        # 4. คำนวณความหนาของขอบ (Border Width)
        left_border = ix
        right_border = w - (ix + iw)
        top_border = iy
        bottom_border = h - (iy + ih)
        
        # 5. คำนวณ Ratio (เช่น 50/50 คือเป๊ะมาก)
        lr_ratio_val = round((left_border / (left_border + right_border)) * 100)
        tb_ratio_val = round((top_border / (top_border + bottom_border)) * 100)
        
        # วาดเส้นไกด์แสดงขอบใน (สีเหลือง VaultLive)
        cv2.rectangle(card_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 3)
        
        centering_text = f"L/R: {lr_ratio_val}/{100-lr_ratio_val} T/B: {tb_ratio_val}/{100-tb_ratio_val}"
    else:
        centering_text = "Analysis Failed"

    # แปลงกลับเป็น Base64
    _, buffer = cv2.imencode('.jpg', card_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{img_base64}"
    }