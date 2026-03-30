import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "VaultLive Outer-Edge Tracker is Ready"}

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format"}

    # --- Preprocessing เพื่อเน้นขอบนอก ---
    # 1. แปลงเป็นขาวดำและเบลอเพื่อลบ Noise เล็กๆ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0) # เบลอเยอะหน่อยเพื่อให้ขอบหลักชัด
    
    # 2. ใช้ Threshold แบบ Otsu เพื่อแยกวัตถุออกจากพื้นหลังอัตโนมัติ
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. หา Contours ทั้งหมด
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"error": "ไม่พบวัตถุในภาพ"}

    # 4. เลือก Contour ที่ใหญ่ที่สุด (ซึ่งควรจะเป็นตัวการ์ด)
    card_cnt = max(contours, key=cv2.contourArea)
    
    # 5. สร้าง Bounding Box แบบตั้งตรง (Green Box)
    x, y, w, h = cv2.boundingRect(card_cnt)
    
    # วาดเส้นแสดงผลบนรูปต้นฉบับ
    output_img = img.copy()
    
    # วาดสี่เหลี่ยมขอบนอก (สีเขียวสะท้อนแสง)
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 8)
    
    # วาดจุดกึ่งกลาง (สีแดง) เพื่อดูว่า AI มองกลางภาพตรงไหม
    center_x, center_y = x + (w // 2), y + (h // 2)
    cv2.circle(output_img, (center_x, center_y), 15, (0, 0, 255), -1)
    
    # เพิ่มข้อความบอกขนาดพิกเซลของการ์ด
    info_text = f"Size: {w}x{h} px"
    cv2.putText(output_img, info_text, (x, y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    # แปลงกลับเป็น Base64 ส่งให้ App
    _, buffer = cv2.imencode('.jpg', output_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "Success",
        "card_width": w,
        "card_height": h,
        "visual_result": f"data:image/jpeg;base64,{img_base64}"
    }