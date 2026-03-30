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

    if img is None:
        return {"error": "Invalid image format"}

    # --- Logic หาขอบการ์ด ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {"centering": "N/A", "error": "No card detected"}

    card_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_cnt)

    # --- วาดเส้น Guide บนรูป (Visual Feedback) ---
    output_img = img.copy()
    # วาดกรอบสี่เหลี่ยมสีเขียวรอบการ์ด
    cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 5)
    # วาดเส้นกึ่งกลางการ์ด (สีแดง)
    card_center_x = x + (w // 2)
    cv2.line(output_img, (card_center_x, y), (card_center_x, y + h), (0, 0, 255), 4)

    # --- คำนวณ Centering ---
    img_center_x = img.shape[1] // 2
    diff = card_center_x - img_center_x
    left_ratio = round(50 + (diff / w * 100))
    right_ratio = 100 - left_ratio

    # แปลงรูปกลับเป็น Base64 ส่งให้ App โชว์
    _, buffer = cv2.imencode('.jpg', output_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "centering": f"{left_ratio}/{right_ratio}",
        "visual_result": f"data:image/jpeg;base64,{img_base64}"
    }