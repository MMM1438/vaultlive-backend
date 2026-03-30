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
    return {"status": "VaultLive AI Precision Edge Engine is Running"}

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image format"}

    # --- ⚙️ ขั้นตอนที่ 1: หาและตัดตัวการ์ด (Warp Perspective & Crop) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 1.1 ใช้ Auto Canny เพื่อหาขอบหลักของการ์ด
    v = np.median(gray)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    
    # 1.2 หา Contours และเลือกสี่เหลี่ยมที่ใหญ่ที่สุด (Card Outer Boundary)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"error": "ไม่พบตัวการ์ดในภาพ กรุณาถ่ายใหม่บนพื้นหลังที่ตัดกัน"}

    card_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_cnt)

    # 1.3 ตัดรูป (Crop) เฉพาะตัวการ์ดออกมาเพื่อลด Noise
    card_img = img[y:y+h, x:x+w].copy()
    
    # --- ⚙️ ขั้นตอนที่ 2: หาขอบใน (Inner Border) และคำนวณ Centering ---
    # 2.1 เตรียมรูปการ์ดที่ Crop แล้ว (Warped/Cropped Preprocessing)
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    
    # 2.2 ใช้ Adaptive Threshold เพื่อสู้แสงและหาขอบด้านใน (Artwork Frame)
    thresh = cv2.adaptiveThreshold(card_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # 2.3 Clean Noise ออกเล็กน้อย
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 2.4 หา Inner Contours ทั้งหมด (RETR_TREE เพื่อดูโครงสร้างด้านใน)
    inner_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2.5 กรองเฉพาะ Contour ที่เป็น "สี่เหลี่ยม" ขนาดพอเหมาะของการ์ด
    possible_artworks = []
    card_area = w * h
    for cnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        area = iw * ih
        # เงื่อนไข: ต้องมีพื้นที่ระหว่าง 20% ถึง 85% ของการ์ด (ขนาดมาตรฐาน Artwork)
        if 0.20 * card_area < area < 0.85 * card_area:
            possible_artworks.append((ix, iy, iw, ih))

    # 2.6 เลือกสี่เหลี่ยมที่ "อยู่กึ่งกลาง" ของการ์ดที่สุด (Inner Border สุดนิ่ง)
    center_x, center_y = w / 2, h / 2
    
    if possible_artworks:
        final_inner_rect = min(possible_artworks, 
                               key=lambda r: (r[0] + r[2]/2 - center_x)**2 + (r[1] + r[3]/2 - center_y)**2)
        ix, iy, iw, ih = final_inner_rect

        # --- ⚙️ ขั้นตอนที่ 3: คำนวณความหนาของขอบ (Border Thickness) และ Ratio ---
        # วัดระยะห่างจากขอบการ์ดนอก (0,0) ถึงขอบ Artwork ใน (ix,iy)
        left_dist = ix
        right_dist = w - (ix + iw)
        top_dist = iy
        bottom_dist = h - (iy + ih)

        # ป้องกันการหารด้วยศูนย์ (ถ้า ix หรือ w-iw+ix เป็น 0)
        lr_total = left_dist + right_dist if (left_dist + right_dist) > 0 else 1
        tb_total = top_dist + bottom_dist if (top_dist + bottom_dist) > 0 else 1

        lr_ratio = round((left_dist / lr_total) * 100)
        tb_ratio = round((top_dist / tb_total) * 100)

        # --- ⚙️ ขั้นตอนที่ 4: วาดเส้น Guide บนรูปเพื่อแสดงผล (VaultLive Pro Look) ---
        output_img = card_img.copy()
        
        # วาดเส้นขอบใน (สีทอง VaultLive - #FFD700)
        cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 4)
        
        # วาดสี่เหลี่ยมรอบนอกของการ์ด (สีเขียวสะท้อนแสง)
        cv2.rectangle(output_img, (0, 0), (w, h), (0, 255, 0), 6)
        
        # วาดจุดกึ่งกลาง (สีแดง) เพื่อดูความเบี่ยงเบน
        cv2.circle(output_img, (int(w/2), int(h/2)), 10, (0, 0, 255), -1)

        centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} | T/B: {tb_ratio}/{100-tb_ratio}"
    else:
        output_img = card_img
        centering_text = "Analysis Failed: Inner Frame not found"

    # แปลงกลับเป็น Base64 ส่งให้ App
    _, buffer = cv2.imencode('.jpg', output_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "status": "Success" if possible_artworks else "Failed",
        "centering": centering_text,
        "visual_result": f"data:image/jpeg;base64,{img_base64}",
        "raw_borders": {
            "left": left_dist if possible_artworks else 0,
            "right": right_dist if possible_artworks else 0,
            "top": top_dist if possible_artworks else 0,
            "bottom": bottom_dist if possible_artworks else 0
        }
    }