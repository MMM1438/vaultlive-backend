import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def order_points(pts):
    """เรียงพิกัด 4 มุม: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย เพื่อเตรียมดัดภาพ"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Invalid image"}

    # --- 1. Pre-processing ขั้นสูง (สู้แสงสะท้อนบนซอง/สลีฟ) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ใช้ CLAHE ดึง Contrast ให้ขอบการ์ดเด้งออกมาจากพื้นหลัง
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # เบลอภาพนิดหน่อยแต่รักษาขอบไว้
    blurred = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)

    # --- 2. หาขอบนอกการ์ด (Outer Edge) ---
    v = np.median(blurred)
    edged = cv2.Canny(blurred, int(max(0, (1.0 - 0.33) * v)), int(min(255, (1.0 + 0.33) * v)))
    edged = cv2.dilate(edged, np.ones((5,5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: 
        _, buf = cv2.imencode('.jpg', img)
        return {"centering": "ไม่พบการ์ด", "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"}

    # เลือกสี่เหลี่ยมที่ใหญ่ที่สุด
    card_cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(card_cnt, True)
    approx = cv2.approxPolyDP(card_cnt, 0.02 * peri, True)

    # --- 3. ดัดภาพให้ตรงเป๊ะ (Perspective Warp) ---
    if len(approx) == 4 and cv2.contourArea(card_cnt) > 50000:
        rect = order_points(approx.reshape(4, 2))
        
        # ล็อคสัดส่วนการ์ดมาตรฐาน 2.5 x 3.5 นิ้ว (กว้าง 500px, สูง 700px)
        maxWidth, maxHeight = 500, 700
        dst = np.array([
            [0, 0], [maxWidth - 1, 0], 
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
        ], dtype="float32")
        
        # รีดภาพให้แบน
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_color = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        warped_gray = cv2.warpPerspective(enhanced_gray, M, (maxWidth, maxHeight))

        # --- 4. หาขอบใน (Artwork) บนรูปที่แบนราบแล้ว ---
        w_thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        inner_cnts, _ = cv2.findContours(w_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_inner = None
        min_dist = float('inf')
        
        for cnt in inner_cnts:
            ix, iy, iw, ih = cv2.boundingRect(cnt)
            area = iw * ih
            # Artwork ปกติจะกินพื้นที่ 40% - 85% ของการ์ด
            if (maxWidth * maxHeight * 0.4) < area < (maxWidth * maxHeight * 0.85):
                # ต้องอยู่ใกล้จุดกึ่งกลาง (กันไปจับโดนตัวหนังสือขอบๆ)
                dist = abs((ix + iw/2) - maxWidth/2) + abs((iy + ih/2) - maxHeight/2)
                if dist < min_dist:
                    min_dist = dist
                    best_inner = (ix, iy, iw, ih)

        # --- 5. คำนวณความแม่นยำระดับพิกเซล ---
        output_img = warped_color.copy()
        
        if best_inner:
            ix, iy, iw, ih = best_inner
            
            # การวัดแบบนี้จะแม่นยำ 100% เพราะรูปถูกดัดตรงแล้ว
            left, right = ix, maxWidth - (ix + iw)
            top, bottom = iy, maxHeight - (iy + ih)
            
            lr_ratio = round((left / (left + right)) * 100) if (left + right) > 0 else 50
            tb_ratio = round((top / (top + bottom)) * 100) if (top + bottom) > 0 else 50
            centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} T/B: {tb_ratio}/{100-tb_ratio}"

            # วาดเส้นไกด์ไลน์ (VaultLive Gold)
            cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 4)
            # ขีดเส้นกางเขนบางๆ กลางการ์ดให้ดู Pro
            cv2.line(output_img, (int(maxWidth/2), 0), (int(maxWidth/2), maxHeight), (0, 255, 0), 1)
            cv2.line(output_img, (0, int(maxHeight/2)), (maxWidth, int(maxHeight/2)), (0, 255, 0), 1)
        else:
            centering_text = "ไม่พบขอบ Artwork ด้านใน"

        # ส่งรูปการ์ดที่ "ตัดและดัดตรง" กลับไป
        _, buffer = cv2.imencode('.jpg', output_img)
        return {
            "centering": centering_text,
            "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        }
    else:
        # กรณีหา 4 มุมไม่เจอ ให้ตีกล่องแดงบนรูปเดิม
        x, y, w, h = cv2.boundingRect(card_cnt)
        output_img = img.copy()
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 0, 255), 6)
        _, buffer = cv2.imencode('.jpg', output_img)
        return {
            "centering": "วางการ์ดให้เห็น 4 มุมบนพื้นสีตัดกัน",
            "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        }