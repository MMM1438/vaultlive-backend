import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def order_points(pts):
    """ฟังก์ชันเรียงพิกัด 4 มุม: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # บนซ้าย
    rect[2] = pts[np.argmax(s)] # ล่างขวา
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # บนขวา
    rect[3] = pts[np.argmax(diff)] # ล่างซ้าย
    return rect

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Invalid image"}

    # --- Step 1: Pre-processing สู้แสงสะท้อน ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ใช้ CLAHE ดึง Contrast ให้ขอบขาวเด้งออกมาแม้แสงจ้า
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # เบลอเพื่อลด Noise พื้นผิว แต่รักษาความคมของขอบการ์ดไว้
    blurred = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)

    # --- Step 2: ค้นหาขอบนอก (Outer Edge) ---
    v = np.median(blurred)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edged = cv2.Canny(blurred, lower, upper)
    edged = cv2.dilate(edged, np.ones((3,3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: 
        _, buf = cv2.imencode('.jpg', img)
        return {"centering": "หาการ์ดไม่เจอ", "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}"}

    # หาสี่เหลี่ยมที่ใหญ่ที่สุดที่มี 4 มุม
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    card_approx = None
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > 50000: # ขนาดต้องใหญ่พอสมควร
            card_approx = approx
            break

    # --- Step 3: ดัดภาพและวัด Centering (The Pro Way) ---
    if card_approx is not None:
        rect = order_points(card_approx.reshape(4, 2))
        
        # สัดส่วนการ์ดมาตรฐาน (กว้าง 500px, สูง 700px)
        maxWidth, maxHeight = 500, 700 
        dst = np.array([
            [0, 0], [maxWidth - 1, 0], 
            [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
        ], dtype="float32")
        
        # รีดภาพให้แบนราบ
        M = cv2.getPerspectiveTransform(rect, dst)
        warped_color = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        warped_gray = cv2.warpPerspective(enhanced_gray, M, (maxWidth, maxHeight))

        # หาขอบ Artwork จากรูปที่ถูกรีดตรงแล้ว (แม่นยำระดับพิกเซล)
        w_thresh = cv2.adaptiveThreshold(warped_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        inner_cnts, _ = cv2.findContours(w_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        best_inner = None
        max_inner_area = 0
        
        for cnt in inner_cnts:
            ix, iy, iw, ih = cv2.boundingRect(cnt)
            area = iw * ih
            # Artwork ปกติจะกินพื้นที่ 40% - 85% ของการ์ด
            if (maxWidth * maxHeight * 0.4) < area < (maxWidth * maxHeight * 0.85):
                # ป้องกันไม่ให้ไปจับตัวหนังสือขอบๆ
                if abs((ix + iw/2) - maxWidth/2) < maxWidth*0.2 and abs((iy + ih/2) - maxHeight/2) < maxHeight*0.2:
                    if area > max_inner_area:
                        max_inner_area = area
                        best_inner = (ix, iy, iw, ih)

        output_img = warped_color.copy()
        
        if best_inner:
            ix, iy, iw, ih = best_inner
            
            # การวัดแบบนี้จะแม่นยำ 100% เพราะรูปถูกดัดตรงแล้ว
            left, right = ix, maxWidth - (ix + iw)
            top, bottom = iy, maxHeight - (iy + ih)
            
            lr_ratio = round((left / (left + right)) * 100) if (left + right) > 0 else 50
            tb_ratio = round((top / (top + bottom)) * 100) if (top + bottom) > 0 else 50
            centering_text = f"L/R: {lr_ratio}/{100-lr_ratio} | T/B: {tb_ratio}/{100-tb_ratio}"

            # วาดเส้นไกด์ไลน์ขอบใน (สีทอง)
            cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 4)
            # ขีดเส้นกางเขนกึ่งกลาง (สีเขียวบางๆ) 
            cv2.line(output_img, (int(maxWidth/2), 0), (int(maxWidth/2), maxHeight), (0, 255, 0), 1)
            cv2.line(output_img, (0, int(maxHeight/2)), (maxWidth, int(maxHeight/2)), (0, 255, 0), 1)
        else:
            centering_text = "Analysis Failed: ไม่พบขอบ Artwork ด้านใน"

        _, buffer = cv2.imencode('.jpg', output_img)
        return {
            "centering": centering_text,
            "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        }
    else:
        # กรณีหา 4 มุมไม่เจอ ให้ตีกล่องแดงเตือนบนรูปเดิม
        output_img = img.copy()
        centering_text = "วางการ์ดให้เห็น 4 มุมบนพื้นสีตัดกัน"
        _, buffer = cv2.imencode('.jpg', output_img)
        return {
            "centering": centering_text,
            "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        }