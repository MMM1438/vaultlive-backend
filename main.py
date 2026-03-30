import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Invalid image format"}

    # --- 1. เตรียมภาพด้วย Bilateral Filter (ลบ Noise แต่ขอบยังคม) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 9 = Diameter, 75 = Color Sigma, 75 = Space Sigma (ค่ามาตรฐานสำหรับรักษาขอบ)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # --- 2. ใช้ Scharr Operator หา Gradient (หาขอบที่จางมากๆ) ---
    gradX = cv2.Scharr(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Scharr(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # --- 3. Morphological Closing (เชื่อมเส้นขอบที่ขาดให้เป็นแผ่นเดียว) ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 4. ค้นหาและคัดเลือก Contour ที่ "ทรงเหมือนการ์ด" ---
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    max_area = 0
    img_area = img.shape[0] * img.shape[1]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # ต้องมีขนาดอย่างน้อย 15% ของรูป เพื่อกันเศษฝุ่นหรือวัตถุอื่นๆ
        if area < (img_area * 0.15): continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        
        # สัดส่วนการ์ดแนวตั้ง ~0.7, แนวนอน ~1.4 (เผื่อระยะขอบนิดหน่อย 0.5 - 1.8)
        if 0.5 < aspect_ratio < 1.8:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    # --- 5. ประมวลผล Centering บนรูป RAW ---
    output_img = img.copy()
    centering_result = "ไม่พบขอบการ์ดที่ชัดเจน"

    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        
        # หาขอบใน (Artwork) เฉพาะในบริเวณที่เจอการ์ด
        roi_gray = gray[y:y+h, x:x+w]
        # ใช้ Canny แบบละเอียด
        inner_edged = cv2.Canny(roi_gray, 50, 150)
        inner_contours, _ = cv2.findContours(inner_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        best_inner = None
        for icnt in inner_contours:
            ix, iy, iw, ih = cv2.boundingRect(icnt)
            iarea = iw * ih
            # Artwork ควรมีพื้นที่ประมาณ 35% - 80% ของตัวการ์ด
            if (w * h * 0.35) < iarea < (w * h * 0.85):
                # ต้องอยู่ใกล้จุดศูนย์กลางของ ROI
                if abs((ix + iw/2) - w/2) < (w * 0.2) and abs((iy + ih/2) - h/2) < (h * 0.2):
                    best_inner = (ix + x, iy + y, iw, ih)
                    break

        # วาด Guide Lines
        cv2.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 6) # ขอบนอกเขียว
        
        if best_inner:
            ix, iy, iw, ih = best_inner
            cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 8) # ขอบในทอง
            
            # คำนวณระยะห่าง 4 ทิศ
            l, r = ix - x, (x + w) - (ix + iw)
            t, b = iy - y, (y + h) - (iy + ih)
            
            lr = round(l / (l + r) * 100) if (l + r) > 0 else 50
            tb = round(t / (t + b) * 100) if (t + b) > 0 else 50
            centering_result = f"L/R: {lr}/{100-lr} | T/B: {tb}/{100-tb}"
        else:
            centering_result = "เจอการ์ดแต่หา Artwork ไม่พบ"

    # --- 6. ส่งรูปกลับ ---
    _, buffer = cv2.imencode('.jpg', output_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "centering": centering_result,
        "visual_result": f"data:image/jpeg;base64,{img_base64}"
    }