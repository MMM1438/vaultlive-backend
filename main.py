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
    if img is None: return {"error": "Invalid image"}

    # --- 1. ยกระดับการหาขอบ (Advanced Edge Detection) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Bilateral Filter เพื่อลด Noise แต่ "รักษาความคมของขอบ" (ดีกว่า Gaussian)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # ใช้ Scharr Operator หาการเปลี่ยนแปลงของแสง (Gradient) ทั้งแนวตั้งและแนวนอน
    gradX = cv2.Scharr(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Scharr(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    
    # ปิดช่องว่างของเส้นขอบด้วย Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, kernel)
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 2. ค้นหาสี่เหลี่ยมที่ "หน้าตาเหมือนการ์ด" ที่สุด ---
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    max_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50000: continue # ข้ามเศษขยะเล็กๆ
        
        # ตรวจสอบสัดส่วน (Aspect Ratio)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        
        # คะแนนความเป็นการ์ด: สัดส่วนควรอยู่ระหว่าง 0.6 - 0.8 (หรือ 1.2 - 1.6 แนวนอน)
        score = 0
        if 0.5 < aspect_ratio < 1.8: score += 50
        if len(approx) == 4: score += 50 # ถ้าเป็นสี่เหลี่ยมเป๊ะจะได้คะแนนเพิ่ม
        
        if score >= max_score:
            max_score = score
            best_cnt = cnt

    if best_cnt is None:
        # ถ้าหาไม่เจอจริงๆ ให้ถอยกลับไปใช้รูป Raw โดยไม่วาดเส้น
        return {"centering": "หาการ์ดไม่เจอ ลองขยับมุมกล้องครับ", "visual_result": f"data:image/jpeg;base64,{base64.b64encode(contents).decode('utf-8')}"}

    # --- 3. หาขอบใน (Artwork) ด้วย Canny ที่ละเอียดขึ้น ---
    x, y, w, h = cv2.boundingRect(best_cnt)
    roi_gray = gray[y:y+h, x:x+w]
    
    # ใช้ Canny บนตัวการ์ดที่เจอแล้ว
    inner_edged = cv2.Canny(roi_gray, 30, 100)
    inner_contours, _ = cv2.findContours(inner_edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    best_inner = None
    for icnt in inner_contours:
        ix, iy, iw, ih = cv2.boundingRect(icnt)
        iarea = iw * ih
        if (w * h * 0.3) < iarea < (w * h * 0.85):
            # ตรวจสอบว่าอยู่กลางการ์ดไหม
            if abs((ix + iw/2) - w/2) < (w * 0.15) and abs((iy + ih/2) - h/2) < (h * 0.15):
                best_inner = (ix+x, iy+y, iw, ih)
                break

    # --- 4. วาดผลลัพธ์ ---
    output_img = img.copy()
    cv2.drawContours(output_img, [best_cnt], -1, (0, 255, 0), 6) # วาดขอบนอกตามรูปทรงจริง
    
    if best_inner:
        ix, iy, iw, ih = best_inner
        cv2.rectangle(output_img, (ix, iy), (ix + iw, iy + ih), (0, 215, 255), 8)
        
        # คำนวณ Ratio
        l, r = ix - x, (x + w) - (ix + iw)
        t, b = iy - y, (y + h) - (iy + ih)
        lr_p = round(l/(l+r)*100) if (l+r)>0 else 50
        tb_p = round(t/(t+b)*100) if (t+b)>0 else 50
        res_text = f"L/R: {lr_p}/{100-lr_p} T/B: {tb_p}/{100-tb_p}"
    else:
        res_text = "Found Card, but Inner Border hidden"

    _, buffer = cv2.imencode('.jpg', output_img)
    return {
        "centering": res_text,
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }