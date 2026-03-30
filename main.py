import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import base64

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def stabilize_points(pts):
    """ เรียงจุดให้แม่นยำที่สุด: [บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย] """
    pts = pts.reshape((4, 2))
    new_pts = np.zeros((4, 2), dtype="float32")
    
    # บนซ้ายคือจุดที่ sum (x+y) น้อยสุด, ล่างขวาคือจุดที่ sum มากสุด
    s = pts.sum(axis=1)
    new_pts[0] = pts[np.argmin(s)]
    new_pts[2] = pts[np.argmax(s)]
    
    # บนขวาคือจุดที่ diff (y-x) น้อยสุด, ล่างซ้ายคือจุดที่ diff มากสุด
    diff = np.diff(pts, axis=1)
    new_pts[1] = pts[np.argmin(diff)]
    new_pts[3] = pts[np.argmax(diff)]
    
    return new_pts

@app.post("/analyze")
async def analyze_card(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return {"error": "Invalid image"}

    # --- Pre-processing ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # เชื่อมเส้นขอบที่ขาด
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return {"error": "หาการ์ดไม่เจอ"}

    # เลือกสี่เหลี่ยมที่ใหญ่ที่สุด
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c) # หา Rect ที่หมุนได้ (ช่วยเรื่อง Rotation)
    box = cv2.boxPoints(rect)
    box = stabilize_points(box)

    # คำนวณขนาดจริงของการ์ด (เพื่อกันรูปยืด/บาง)
    (tl, tr, br, bl) = box
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    # ทำ Warp Perspective
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box, dst)
    warped = cv2.warpPerspective(img, M, (width, height))

    # --- เช็ค Rotation: ถ้ามาแนวนอน ให้หมุนเป็นแนวตั้ง ---
    if width > height:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        width, height = height, width

    # --- หาขอบใน (Centering) ---
    # (ใช้ Logic เดิมที่คุณมี แต่รันบนรูปที่ดัดตรงแล้ว)
    # ... (ส่วนหาขอบในและวาดเส้น) ...

    _, buffer = cv2.imencode('.jpg', warped)
    return {
        "centering": "Ready to measure", 
        "visual_result": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
    }