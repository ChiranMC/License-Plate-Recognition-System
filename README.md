# Computer Vision Coursework [Part-B]
# License Plate Recognition with YOLOv8 + OCR  
This project was made for the completion of my computer vision module on my campus.

This project implements **License Plate Recognition (LPR)** using **YOLOv8** for plate detection and **EasyOCR** for character recognition. It supports **training custom models**, **evaluating performance**, and running inference on **images, video files, or live webcam streams**.  

---

## Installation  

Clone the repository and install dependencies:  

```bash
pip install ultralytics opencv-python matplotlib lxml easyocr scikit-learn
```

---

## Dataset Preparation  

1. Place your dataset in the following structure:  
```
LPR_DataSet/
├── annotations/   # XML annotation files (Pascal VOC format)
├── images/        # Corresponding images
```

2. Convert annotations from **VOC XML → YOLO format**:  
The provided script converts bounding boxes to YOLO format and saves them into `yolo_annotations/`.  

3. Split dataset into **train/val sets**:  
The script automatically organizes files into:  
```
train/
  ├── images/
  ├── labels/
val/
  ├── images/
  ├── labels/
```

---

## Training  

Train two YOLOv8 models for comparison:  

```python
from ultralytics import YOLO

# Train YOLOv8n (Nano)
model_n = YOLO("yolov8n.pt")
model_n.train(data="data.yaml", epochs=20, imgsz=640, name="lpr_yolov8n")

# Train YOLOv8s (Small)
model_s = YOLO("yolov8s.pt")
model_s.train(data="data.yaml", epochs=20, imgsz=640, name="lpr_yolov8s")
```

---

## Evaluation  

Evaluate trained models and extract metrics:  

```python
metrics_n = model_n.val()
metrics_s = model_s.val()

print("Precision:", metrics_n.box.p)
print("Recall:", metrics_n.box.r)
print("F1-score:", metrics_n.box.f1)
print("mAP@0.5:", metrics_n.box.map50)
print("mAP@0.5:0.95:", metrics_n.box.map)
```

---

## Inference  

### On Images  
```python
results = model.predict(source="final_test_data/image_test_footage/test_image.png", show=True, save=True)
```

---

### On Video Files  
```python
video_path = "final_test_data/video_test_footage/video_test_footage_1.mp4"
results = model.predict(source=video_path, show=True, save=True)
```

---

### On Live Webcam + OCR  
The system detects plates using YOLOv8, preprocesses them (grayscale, blur, threshold), and extracts text with **EasyOCR**.  
A majority-vote mechanism smooths noisy OCR results.  

Run with:  

```python
python live_lpr.py
```

This will:  
- Open webcam  
- Detect plates in real-time  
- Apply OCR on cropped regions  
- Display recognized text on screen  

---

## Results  

YOLOv8n (Nano) performed better than YOLOv8s (Small) in this dataset:  

| Model   | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
|---------|-----------|--------|----------|---------|--------------|
| YOLOv8n | 0.9027    | 0.9032 | 0.9030   | 0.941   | 0.590        |
| YOLOv8s | 0.8621    | 0.8495 | 0.8557   | 0.894   | 0.566        |

---

## Features  

-  Automatic dataset conversion (VOC → YOLO)  
-  Custom YOLOv8 training (Nano & Small models)  
-  Model evaluation with P, R, F1, mAP  
-  Inference on images, video, and live webcam  
-  OCR integration with **EasyOCR** + preprocessing (grayscale, blur, threshold)  
-  Noisy predictions handled with majority-vote smoothing  
