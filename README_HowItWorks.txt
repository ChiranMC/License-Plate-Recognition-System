README - How the License Plate Recognition Project Works

1. Model Training and Evaluation
- YOLOv8 models were trained for detecting license plates.
- Two architectures were used: YOLOv8n (Nano, lightweight) and YOLOv8s (Small, more complex).
- Models were evaluated with metrics such as Precision, Recall, F1-score, and mAP.

2. Live Video Capture
- The code uses OpenCV to access the webcam (cv2.VideoCapture).
- Frames are captured continuously for processing.

3. Plate Detection (YOLOv8)
- Each frame is passed through the trained YOLOv8 model.
- The model identifies bounding boxes around license plates.

4. Cropping the Plate
- Detected plates are cropped out of the frame using bounding box coordinates.
- These cropped images are isolated for text recognition.

5. Preprocessing for OCR
- Cropped plate images are converted to grayscale.
- Thresholding and denoising are applied to reduce noise and improve readability.

6. OCR (Optical Character Recognition)
- Tesseract OCR is used to extract text (characters) from the preprocessed license plate image.
- The recognized text is returned as the license plate number.

7. Displaying Results
- Bounding boxes and extracted plate numbers are drawn on the live video feed.
- The final video feed shows both detected plates and recognized text in real-time.

This project therefore performs: Live video input -> Plate detection -> Preprocessing -> OCR -> Output on video.
