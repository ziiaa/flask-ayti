from flask import Flask, jsonify, request, render_template_string

import os
import tempfile
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("best.pt")
names = model.model.names

prev_frame = None  # Variable untuk menyimpan frame sebelumnya
prev_keypoints = None  # Variable untuk menyimpan keypoints sebelumnya


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')


def detect_objects_from_webcam():
    global prev_frame, prev_keypoints

    count = 0
    cap = cv2.VideoCapture(0)  # 0 untuk webcam default
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue

        # Resize frame ke ukuran tertentu (1020, 600)
        frame = cv2.resize(frame, (1020, 600))

        # Jalankan YOLOv8 tracking pada frame
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[class_id]
                x1, y1, x2, y2 = box

                # Ukuran teks
                (text_width, text_height), baseline = cv2.getTextSize(
                    f'{track_id} - {label}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Posisi kotak latar belakang
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)

                # Teks dengan latar belakang
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Kotak deteksi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Optical Flow untuk tracking
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is None or prev_keypoints is None:
            prev_frame = frame_gray.copy()
            prev_keypoints = cv2.goodFeaturesToTrack(
                frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
        else:
            try:
                if prev_frame.shape == frame_gray.shape:
                    matched_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_frame, frame_gray, prev_keypoints, None)
                    
                    # Pastikan matched_keypoints tidak kosong dan sesuai dimensi
                    if matched_keypoints is not None and status is not None and len(matched_keypoints) == len(status):
                        prev_keypoints = matched_keypoints[status.flatten() == 1]
                    else:
                        prev_keypoints = None

                    prev_frame = frame_gray.copy()
                else:
                    # Reset jika ukuran frame tidak sama
                    prev_frame = frame_gray.copy()
                    prev_keypoints = cv2.goodFeaturesToTrack(
                        frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
            except cv2.error as e:
                print(f"Error in Optical Flow: {e}")
                prev_frame = frame_gray.copy()
                prev_keypoints = cv2.goodFeaturesToTrack(
                    frame_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)

        # Encode frame ke JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save video ke file sementara
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    # Redirect ke video feed
    return redirect(url_for('video_feed_temp', temp_filename=temp_path))


@app.route('/video_feed_temp/<path:temp_filename>')
def video_feed_temp(temp_filename):
    return Response(detect_objects_from_video(temp_filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def detect_objects_from_video(video_path):
    global prev_frame, prev_keypoints

    count = 0
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 2 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                label = names[class_id]
                x1, y1, x2, y2 = box

                # Ukuran teks
                (text_width, text_height), baseline = cv2.getTextSize(
                    f'{track_id} - {label}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Posisi kotak latar belakang
                cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)

                # Teks dengan latar belakang
                cv2.putText(frame, f'{track_id} - {label}', (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Kotak deteksi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    os.remove(video_path)  # Hapus file sementara


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Simpan gambar ke file sementara
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    # Baca gambar
    frame = cv2.imread(temp_path)
    frame = cv2.resize(frame, (1020, 600))  # Sesuaikan ukuran gambar

    # Jalankan YOLOv8 untuk deteksi
    results = model(frame)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for box, class_id in zip(boxes, class_ids):
            label = names[class_id]
            x1, y1, x2, y2 = box

            # Ukuran teks
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Posisi kotak latar belakang
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 0, 0), -1)

            # Teks dengan latar belakang
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Kotak deteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Encode frame ke JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    response = Response(buffer.tobytes(), content_type='image/jpeg')

    # Hapus file sementara
    os.remove(temp_path)

    return response


if __name__ == '__main__':
    app.run(debug=True)
