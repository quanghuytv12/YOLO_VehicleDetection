from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from math import sqrt
import torch
from torchvision.ops import nms


# Load the fine-tuned YOLO model
model = YOLO(r'yolov8_train2.pt')  # Đường dẫn đến mô hình đã fine-tuning

def calculate_distance(box1, box2):
    """
    Tính khoảng cách Euclidean giữa tâm của hai bounding box.
    """
    x1, y1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
    x2, y2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
    return sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_warning(frame, box1, box2, distance):
    """
    Tô bounding box màu đỏ và hiển thị text cảnh báo trên từng khung hình video.
    """
    # Vẽ bounding box màu đỏ
    cv2.rectangle(frame, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 0, 255), 2)
    cv2.rectangle(frame, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (0, 0, 255), 2)

    # Hiển thị text cảnh báo ở giữa hai box
    center_x = int((box1[0] + box2[0]) / 2)
    center_y = int((box1[1] + box2[1]) / 2)
    cv2.putText(frame, f"Warning: {distance:.2f}px", (center_x, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def process_image(image_path):
    image = cv2.imread(image_path)
    results = model.predict(source=image, imgsz=640)
    annotated_image = results[0].plot()
    
    # Hiển thị kết quả
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Lưu kết quả
    output_image_path = r'D:\UIT\Code\Python\Code\DA2_Quantization\1.png'
    cv2.imwrite(output_image_path, annotated_image)
    print("Annotated image saved to:", output_image_path)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(r'D:\UIT\Nam4\video.mp4', fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=640)
        annotated_frame = results[0].plot()

        # Lấy tất cả bounding box (dạng [x_min, y_min, x_max, y_max])
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Chuyển sang numpy để dễ xử lý

        # Duyệt qua từng cặp bounding box và kiểm tra khoảng cách
        for i, box1 in enumerate(boxes):
            for j, box2 in enumerate(boxes):
                if i != j:  # Không so sánh chính nó
                    distance = calculate_distance(box1, box2)
                    if distance < 5:
                        print(f"Warning! Objects {i} and {j} are too close. Distance: {distance:.2f}")
                        draw_warning(frame, box1, box2, distance)  # Vẽ cảnh báo lên khung hình

        # Ghi từng khung hình vào video kết quả
        out.write(frame)
        # Hiển thị cả hai khung hình trong hai cửa sổ
        cv2.namedWindow('Annotated Frame (YOLO)', cv2.WINDOW_NORMAL)  # Đảm bảo cửa sổ có thể được điều chỉnh
        cv2.namedWindow('Warning Frame (With Alerts)', cv2.WINDOW_NORMAL)

        cv2.imshow('Annotated Frame (YOLO)', annotated_frame)
        cv2.imshow('Warning Frame (With Alerts)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Annotated video saved to:", r'D:\UIT\Nam4\video.mp4')

def process_video_with_nms_and_class(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO prediction
        results = model.predict(source=frame, imgsz=640)

        # Lấy bounding boxes, confidence scores và class IDs
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

        # Áp dụng NMS
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

        # Lọc kết quả
        filtered_boxes = boxes_tensor[keep_indices].numpy()
        filtered_scores = scores_tensor[keep_indices].numpy()
        filtered_class_ids = class_ids[keep_indices.numpy()]

        # Vẽ các bounding box và tên class
        for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids):
            x1, y1, x2, y2 = box.astype(int)
            class_names = model.names
            class_name = class_names[class_id]  # Tên class tương ứng
            label = f"{class_name} {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Bounding box màu xanh lá
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow('Frame with NMS and Class', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# D:/UIT/Do_An2/opencv-example-main/test.mp4  D:/UIT/Do_An2/pothole_test1.jpg
input_path = r"D:\Download\Riding a Motorbike in Saigon (Ho Chi Minh City) Vietnam.mp4"  # Đường dẫn tới ảnh hoặc video

# Kiểm tra và xử lý loại tệp
if input_path.endswith(('.jpg', '.png', '.jpeg')):
    process_image(input_path)
elif input_path.endswith(('.mp4', '.avi', '.mov')):
    process_video(input_path)
else:
    print("Unsupported file format.")
