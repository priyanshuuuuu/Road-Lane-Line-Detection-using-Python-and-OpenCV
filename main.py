import cv2
import numpy as np

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 150, 250)

    height, width = edges.shape
    mask = np.zeros_like(edges)
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]
    cv2.fillPoly(mask, np.array([region_of_interest_vertices], np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=6,
        theta=np.pi / 60,
        threshold=230,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25,
    )

    line_image = np.zeros((height, width, 3), dtype=np.uint8)
    draw_lines(line_image, lines)

    combined_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    return combined_image

def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

cap = cv2.VideoCapture('test.mp4')  
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = detect_lanes(frame)
    cv2.imshow('Lane Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
