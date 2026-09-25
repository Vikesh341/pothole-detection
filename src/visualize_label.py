import cv2

image_path = "dataset/train/images/img-10_jpg.rf.2e100826fee939c2cc5162e2b1431fa2.jpg"
label_path = "dataset/train/labels/img-10_jpg.rf.2e100826fee939c2cc5162e2b1431fa2.txt"

image = cv2.imread(image_path)

with open(label_path, "r") as file:
    line = file.readline().strip()

class_id, center_x, center_y, width, height = map(float, line.split())

image_height, image_width = image.shape[:2]

center_x *= image_width
center_y *= image_height
width *= image_width
height *= image_height

x1 = int(center_x - width / 2)
y1 = int(center_y - height / 2)
x2 = int(center_x + width / 2)
y2 = int(center_y + height / 2)

cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.putText(
    image,
    "pothole",
    (x1, y1 - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)

cv2.imshow("YOLO Annotation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()