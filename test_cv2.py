import cv2
import numpy as np
import sys

print(f"Python: {sys.version}")
print(f"OpenCV: {cv2.__version__}")

img = np.zeros((100, 100), dtype=np.uint8)
cv2.rectangle(img, (10, 10), (50, 50), 255, -1)

try:
    ret = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"findContours returned {len(ret)} values")
except Exception as e:
    print(f"Error: {e}")
