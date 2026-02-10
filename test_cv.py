import cv2

img = cv2.imread("test.jpg")
if img is None:
    raise RuntimeError("Image not found")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show both images
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
