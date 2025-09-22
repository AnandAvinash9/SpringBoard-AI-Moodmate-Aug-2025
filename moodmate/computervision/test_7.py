
import cv2

# Load an image
img = cv2.imread("C:/Users/avins/Downloads/me.PNG")   # Replace with your file path

# Check if image loaded correctly
if img is None:
    print("Error: Could not read image.")
    exit()

# Apply Gaussian Blur (15x15 kernel)
blur = cv2.GaussianBlur(img, (15, 15), 0)

# Show both
cv2.imshow("Original", img)
cv2.imshow("Blurred", blur)

# Wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Save the blurred image
cv2.imwrite("blurred_output.jpg", blur)
