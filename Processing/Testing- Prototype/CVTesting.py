import cv2
import numpy as np

# Load user's drawing and correct racing line images
drawing_img = cv2.imread('user_drawing.png')
racing_line_img = cv2.imread('correct_racing_line.png')

# Convert images to grayscale
drawing_gray = cv2.cvtColor(drawing_img, cv2.COLOR_BGR2GRAY)
racing_line_gray = cv2.cvtColor(racing_line_img, cv2.COLOR_BGR2GRAY)

# Apply edge detection to images
drawing_edges = cv2.Canny(drawing_gray, 100, 200)
racing_line_edges = cv2.Canny(racing_line_gray, 100, 200)

# Find contours in the images
drawing_contours, _ = cv2.findContours(drawing_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
racing_line_contours, _ = cv2.findContours(racing_line_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour in the user's drawing
drawing_contour = max(drawing_contours, key=cv2.contourArea)

# Find the bounding rectangle of the largest contour in the user's drawing
drawing_rect = cv2.boundingRect(drawing_contour)

# Crop the user's drawing to the bounding rectangle
drawing_cropped = drawing_gray[drawing_rect[1]:drawing_rect[1]+drawing_rect[3], drawing_rect[0]:drawing_rect[0]+drawing_rect[2]]

# Resize the cropped user's drawing to match the size of the racing line image
drawing_resized = cv2.resize(drawing_cropped, racing_line_gray.shape[::-1])

# Calculate the mean squared error between the user's drawing and the correct racing line
mse = np.mean((drawing_resized.astype("float") - racing_line_gray.astype("float")) ** 2)

# Print the mean squared error
print("Mean Squared Error:", mse)
