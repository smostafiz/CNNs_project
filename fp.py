import cv2
import numpy as np
import csv
import pandas as pd


def preprocess_faces(image_path):
    """This function is used to preprocess the image, perform face detection,
    and draw bounding boxes around the detected faces."""

    """Load the input image."""
    image = cv2.imread(image_path)

    if image is None:
        print("Error loading image:", image_path)
        return None

    """Convert the image to grayscale."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    """Apply histogram equalization for illumination enhancement."""
    equalized_image = cv2.equalizeHist(gray_image)

    """Load the Haar cascade classifier for face detection."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    """Detect faces using detectMultiScale() with scaleFactor, minNeighbors, and minSize parameters."""
    faces = face_cascade.detectMultiScale(equalized_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    """If no face is detected, display a message and return."""
    if len(faces) == 0:
        print("Face not found in:", image_path)
        return None

    """Resize the detected faces."""
    resized_faces = []
    for (x, y, w, h) in faces:
        face_roi = equalized_image[y:y + h, x:x + w]
        resized_face = cv2.resize(face_roi, (110, 110))
        resized_faces.append(resized_face)

        """Append the bounding box coordinates to the detection_result data list."""
        ground_truth_data.append([image_path, x, y, x + w, y + h, "face"])

        """Draw red bounding boxes around the detected faces."""
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image, resized_faces


def preprocess_eyes(face_roi):
    """This function is used to preprocess the faces, perform eye detection, blur the eyes,
    and finally display the processed faces."""

    """Enhance the contrast and brightness of the eye region."""
    contrast = 3.7
    brightness = 63
    enhanced_eye = cv2.convertScaleAbs(face_roi, alpha=contrast, beta=brightness)

    """Blur only the enhanced eye to reduce noise."""
    blurred_face = cv2.GaussianBlur(enhanced_eye, (1, 1), sigmaX=10)

    """Use a Haar cascade classifier for eye detection to identify eyes within the face region of interest (ROI)."""
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(blurred_face)

    """If less than two eyes are detected, 
    prompt manual eye localization by displaying the face image and allowing the user to click on the eye locations."""
    if len(eyes) != 2:
        print("Eyes not found. Manually click on the locations of the eyes.")
        cv2.imshow("Manually Locate Eyes", face_roi)
        clicked_points = []

        def mouse_callback(event, x, y, flags, param):
            """Collect clicked points until two points are selected, then close the window."""
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_points.append((x, y))
                if len(clicked_points) == 2:
                    cv2.destroyWindow("Manually Locate Eyes")

        cv2.setMouseCallback("Manually Locate Eyes", mouse_callback)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        """If only one or no points were clicked, cancel manual eye detection and return None."""
        if len(clicked_points) != 2:
            print("Manual eye detection canceled.")
            return None

        left_eye_center, right_eye_center = clicked_points

    else:
        """If two eyes are detected, calculate eye centers for both eyes."""
        eye_centers = [(ex + ew // 2, ey + eh // 2) for (ex, ey, ew, eh) in eyes]
        left_eye_center, right_eye_center = eye_centers

    """Calculate the rotation angle based on the positions of the eyes and rotate the face ROI accordingly."""
    angle = np.degrees(np.arctan2(right_eye_center[1] - left_eye_center[1], right_eye_center[0] - left_eye_center[0]))
    rows, cols = face_roi.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_face = cv2.warpAffine(face_roi, rotation_matrix, (cols, rows))

    """Blur the eyes in the rotated face to anonymize them."""
    for (ex, ey, ew, eh) in eyes:
        eye_x, eye_y, eye_w, eye_h = ex, ey, ew, eh
        eye_roi = rotated_face[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w]
        blurred_eye = cv2.GaussianBlur(eye_roi, (0, 0), sigmaX=30)
        rotated_face[eye_y:eye_y + eye_h, eye_x:eye_x + eye_w] = blurred_eye

        """Append the bounding box coordinates of the blurred eyes to the detection_result data list."""
        ground_truth_data.append([image_path, eye_x, eye_y, eye_x + eye_w, eye_y + eye_h, "eye"])

    return rotated_face


"""Initialize a list to store ground truth bounding box coordinates."""
ground_truth_data = []
image_path = "multiple_fullbody_frontal_faraway.jpg"

"""Display the original image with red bounding boxes around faces and the processed faces with blurred eyes."""
result, processed_faces = preprocess_faces(image_path)

if result is not None:
    cv2.imshow("Face Detection", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for idx, processed_face in enumerate(processed_faces):
        processed_face = preprocess_eyes(processed_face)  # Apply eye blurring here
        cv2.imshow(f"Processed Face {idx + 1}", processed_face)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

"""Define the CSV file path for saving detection results data."""
csv_file_path = "detection_result.csv"

"""Write the detection result data (bounding box coordinates and class labels) to a CSV file."""
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Image_Path", "XMin", "YMin", "XMax", "YMax", "Class"])
    for bbox in ground_truth_data:
        csv_writer.writerow(bbox)

"""Display a message confirming that the detection result data has been saved."""
print("Detection result data saved to:", csv_file_path)

"""Append the detection result data (bounding box coordinates and class labels) to a CSV file."""
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    if csv_file.tell() == 0:
        csv_writer.writerow(["Image_Path", "XMin", "YMin", "XMax", "YMax", "Class"])
    for bbox in ground_truth_data:
        csv_writer.writerow(bbox)

"""Display a message confirming that the detection result data has been saved."""
print("Detection result data saved to:", csv_file_path)

"""Load ground truth data and detection data."""
ground_truth_df = pd.read_csv("ground_truth.csv")
detection_df = pd.read_csv("detection_result.csv")

"""Initialize variables."""
true_positives = 0
false_positives = 0
false_negatives = 0

"""Loop through each row in the ground truth data."""
for _, gt_row in ground_truth_df.iterrows():
    gt_image_path = gt_row["Image_Path"]
    gt_xmin = gt_row["XMin"]
    gt_ymin = gt_row["YMin"]
    gt_xmax = gt_row["XMax"]
    gt_ymax = gt_row["YMax"]
    gt_class = gt_row["Class"]

    """Check if there is a corresponding detection in the detection data."""
    matching_detection = detection_df[
        (detection_df["Image_Path"] == gt_image_path)
        & (detection_df["Class"] == gt_class)
        & (detection_df["XMin"] == gt_xmin)
        & (detection_df["YMin"] == gt_ymin)
        & (detection_df["XMax"] == gt_xmax)
        & (detection_df["YMax"] == gt_ymax)
    ]

    if not matching_detection.empty:
        true_positives += 1
    else:
        false_negatives += 1

"""Calculate false positives."""
false_positives = len(detection_df) - true_positives

"""Calculate precision, recall, F1 score, and accuracy."""
precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)
f1_score = 2 * (precision * recall) / (precision + recall)
total_samples = len(ground_truth_df)
accuracy = true_positives / total_samples

"""Print the results."""
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Accuracy:", accuracy)
