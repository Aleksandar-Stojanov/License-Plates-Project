# !pip install easyocr
# !sudo apt-get install tesseract-ocr
# !pip install pytesseract
# !pip install imutils

import pytesseract
import cv2
import glob
import os
import easyocr
import numpy as np
import imutils
from difflib import SequenceMatcher
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\pc\PycharmProjects\FinalLicensePlatesProject\venv\Scripts\pytesseract.exe"

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            roi = img[y:y + h, x:x + w]

            text = pytesseract.image_to_string(roi, lang='eng', config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

            if text and len(text) > 4:
                location = approx
                break


    mask = np.zeros(gray.shape, np.uint8)

    if location is not None:
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)
        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = gray[x1:x2 + 1, y1:y2 + 1]

        if cropped_image.size != 0:
            filename = os.path.basename(input_path)
            output_file = os.path.join(output_path, filename)
            cv2.imwrite(output_file, cropped_image)
            print(f"Processed and saved: {output_file}")
        else:
            print("Cropped image is empty.")
    else:
        print("No license plate found in:", input_path)

def calculate_predicted_similarity(actual_list, predicted_list):
    similarities = []

    for actual_plate, predict_plate in zip(actual_list, predicted_list):
        similarity_ratio = SequenceMatcher(None, actual_plate, predict_plate).ratio()
        similarities.append(similarity_ratio)

    return similarities


def calculate_cumulative_similarity(actual_list, predicted_list):
    total_similarity = 0

    for actual_plate, predict_plate in zip(actual_list, predicted_list):
        similarity_ratio = SequenceMatcher(None, actual_plate, predict_plate).ratio()
        total_similarity += similarity_ratio

    average_similarity = total_similarity / len(actual_list)

    return average_similarity

input_folder = './downloaded_images'
output_folder = './processed_images'

os.makedirs(output_folder, exist_ok=True)

for path_to_license_plate in os.listdir(input_folder):
    if path_to_license_plate.endswith('.jpg'):
        input_path = os.path.join(input_folder, path_to_license_plate)
        process_image(input_path, output_folder)
path_for_license_plates = r"/content/drive/MyDrive/Colab Notebooks/LicensePlatesProject/processed_images/*.jpg"

best_psm_value = 0
best_accuracy = 0
for i in range(6,14):
  list_license_plates = []
  predicted_license_plates = []
  for path_to_license_plate in glob.glob(path_for_license_plates, recursive=True):

      license_plate_file = path_to_license_plate.split("/")[-1]
      license_plate, _ = os.path.splitext(license_plate_file)

      list_license_plates.append(license_plate)


      img = cv2.imread(path_to_license_plate)

      predicted_result = pytesseract.image_to_string(img, lang='eng', config=f'--oem 3 --psm {i} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
      filter_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")
      predicted_license_plates.append(filter_predicted_result)
      average_similarity = calculate_cumulative_similarity(list_license_plates, predicted_license_plates)
  if best_accuracy < average_similarity:
    best_prediction=[]
    best_psm_value = i
    best_accuracy = average_similarity
    best_prediction = [i for i in predicted_license_plates]

similarities = calculate_predicted_similarity(list_license_plates, best_prediction)
for actual_plate, predict_plate, similarity in zip(list_license_plates, best_prediction, similarities):
    print(f"Actual: {actual_plate}\tPredicted: {predict_plate}\tSimilarity: {similarity:.2f}")

average_similarity = calculate_cumulative_similarity(list_license_plates, best_prediction)

print(f"Average Similarity: {average_similarity:.2%}")