# Don't need modification here

import os
import cv2

DATA_DIR = './data'
CATEGORIES = ['hands', 'body']  # Two main categories
number_of_classes = 10
dataset_size = 100

cap = cv2.VideoCapture(0)

# Create directories if not exist
for category in CATEGORIES:
    category_path = os.path.join(DATA_DIR, category)
    if not os.path.exists(category_path):
        os.makedirs(category_path)

    for j in range(number_of_classes):
        class_path = os.path.join(category_path, str(j))
        if not os.path.exists(class_path):
            os.makedirs(class_path)

# Loop through categories (Hands & Body)
for category in CATEGORIES:
    print(f"Collecting data for {category}...")

    for j in range(number_of_classes):
        print(f'Collecting data for class {j} in {category}')

        # Ready phase
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, f'Collecting {category} - Class {j}. Press "Q" to start!', 
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        # Data collection phase
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            
            # Save images in respective folders
            img_path = os.path.join(DATA_DIR, category, str(j), f'{counter}.jpg')
            cv2.imwrite(img_path, frame)

            counter += 1

cap.release()
cv2.destroyAllWindows()
