from fastapi import FastAPI, File, UploadFile
import pickle
import cv2
import mediapipe as mp
import numpy as np
import uvicorn
import io
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# Load trained models
model_hands_dict = pickle.load(open('models/model_hands.p', 'rb'))
model_hands = model_hands_dict['model']

model_body_dict = pickle.load(open('models/model_body.p', 'rb'))
model_body = model_body_dict['model']

# Define labels for hand gestures
labels_dict_hands = {
    0: 'Pataka', 1: 'Tripataka', 2: 'Ardhapataka', 3: 'Katarimukha', 4: 'Mayura', 
    5: 'Ardhachandra', 6: 'Arala', 7: 'Sukatunda', 8: 'Musti', 9: 'Sikhara', 
    10: 'Kapittha', 11: 'Katamukha', 12: 'Suchi', 13: 'Chandrakala', 14: 'Padmakosa', 
    15: 'Sarpashirsa', 16: 'Mrgasirsa', 17: 'Simhamukha', 18: 'Kangula', 19: 'Alapadma', 
    20: 'Catura', 21: 'Bhramara', 22: 'Hamsasya', 23: 'Hamsapaksa', 24: 'Sandamsa', 
    25: 'Mukula', 26: 'Tamracuda', 27: 'Trisula', 28: 'Anjali', 29: 'Kapota', 
    30: 'Karkata', 31: 'Svastika', 32: 'Dola', 33: 'Puspaputa', 34: 'Utsanga', 
    35: 'Sivalinga', 36: 'Katakavardhana', 37: 'Kartarisvastika', 38: 'Sakata', 
    39: 'Sankha', 40: 'Chakra', 41: 'Pasa', 42: 'Kilaka', 43: 'Matsya', 44: 'Kurma', 
    45: 'Varaha', 46: 'Garuda', 47: 'Nagabandha', 48: 'Katva', 49: 'Bherunda'
}

# Define labels for body poses
labels_dict_body = {
    0: 'Samapada', 
    1: 'Nagabandha', 
    2: 'Garuda', 
    3: 'Brahma', 
    4: 'Prenkhana', 
    5: 'Swastika', 
    6: 'Muzhumandi'
}

# Setup FastAPI
app = FastAPI()

# Fix CORS issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mediapipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3)


@app.post("/predict/hand/")
async def predict_hand(file: UploadFile = File(...)):
    """Predict hand gestures from an uploaded image."""
    image = Image.open(io.BytesIO(await file.read()))
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data_hands = []
    x_hands, y_hands = [], []

    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x_hands.append(hand_landmarks.landmark[i].x)
                y_hands.append(hand_landmarks.landmark[i].y)

            for i in range(len(hand_landmarks.landmark)):
                data_hands.append(hand_landmarks.landmark[i].x - min(x_hands))
                data_hands.append(hand_landmarks.landmark[i].y - min(y_hands))

        prediction_hands = model_hands.predict([np.asarray(data_hands)])[0]
        predicted_hand_gesture = labels_dict_hands.get(int(prediction_hands), "Unknown Hand Gesture")

        return {"hand_gesture": predicted_hand_gesture}

    return {"hand_gesture": "No hand detected"}


@app.post("/predict/body/")
async def predict_body(file: UploadFile = File(...)):
    """Predict body pose from an uploaded image."""
    image = Image.open(io.BytesIO(await file.read()))
    image = np.array(image)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data_body = []
    x_body, y_body = [], []

    results = pose.process(image_rgb)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            x_body.append(lm.x)
            y_body.append(lm.y)

        min_x_body, min_y_body = min(x_body), min(y_body)

        for i in range(len(results.pose_landmarks.landmark)):
            data_body.append(x_body[i] - min_x_body)
            data_body.append(y_body[i] - min_y_body)

        prediction_body = model_body.predict([np.asarray(data_body)])[0]
        predicted_body_pose = labels_dict_body.get(int(prediction_body), "Unknown Body Pose")

        return {"body_pose": predicted_body_pose}

    return {"body_pose": "No body detected"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
