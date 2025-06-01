import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
from googletrans import Translator

model = load_model('action5.h5')
params = np.load('preprocess_params5.npz')
mean, std = params['mean'], params['std']
actions = np.array([
    'use', 'crop', 'seed', 'area', 'for', 'sowing', 'disease', 'in',
    'lower', 'half', 'leaves', 'fertilize', 'with',
    'quantity', 'medicine', 'spray', 'mixed', 'of', 'water'
])

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

translator = Translator()
latest_sentence = ""
latest_hindi = ""
sentence = []  # Make sentence a global variable

def extract_keypoints(results):
    pose_results, hand_results = results
    pose = np.array([[lm.x, lm.y, lm.z] for lm in
                     pose_results.pose_landmarks.landmark]).flatten() if pose_results.pose_landmarks else np.zeros(33 * 3)

    hands = []
    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            hands.extend([[lm.x, lm.y, lm.z] for lm in hand.landmark])
        hands = np.array(hands).flatten()
        if len(hands) < 126:
            hands = np.concatenate([hands, np.zeros(126 - len(hands))])
    else:
        hands = np.zeros(126)

    return np.concatenate([pose, hands])

def mediapipe_detection(image, pose, hands):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    pose_results = pose.process(image_rgb)
    hand_results = hands.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, (pose_results, hand_results)

def update_latest(sentence, hindi):
    global latest_sentence, latest_hindi
    latest_sentence = sentence
    latest_hindi = hindi

def get_latest_sentence():
    return latest_sentence

def get_latest_hindi():
    return latest_hindi

def clear_last_word():
    global latest_sentence, latest_hindi, sentence
    # Split the sentence into words and remove the last one
    words = latest_sentence.strip().split()
    if words:
        words.pop()  # Remove the last word
        sentence = words  # Update the global sentence list
        latest_sentence = " ".join(words)
        # Update the Hindi translation
        if latest_sentence:
            try:
                latest_hindi = translator.translate(latest_sentence, src='en', dest='hi').text
            except Exception as e:
                latest_hindi = "Translation Error"
        else:
            latest_hindi = ""
    else:
        latest_sentence = ""
        latest_hindi = ""
        sentence = []

def clear_all():
    global latest_sentence, latest_hindi, sentence
    latest_sentence = ""
    latest_hindi = ""
    sentence = []  # Reset the global sentence list

def run_detection():
    global sentence
    sequence = []
    predictions = []
    threshold = 0.90
    history_length = 15

    print("[INFO] Starting detection... Press 'q' to stop.")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:

        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, pose, hands)
            keypoints = extract_keypoints(results)

            # Normalize keypoints and check shape
            norm_keypoints = (keypoints - mean) / (std + 1e-8)
            if norm_keypoints.shape == (225,):  # Adjust shape if needed
                sequence.append(norm_keypoints)
            else:
                print(f"Warning: Skipping frame with unexpected keypoints shape {norm_keypoints.shape}")
                continue

            sequence = sequence[-30:]

            if len(sequence) == 30 and all(kp.shape == (225,) for kp in sequence):
                input_data = np.array(sequence)
                res = model.predict(np.expand_dims(input_data, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))

                if len(predictions) >= history_length:
                    last_preds = predictions[-history_length:]
                    confidence = np.mean([res[p] for p in last_preds])

                    if confidence > threshold:
                        current_action = actions[np.argmax(res)]
                        if len(sentence) == 0 or sentence[-1] != current_action:
                            sentence.append(current_action)
                            if len(sentence) > 20:
                                sentence = sentence[-20:]

                            full_sentence = ' '.join(sentence)
                            try:
                                translated = translator.translate(full_sentence, src='en', dest='hi').text
                            except Exception as e:
                                translated = "Translation Error"

                            update_latest(full_sentence, translated)

            cv2.imshow('Sign Detection - Press Q to exit', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Detection stopped.")