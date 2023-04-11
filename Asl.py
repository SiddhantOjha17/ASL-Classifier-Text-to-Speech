import numpy as np
import cv2
import mediapipe as mp
import joblib
import scipy.stats as st
import warnings 
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt')
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

print(joblib.__version__)

# Define the ASL labels
asl_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']

# Initialize variables for sentence
sentence = []
last_sent = None

# Load trained classifier
clf = joblib.load('rf_model.joblib')

# Initialize hand tracking model
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
# Initialize video capture device
cap = cv2.VideoCapture(0)
prediction = []

while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print('Unable to read frame from video capture device')
        break

    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use the hand tracking model to detect hand landmarks
    results = hands.process(image)

    # Extract the hand landmarks
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark

        # Convert the hand landmarks to a flattened NumPy array
        landmarks_arr = np.array([(lmk.x, lmk.y, lmk.z) for lmk in landmarks]).flatten()

        # Classify the ASL sign using your trained classifier
        asl_class = clf.predict([landmarks_arr])[0]
        prediction.append(asl_class)
        mode_class = st.mode(prediction[-9:])[0][0]
        asl_label = asl_labels[mode_class]

        if asl_label != last_sent:
            if asl_label == 'space':
                sentence.append(' ')
                last_sent = asl_label
            elif asl_label == 'del':
                if sentence:
                    sentence.pop()
            else:
                sentence.append(asl_label)
                last_sent = asl_label

        # Concatenate the words in the sentence
        sentence_str = ''.join(sentence)
        
        # Correct the sentence
        words = word_tokenize(sentence_str)
        corrected_words = []
        for word in words:
            syns = wordnet.synsets(word)
            if syns:
                corrected_words.append(syns[0].lemmas()[0].name().replace('_', ' '))
            else:
                corrected_words.append(word)
        corrected_sentence = ' '.join(corrected_words)

        # Draw the ASL label and corrected sentence on the frame
    try:
        cv2.putText(frame, asl_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
        cv2.putText(frame, sentence_str, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    except:
        pass

    # Show the frame
    cv2.imshow('ASL Classification', frame)

    # Check for key press to exit
    k = cv2.waitKey(20) & 0xff 
    if k ==27:
        break

# Release the video capture device and close
cap.release()
cv2.destroyAllWindows()


print(sentence_str)


"""----------------------------------------------------------------------------------------------------------------------------"""

url = "https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/7c727344-6899-453a-ad30-44decb1cb6c6"
apikey ="ROlCMaeBf3rjFSCVM8J3mEoAHOL0tIXsBcGysx5kTwPB"

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

authenticator = IAMAuthenticator(apikey)
tts = TextToSpeechV1(authenticator=authenticator)
tts.set_service_url(url)

with open('output.mp3', 'wb') as audio_file:
    res = tts.synthesize(sentence_str, accept='audio/mp3', voice='en-US_AllisonV3Voice').get_result()
    audio_file.write(res.content)

from playsound import playsound
playsound("output.mp3")




