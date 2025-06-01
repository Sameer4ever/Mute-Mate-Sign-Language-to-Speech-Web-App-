from flask import Flask, render_template, request
from threading import Thread
import model_core
from gtts import gTTS
import tempfile
import pygame
import os
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('2.html')

@app.route('/start-detection', methods=['POST'])
def start_detection():
    detection_thread = Thread(target=model_core.run_detection)
    detection_thread.daemon = True
    detection_thread.start()
    return "Sign detection started!"

@app.route('/get-latest', methods=['GET'])
def get_latest():
    return model_core.get_latest_sentence()

@app.route('/get-latest-hindi', methods=['GET'])
def get_latest_hindi():
    return model_core.get_latest_hindi()

@app.route('/clear-last-word', methods=['POST'])
def clear_last_word():
    model_core.clear_last_word()
    return "Last word cleared."

@app.route('/clear-all', methods=['POST'])
def clear_all():
    model_core.clear_all()
    return "All cleared."

def play_speech_async(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            temp_path = fp.name

        pygame.mixer.init()
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        pygame.mixer.music.unload()
        pygame.mixer.quit()
        os.remove(temp_path)
    except Exception as e:
        print(f"Speech playback error: {e}")

@app.route('/speak-sentence', methods=['POST'])
def speak_sentence():
    sentence = model_core.get_latest_sentence()
    if not sentence:
        return "No sentence available to speak."

    # Run speech playback asynchronously so Flask responds immediately
    Thread(target=play_speech_async, args=(sentence,), daemon=True).start()
    return "Speech playback started."

if __name__ == '__main__':
    app.run(debug=True)