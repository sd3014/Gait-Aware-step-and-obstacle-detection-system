import pyttsx3
import threading

def speak(text):
    if not text:
        return

    def run():
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=run, daemon=True).start()