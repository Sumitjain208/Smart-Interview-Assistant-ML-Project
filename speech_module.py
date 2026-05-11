import speech_recognition as sr
import pyaudio
def speech_to_txt():
      recognizer=sr.Recognizer()
      with sr.Microphone() as source:
            print("speak now...")
            audio=recognizer.listen(source)
      try:
            text=recognizer.recognize_google(audio)
            return text
      except:
            return "Could not recognize speech"