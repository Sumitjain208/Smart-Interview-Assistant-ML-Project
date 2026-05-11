import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
def analyze_voice():
      fs=44100
      seconds=5
      print("Recording Voice...")
      audio=sd.rec(
            int(seconds*fs),
            samplerate=fs,
            channels=1
      )
      sd.wait()
      write("voice.wav",fs,audio)

      #Load audio
      y,sr=librosa.load("voice.wav")

      #Energy
      energy=np.mean(librosa.feature.rms(y=y))

      #Tempo
      tempo,_=librosa.beat.beat_track(y=y,sr=sr)

      #Confidence score
      score=0

      if energy>0.02:
            score+=1
      if tempo>100:
            score+=1
      return score
