# load libraries
import sounddevice as sd
import soundfile as sf


# to record and save custom voice
samplerate = 16000
duration = 1 # seconds
filename = 'G:/rauf/STEPBYSTEP/Projects/SPEECH/Speech to Text/Speech to Text Model/yes.wav'
print("start")
mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
    channels=1, blocking=True)
print("end")
sd.wait()
sf.write(filename, mydata, samplerate)


#saved voice directory
os.listdir('G:/rauf/STEPBYSTEP/Projects/SPEECH/Speech to Text/Speech to Text Model/yes.wav')
filepath='G:/rauf/STEPBYSTEP/Projects/SPEECH/Speech to Text/Speech to Text Model/yes.wav'

#reading the voice commands
samples, sample_rate = librosa.load(filepath + '/' + 'stop.wav', sr = 16000)
samples = librosa.resample(samples, sample_rate, 8000)
ipd.Audio(samples,rate=8000)  

predict(samples)