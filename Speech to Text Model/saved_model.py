# load libraries
from keras.models import load_model
import random


# load the model
model=load_model('G:/rauf/STEPBYSTEP/Projects/SPEECH/Speech to Text/Speech to Text Model/best_model.hdf5')


# create function to predict given audio
def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]


# predict validation data
index=random.randint(0,len(x_val)-1)
samples=x_val[index].ravel()
print("Audio:",classes[np.argmax(y_val[index])])
ipd.Audio(samples, rate=8000)
print("Text:",predict(samples))


# 