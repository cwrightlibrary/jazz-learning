import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from collections import Counter
from keras.utils import set_random_seed
from keras.layers import *
from keras.models import *
from music21 import *
from numpy.random import seed
from os.path import dirname, join, realpath
from pathlib import Path
from sklearn.model_selection import train_test_split

seed(1)
set_random_seed(2)

# https://medium.com/analytics-vidhya/want-to-generate-your-own-music-using-deep-learning-heres-a-guide-to-do-just-that-dd35d6ddcd01

songs = []
folder = Path(join(dirname(realpath(__file__)), "mids-small-set"))

for file in folder.rglob("*.mid"):
    songs.append(file)

def cycle_midis():
	notes = []
	for i,file in enumerate(songs):
		print(f"{int(((i + 1) / (len(songs))) * 100)}%: {file}")
		try:
			midi = converter.parse(file)
			notesToParse = None
			parts = instrument.partitionByInstrument(midi)
			if parts and not "percussion" in str(parts):
				notesToParse = parts.parts[0].recurse()
			elif not parts:
				notesToParse = midi.flat.notes
			for element in notesToParse:
				if isinstance(element, note.Note):
					notes.append(str(element.pitch))
				elif isinstance(element, chord.Chord):
					notes.append(".".join(str(n) for n in element.normalOrder))
		except:
			print(f"FAILED: {i + 1}/{len(songs): {file}}")
	return np.array(notes)

def export_notes():
	music21Notes = cycle_midis()
	with open("music21-notes", "wb") as filepath:
		pickle.dump(music21Notes, filepath)

# export_notes()
with open(join(dirname(realpath(__file__)), "music21-notes"), "rb") as file:
    notesArray = pickle.load(file)

nVocab = len(set(notesArray))

notes_ = [element for note_ in notesArray for element in note_]

freq = dict(Counter(notes_))
frequentNotes = [note_ for note_, count in freq.items() if count >= 50]

newMusic = []
for notes in notesArray:
    temp = []
    for note_ in notes:
        if note_ in frequentNotes:
            temp.append(note_)
    newMusic.append(temp)

noOfTimesteps = 32
x = []
y = []

for note_ in newMusic:
    for i in range(0, len(note_) - noOfTimesteps, 1):
        input_ = note_[i:i + noOfTimesteps]
        output = note_[i + noOfTimesteps]
        x.append(input_)
        y.append(output)

x = np.array(x)
y = np.array(y)

uniqueX = list(set(x.ravel()))
xNoteToInt = dict((note_, number) for number, note_ in enumerate(uniqueX))

uniqueY = list(set(y.ravel()))
yNoteToInt = dict((note_, number) for number, note_ in enumerate(uniqueY))

xSeq = []
for i in x:
    temp = []
    for j in i:
        temp.append(xNoteToInt[j])
    xSeq.append(temp)

ySeq = []
for i in y:
    temp = []
    for j in i:
        temp.append(yNoteToInt[j])
    ySeq.append(temp)

xSeq = np.array(xSeq)
ySeq = np.array(ySeq)

xTr, xVal, yTr, yVal = train_test_split(xSeq, ySeq, test_size=0.2, random_state=0)

def ltsm():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(nVocab))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    return model

