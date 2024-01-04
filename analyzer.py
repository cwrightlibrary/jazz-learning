import pickle
import numpy as np
import keras.utils
from math import floor
from pathlib import Path
from os.path import dirname, join, realpath
from music21 import converter, instrument, note, chord

# following this guide:
# https://medium.com/mlearning-ai/how-to-generate-music-using-machine-learning-72360ba4a085

# songs = []
# folder = Path(join(dirname(realpath(__file__)), "mids"))

# for file in folder.rglob("*.mid"):
# 	songs.append(file)

# notes = []
# for i,file in enumerate(songs):
# 	print(f"{i + 1}: {file}")
# 	try:
# 		midi = converter.parse(file)
# 		notesToParse = None
# 		parts = instrument.partitionByInstrument(midi)
# 		if parts:
# 			notesToParse = parts.parts[0].recurse()
# 		else:
# 			notesToParse = midi.flat.notes
# 		for element in notesToParse:
# 			if isinstance(element, note.Note):
# 				notes.append(str(element.pitch))
# 			elif isinstance(element, chord.Chord):
# 				notes.append(".".join(str(n) for n in element.normalOrder))
# 	except:
# 		print(f"FAILED: {i + 1}: {file}")

# with open("notes", "wb") as filepath:
# 	pickle.dump(notes, filepath)

def prepareSequences(notes, nVocab):
	sequenceLength = 32
	pitchNames = sorted(set(item for item in notes))
	numPitches = len(pitchNames)
	noteToInt = dict((note, number) for number, note in enumerate(pitchNames))

	networkInput = []
	networkOutput = []

	for i in range(0, len(notes) - sequenceLength, 1):
		sequenceIn = notes[i:i + sequenceLength]
		sequenceOut = notes[i + sequenceLength]
		networkInput.append([noteToInt[char] for char in sequenceIn])
		networkOutput.append(noteToInt[sequenceOut])
	
	nPatterns = len(networkInput)
	print(len(networkInput))
	networkInput = np.reshape(networkInput, (nPatterns, sequenceLength, 1))
	networkInput = networkInput / float(nVocab)
	networkOutput = keras.utils.to_categorical(networkOutput)

	return (networkInput, networkOutput)

with open(join(dirname(realpath(__file__)), "notes"), "rb") as file:
	notes = pickle.load(file) 

nVocab = len(set(notes))
networkInput, networkOutput = prepareSequences(notes, nVocab)
nPatterns = len(networkInput)
pitchNames = sorted(set(item for item in notes))
numPitches = len(pitchNames)
