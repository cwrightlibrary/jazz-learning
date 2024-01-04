import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter
from music21 import converter, instrument, note, chord
from os import listdir
from os.path import dirname, join, realpath

def readMidi(file):
	print("Loading music file:", file)

	notes = []
	notesToParse = None
	midi = converter.parse(file)
	s2 = instrument.partitionByInstrument(midi)
	for part in s2.parts:
		if "Piano" in str(part):
			notesToParse = part.recurse()
			for element in notesToParse:
				if isinstance(element, note.Note):
					notes.append(str(element.pitch))
				elif isinstance(element, chord.Chord):
					notes.append(".".join(str(n) for n in element.normalOrder))
	return np.array(notes)

path = join(dirname(realpath(__file__)), "mids")
print(path)

files = [i for i in listdir(path) if i.endswith(".mid")]

notesArray = np.array([readMidi(path + "\\" + i) for i in files])

with open(join(dirname(realpath(__file__)), "notes-music21"), "wb") as filepath:
		pickle.dump(notesArray, filepath)

notesArrayFile = join(dirname(realpath(__file__)), "notes-music21")

notes_ = [element for note_ in notesArrayFile for element in note_]

uniqueNotes = list(set(notes_))
print(len(uniqueNotes))

freq = dict(Counter(notes_))
no = [count for _, count in freq.items()]

plt.figure(figsize=(5, 5))
plt.hist(no)
