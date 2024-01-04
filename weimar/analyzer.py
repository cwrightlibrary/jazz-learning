import csv, re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from os.path import dirname, join, realpath

songDbFile = join(dirname(realpath(__file__)), "song_db.csv")

def minToMaj(minInput):
	beforeList = ["A-min", "E-min", "B-min", "F#-min", "C#-min", "G#-min", "D#-min", "A#-min", "D-min", "G-min", "C-min", "F-min", "Bb-min", "Eb-min", "Ab-min"]
	afterList = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]
	for key in range(len(beforeList)):
		if beforeList[key] == minInput:
			return afterList[key]

def chordsToRomanNum(dict):
	notes = ["C", "C# Db", "D", "D# Eb", "E", "F", "F# Gb", "G", "G# Ab", "A", "A# Bb", "B", "C", "C# Db", "D", "D# Eb", "E", "F", "F# Gb", "G", "G# Ab", "A", "A# Bb", "B"]

	replaceNotes = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

	chordNames = ["C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb", "C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"]

	romanNumerals = ["I", "bII", "II", "bIII", "III", "bIV", "IV", "V", "#V", "VI", "bVII", "VII"]

	for n in notes:
		if dict["key"] in n:
			keyIndex = notes.index(n)
	newChordList = replaceNotes[keyIndex:keyIndex + 12]

	newChords = []
	for inputMeasure in dict["chord_changes"]:
		newMeasure = []
		for inputChord in inputMeasure:
			for newNote in range(len(newChordList)):
				if newChordList[newNote] in inputChord:
					newMeasure.append(inputChord.replace(newChordList[newNote], romanNumerals[newNote]))
					break
		newChords.append(newMeasure)
	dict["roman_changes"] = newChords

with open(songDbFile, encoding="utf8", mode="r") as file:
	csvReader = csv.DictReader(file)
	songDicts = []
	for row in csvReader:
		songDicts.append(row)
		row["chord_changes"] = row["chord_changes"].replace("||", "|").replace("j7", "maj7").replace("o", "dim").replace("-", "m").replace(" ", "")

		row["chord_changes"] = row["chord_changes"][1:len(row["chord_changes"]) - 1]
		row["chord_changes"] = row["chord_changes"].split("|")

		tempChordList = []

		for chord in row["chord_changes"]:
			tempChordString = []
			tempChordString = re.findall("[A-Z][^A-Z]*", chord)
			tempChordList.append(tempChordString)
		
		row["chord_changes"] = tempChordList

		row["key"] = row["key"].replace("-maj", "")

		if "-min" in row["key"]:
			row["key"] = minToMaj(row["key"])
		
		if row["key"] != None:
			chordsToRomanNum(row)

measureLib = []

for song in songDicts:
	if "roman_changes" in song:
		for measure in song["roman_changes"]:
			newMeasure = "-".join(measure)
			measureLib.append(newMeasure)

originalDatabase = join(dirname(realpath(__file__)), "weimar-jazz-database.csv")

# straightforward testing
with open(originalDatabase, "r") as file:
	plots = csv.reader(file)
	for row in plots:
		print(row)
