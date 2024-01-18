# Emotion Configuration Files

JSON files, where the keys are the exact string you want appearing in the annotation output file of `annotate.py`, and the values are the exact input string to `Demux` (should ideally be lower case, a single word per emotion and a list of emotions for a cluster). Note that our `paletz` model has been trained with `paletz.json`, not `paletz_revised.json`.