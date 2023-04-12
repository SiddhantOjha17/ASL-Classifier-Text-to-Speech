# ASL Classifier and Text-to-Speech Converter

This repository contains code for an American Sign Language (ASL) classifier that uses a random forest machine learning model to predict ASL letters based on hand landmarks detected in real-time video. The classifier also includes a text-to-speech converter that uses IBM Watson's Text-to-Speech API to convert the detected ASL letters into an audio output of the corresponding English words.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required files.

```bash
pip install -r requirements.txt
```

## Demo

https://user-images.githubusercontent.com/96463139/231254650-63bc3d15-8bc8-417e-8481-a7912b6ad179.mp4

## Note
* The ASL classifier is currently set up to detect a maximum of one hand at a time.
* The classifier uses a pre-trained random forest model that is included in the repository in a zip file, please unzip before you run the code (rf_model.joblib).
* The ASL labels used by the classifier are listed in asl_labels in ASL.py.
* The text-to-speech converter uses IBM Watson's Text-to-Speech API, and requires an API key and service URL (see apikey and url variables). You will need to create an IBM Cloud account and generate an API key to use this feature.

