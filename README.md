# Voice Recognition RNN

This project implements a voice detection system for recognizing digits from 0 to 9 using a Recurrent Neural Network (RNN) and spectrograms.

## Overview

The voice recognition system processes audio input, generates spectrograms, and utilizes an RNN model to classify the spoken digits.

## Project Structure

- `src/models`: Contains the RNN model architecture and utility functions.
- `src/data`: Handles data preprocessing and spectrogram generation.
- `src/train`: Responsible for training and evaluating the model.
- `tests`: Contains unit tests for the model and preprocessing functions.
- `requirements.txt`: Lists the project dependencies.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd voice-recognition-rnn
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model, run the following command:
```
python src/train/train.py
```

To evaluate the model, use:
```
python src/train/evaluate.py
```

## License

This project is licensed under the MIT License.