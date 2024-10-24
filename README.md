
# 🎤 Text-to-Speech (TTS) Model Fine-Tuning

This repository contains code for fine-tuning Text-to-Speech (TTS) models for various languages, including **English, Hindi, Telugu, and Malayalam**. The project utilizes the [TTS](https://github.com/coqui-ai/TTS) library to generate high-quality audio files from text input. 🌍🔊

## 📚 Table of Contents

- [🎤 Text-to-Speech (TTS) Model Fine-Tuning](#-text-to-speech-tts-model-fine-tuning)
  - [📚 Table of Contents](#-table-of-contents)
  - [✨ Features](#-features)
  - [🌐 Supported Languages and Models](#-supported-languages-and-models)
  - [⚙️ Requirements](#️-requirements)
  - [📊 Dataset](#-dataset)
  - [🔧 Installation](#-installation)
  - [🚀 Usage](#-usage)
  - [🔄 Fine-Tuning the Model](#-fine-tuning-the-model)
  - [🎧 Generating Audio](#-generating-audio)
  - [🤝 Contributing](#-contributing)
  - [📝 License](#-license)
  - [🛠️ Customization Instructions:](#️-customization-instructions)
    - [Customization Instructions:](#customization-instructions)

## ✨ Features

- Fine-tunes TTS models for multiple languages.
- Generates audio files from text with natural prosody and intonation.
- Custom Dataset class for loading text data.
- Supports batch processing for efficiency.

## 🌐 Supported Languages and Models

The project supports the following languages and their respective models:

- **English**: `tts_models/en/ljspeech/tacotron2-DDC_ph`
- **Hindi**: `tts_models/hi/hindi/ljspeech-DDC`
- **Telugu**: `tts_models/te/telugu/ljspeech-DDC`
- **Malayalam**: `tts_models/ml/malayalam/ljspeech-DDC`

Each model is specifically designed to capture the phonetic nuances of its language, ensuring high-quality audio output. 🎵

## ⚙️ Requirements

- Python 3.x
- PyTorch
- pandas
- tqdm
- TTS library
- Other dependencies listed in `requirements.txt`

## 📊 Dataset

The dataset should contain a `metadata.csv` file for each language, structured with at least one column named `text`, containing the text entries to be synthesized. Below is an example of how the `metadata.csv` file should look:

```
text
"This is a sample sentence."
"Another sentence for TTS."
```

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

1. **Initialize the Model**: The code initializes the TTS model based on the specified language.
2. **Fine-Tune the Model**: The model is fine-tuned using the specified dataset.
3. **Generate Audio**: After fine-tuning, audio files can be generated from the text in the dataset.

Run the main script:
```bash
python main.py
```

## 🔄 Fine-Tuning the Model

The `fine_tune_model` function performs the fine-tuning process, iterating over the dataset for a specified number of epochs, calculating the loss, and updating the model weights.

## 🎧 Generating Audio

The `generate_audio` function generates audio files from the text data in the metadata, saving the output in the specified directory. The generated audio files are high-quality WAV files that can be played back to evaluate the TTS synthesis.

## 🤝 Contributing

Contributions are welcome! Please create a pull request or open an issue to discuss any changes.

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🛠️ Customization Instructions:
1. **Replace** `yourusername/your-repo.git` with your actual GitHub username and repository name.
2. **Ensure that the language and model sections accurately reflect the models you're using.**
3. **Modify any other details as needed** to suit your project's specifics. 

Feel free to adjust any sections further for clarity or detail as you see fit! ✨

### Customization Instructions:
1. Replace `yourusername/your-repo.git` with your actual GitHub username and repository name.
2. Ensure that the language and model sections accurately reflect the models you're using.
3. Modify any other details as needed to suit your project's specifics. 

