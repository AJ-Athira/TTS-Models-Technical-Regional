import os
import pandas as pd
import torch
from tqdm import tqdm
from TTS.api import TTS
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Define paths
dataset_path = 'dataset/'
languages = ['english', 'hindi', 'telugu', 'malayalam']
output_dir = 'outputs/'

# Initialize TTS model
def initialize_model(language):
    # Create a TTS object instance
    tts = TTS()

    # List available models
    available_models = tts.list_models()
    print("Available models:", available_models)  # Print available models

    # Set the appropriate model for each language
    if language == 'english':
        model_name = "tts_models/en/ljspeech/tacotron2-DDC_ph"
    elif language == 'hindi':
        model_name = "tts_models/hi/hindi/ljspeech-DDC"  # Check the model name here
    elif language == 'telugu':
        model_name = "tts_models/te/telugu/ljspeech-DDC"
    elif language == 'malayalam':
        model_name = "tts_models/ml/malayalam/ljspeech-DDC"

    # Load the model using the specified model name
    model = TTS(model_name)
    return model


# Custom Dataset for TTS
class TTSDataset(Dataset):
    def __init__(self, metadata_file):
        self.metadata = pd.read_csv(metadata_file)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        text = self.metadata.iloc[idx]['text']  # Ensure 'text' exists in your CSV
        target_mel = self.generate_target_mel(text)  # Replace with your logic to obtain target mel
        return text, target_mel

    def generate_target_mel(self, text):
        # Placeholder for generating mel spectrograms; implement your logic here
        return torch.randn(1, 80, 100)  # Example: Random tensor; replace with actual mel spectrogram generation logic

def calculate_loss(mel_outputs, target_mel):
    loss_function = nn.MSELoss()
    return loss_function(mel_outputs, target_mel)

def fine_tune_model(model, language, dataloader, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0
        
        for texts, target_mel in dataloader:  # Unpack as (batch of texts, batch of target mel spectrograms)
            outputs = []
            for text in texts:
                output = model.tts(text, return_mel=True)  # Directly pass the text
                outputs.append(output)

            # Debugging: Print output shapes and types
            for i, o in enumerate(outputs):
                print(f"Output {i}: Type: {type(o)}, Value: {o}")

            # Handle the output based on its structure
            mel_outputs = []
            for o in outputs:
                if isinstance(o, tuple):
                    mel_outputs.append(torch.from_numpy(o[0]))  # Assuming o[0] is the mel spectrogram
                else:
                    print(f"Unexpected output format for: {o}")

            # Concatenate if there are multiple mel spectrograms
            if mel_outputs:
                mel_outputs = torch.cat(mel_outputs)
            else:
                print("No valid mel outputs to concatenate.")
                continue

            # Ensure target_mel is a tensor as well
            target_mel_tensor = torch.from_numpy(target_mel)

            # Calculate the loss
            loss = calculate_loss(mel_outputs, target_mel_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}')

    # Save the fine-tuned model
    model_save_path = os.path.join('models/', f'{language}_fine_tuned.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def generate_audio(model, language):
    metadata_file = os.path.join(dataset_path, language, 'metadata.csv')
    metadata = pd.read_csv(metadata_file)

    generated_dir = os.path.join(dataset_path, language, 'wavs/generated/')
    os.makedirs(generated_dir, exist_ok=True)

    for index, row in tqdm(metadata.iterrows(), total=len(metadata)):
        text = row['text']
        filename = f'sentence_{index + 1}.wav'
        audio_path = os.path.join(generated_dir, filename)

        model.tts_to_file(text=text, file_path=audio_path)
        print(f"Generated audio for: {text} -> {audio_path}")

# Main execution

if __name__ == "__main__":
    for language in languages:
        print(f"Processing {language}...")
        model = initialize_model(language)

        # Create a dataset and dataloader
        metadata_file = os.path.join(dataset_path, language, 'metadata.csv')
        dataset = TTSDataset(metadata_file)  # Initialize dataset with your metadata
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Create a DataLoader

        # Initialize an optimizer (e.g., Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Fine-tune the model
        fine_tune_model(model, language, dataloader, optimizer)

        # Save the model after fine-tuning
        model_save_path = os.path.join('models/', f'{language}_fine_tuned.pt')
        torch.save(model.state_dict(), model_save_path)

        # Load the model for inference (optional, depending on your workflow)
        model.load_state_dict(torch.load(model_save_path))  # Load the saved state dict
        model.eval()  # Set to evaluation mode if you're going to do inference

        # Generate audio files
        generate_audio(model, language)

    print("TTS generation completed!")


