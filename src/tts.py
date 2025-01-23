# import base64
# import os
# import tempfile

# import pyttsx4


# def genenrate_and_save_audio(text: str):
#     engine = pyttsx4.init()
#     voices = engine.getProperty('voices')
#     voice = voices[0]  # Arabic accent # 23

#     engine.setProperty('voice', voice.id)

#     # engine.save_to_file(text, 'output_tts.mp3')
#     temp_dir = tempfile.gettempdir()
#     temp_file = os.path.join(temp_dir, 'output_tts.wav')
#     engine.save_to_file(text, temp_file)
#     engine.runAndWait()


# def encode_wav_to_base64(wav_filepath: str) -> str:
#     with open(wav_filepath, "rb") as wav_file:
#         wav_data = wav_file.read()
#     return base64.b64encode(wav_data).decode('utf-8')


# if __name__ == "__main__":
#     text = "Hello, I am your personal medical assistant. How are you feeling today?"
#     genenrate_and_save_audio(text)

#     # Convert the audio file to base64
#     wav_file = tempfile.gettempdir() + '/output_tts.wav'
#     base64_encoded_data = encode_wav_to_base64(wav_file)
#     print(base64_encoded_data)

import base64
import os
import tempfile
# from api import TTS


# def generate_and_save_audio(text: str):
#     # Load a pre-trained TTS model from Coqui TTS
#     tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")  # You can change this to a different model if needed.
    
#     # Define the temporary file path
#     temp_dir = tempfile.gettempdir()
#     temp_file = os.path.join(temp_dir, 'output_tts.wav')
    
#     # Generate and save the audio
#     tts.tts_to_file(text=text, file_path=temp_file)
#     return temp_file


# def encode_wav_to_base64(wav_filepath: str) -> str:
#     # Read the WAV file and encode it as base64
#     with open(wav_filepath, "rb") as wav_file:
#         wav_data = wav_file.read()
#     return base64.b64encode(wav_data).decode('utf-8')


# if __name__ == "__main__":
#     text = "Hello, I am your personal medical assistant. How are you feeling today?"
    
#     # Generate audio and get the file path
#     wav_file = generate_and_save_audio(text)
    
#     # Convert the audio file to base64
#     base64_encoded_data = encode_wav_to_base64(wav_file)
#     print(base64_encoded_data)
