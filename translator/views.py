import os
from django.core.files.storage import default_storage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
from googletrans import Translator
import whisper
import pyttsx3
import torch
from django.conf import settings
from rest_framework.decorators import api_view
from faster_whisper import WhisperModel

import io
from pydub import AudioSegment
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

@api_view(['GET'])
def home(request):
    return Response({
        "status": "True",
        "title": "Laguna Line Balancing"
       }
    )


# Django View
class AudioTranslateView1(APIView):
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request, *args, **kwargs):
        audio_file = request.FILES.get('audio')
        target_language = request.data.get('target_language')
        
        if not audio_file or not target_language:
            return Response({"error": "Audio file and target language are required."}, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the uploaded file temporarily
        temp_audio_path = default_storage.save(f"temp/{audio_file.name}", audio_file)
        temp_audio_path = os.path.join(settings.MEDIA_ROOT, temp_audio_path)
        
        # Load Whisper model
        model = whisper.load_model("small", device=DEVICE)  # Use 'tiny' or 'small' for speed
        # model = WhisperModel("base", device=DEVICE, compute_type="int8")  # Reduces memory usage
        result = model.transcribe(temp_audio_path)
        detected_text = result['text']
        detected_lang = result['language']
        
        # Translate text
        translator = Translator()
        translated_text = translator.translate(detected_text, dest=target_language).text
        
        # Convert translated text to speech
        tts = pyttsx3.init()
        tts.save_to_file(translated_text, "translated_audio.mp3")
        tts.runAndWait()
        
        # Cleanup temporary file
        os.remove(temp_audio_path)
        
        return Response({
            "detected_language": detected_lang,
            "original_text": detected_text,
            "translated_text": translated_text,
            "translated_audio_url": request.build_absolute_uri("/media/translated_audio.mp3")
        }, status=status.HTTP_200_OK)



class AudioTranslateView2(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        audio_file = request.FILES.get('audio')
        target_language = request.data.get('target_language')

        if not audio_file or not target_language:
            return Response({"error": "Audio file and target language are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Read audio file into memory
        audio_bytes = io.BytesIO(audio_file.read())
        audio = AudioSegment.from_file(audio_bytes)
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Convert audio to NumPy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # Normalize audio

        # Load Whisper model
        model = whisper.load_model("tiny")  # Use 'tiny' or 'small' for speed

        # Convert NumPy array to a file-like object
        temp_audio_path = "temp_audio.wav"
        audio.export(temp_audio_path, format="wav")  # Save as temporary WAV file

        # Transcribe using Whisper
        result = model.transcribe(audio)  # Pass file path, not tensor
        detected_text = result['text']
        detected_lang = result['language']

        # Translate text
        translator = Translator()
        translated_text = translator.translate(detected_text, dest=target_language).text

        # Cleanup temporary file
        os.remove(temp_audio_path)

        return Response({
            "detected_language": detected_lang,
            "original_text": detected_text,
            "translated_text": translated_text
        }, status=status.HTTP_200_OK)






# Load Whisper model
model = whisper.load_model("base")  # Use 'tiny' or 'small' for speed
class AudioTranslateView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        audio_file = request.FILES.get('audio')
        target_language = request.data.get('target_language')

        if not audio_file or not target_language:
            return Response({"error": "Audio file and target language are required."}, status=status.HTTP_400_BAD_REQUEST)

        # Read audio file into memory
        audio_bytes = io.BytesIO(audio_file.read())
        audio = AudioSegment.from_file(audio_bytes)
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Convert audio to NumPy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0  # Normalize audio
        samples = torch.tensor(samples)  # Convert to Torch tensor


        # Transcribe using Whisper (pass NumPy array instead of file path)
        result = model.transcribe(samples.numpy())  
        detected_text = result['text']
        detected_lang = result['language']

        # Translate text
        translator = Translator()
        translated_text = translator.translate(detected_text, dest=target_language).text

        return Response({
            "detected_language": detected_lang,
            "original_text": detected_text,
            "translated_text": translated_text
        }, status=status.HTTP_200_OK)
