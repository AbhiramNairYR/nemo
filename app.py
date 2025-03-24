import torch
import torchaudio
import torchaudio.transforms as transforms
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils import logging

# Input and output file paths
input_audio_path = "sample.wav"   # Replace with your actual file
mono_audio_path = "mono_audio.wav"  # New mono-converted audio file

# Convert stereo to mono
waveform, sample_rate = torchaudio.load(input_audio_path)
if waveform.shape[0] > 1:  # Check if audio has more than 1 channel
    waveform = torch.mean(waveform, dim=0, keepdim=True)  # Convert to mono

# Resample to 16kHz if needed (NeMo models often require this)
target_sample_rate = 16000
if sample_rate != target_sample_rate:
    resampler = transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)
    sample_rate = target_sample_rate

# Save the processed mono audio
torchaudio.save(mono_audio_path, waveform, sample_rate)

# Load the local diarization model
model_path = "diar_sortformer_4spk-v1.nemo"
logging.info(f"Loading model from {model_path}")
diar_model = EncDecSpeakerLabelModel.restore_from(model_path)
logging.info("Model loaded successfully!")

# Run diarization on mono audio file
predicted_segments = diar_model.diarize(audio=mono_audio_path, batch_size=1)

# Print results
print("Speaker Diarization Results:")
for segment in predicted_segments:
    print(segment)

