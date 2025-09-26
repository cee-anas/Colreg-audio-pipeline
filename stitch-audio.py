from pydub import AudioSegment

# Load your audio files
audio1 = AudioSegment.from_wav("test-audio/1.wav")
audio2 = AudioSegment.from_wav("test-audio/2.wav")
audio3 = AudioSegment.from_wav("test-audio/3.wav")
other = AudioSegment.from_wav("test-audio/other.wav")

# Make sure all audio is stereo
def ensure_stereo(audio):
    if audio.channels == 1:
        return audio.set_channels(2)
    return audio

audio1 = ensure_stereo(audio1)
audio2 = ensure_stereo(audio2)
audio3 = ensure_stereo(audio3)
other = ensure_stereo(other)

# Stitch audios in order: 1 -> other -> 2 -> other -> 3 -> other
stitched = audio1 + other + audio2 + other + audio3 + other

# Export the final audio
stitched.export("test-audio/test.wav", format="wav")
print("Stitched audio saved as stitched_output.wav")

