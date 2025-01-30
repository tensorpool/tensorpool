from kokoro import KPipeline
import soundfile as sf
import numpy as np

def generate_tts(text, lang_code='a', voice='af_heart', speed=1):
    pipeline = KPipeline(lang_code=lang_code)

    audio_segments = []
    for _, _, audio in pipeline(
        text,
        voice=voice,
        speed=speed,
        split_pattern=r'\n+'
    ):
        audio_segments.append(audio)

    silence = np.zeros(12000)
    full_audio = np.concatenate([
        np.concatenate([segment, silence])
        for segment in audio_segments
    ])

    sf.write('output.wav', full_audio, 24000)
    return full_audio

text = '''
TensorPool is the easiest way to execute ML jobs on the cloud.
Our CLI makes ML model training effortless - just describe your job, and we handle GPU orchestration and execution at half the cost of major cloud providers.
'''

generate_tts(text)
