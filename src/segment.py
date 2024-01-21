import json
import soundfile as sf

audio, sr = sf.read("sample/RESAMPLED_RFIFulfulde_091b72ba_1333_4771_9361_0f2665d6932f.wav")
with open("classified_audios/RESAMPLED_RFIFulfulde_091b72ba_1333_4771_9361_0f2665d6932f.jsonl", "r") as segmentation:
    for idx, line in enumerate(segmentation):
        label, onset, offset = eval(line)
        output_file = f"{label}_{idx}.wav"
        onset = int(onset * sr)
        offset = int(offset * sr)
        utterance = audio[onset:offset]
        sf.write(output_file, utterance, sr)