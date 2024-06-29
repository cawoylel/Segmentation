import os
from pathlib import Path
from itertools import tee
import json
from argparse import ArgumentParser
import torch
from pathlib import Path
from tqdm import tqdm
torch.set_num_threads(os.cpu_count())

SAMPLING_RATE = 16000
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=True)

(get_speech_timestamps,
save_audio,
read_audio,
VADIterator,
collect_chunks) = utils

def segment(input_folder: Path):
    for_total, sub_folders = tee(input_folder.glob("*.wav"), 2)
    total = sum(1 for _ in for_total)
    for audio_file in tqdm(sub_folders, total=total):
        wav = read_audio(audio_file, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=SAMPLING_RATE, max_speech_duration_s=30)
        yield speech_timestamps, audio_file
    yield None, None

def write_on_file(utterances, output_folder, output_filename):
    with open(output_folder / f"{output_filename}.jsonl", "w") as output_file:
        for utterance in utterances:
            output_file.write(json.dumps(utterance) + "\n")

def save_timestamps(input_folder: Path, output_folder: Path, shard_size: int):
    utterances = []
    shard_idx = 0
    for speech_timestamps, audio_file in segment(input_folder):
        if shard_idx < shard_size and None not in (speech_timestamps, audio_file):
            utterances.append(speech_timestamps)
            shard_idx += 1
            continue
        output_folder = output_folder / audio_file.stem / f"shard_{shard_idx:02d}"
        output_folder.mkdir(exist_ok=True, parents=True)
        file_id = None
        output_path = output_folder / f"{file_id}.jsonl"
        write_on_file(utterances, output_path)
        utterances = []
        shard_idx = 0

def parse_args():
    parser = ArgumentParser(description="Performs segmentation in order to get small chunk of speech.")
    parser.add_argument("-i", "--input_folder",
                        required=True,
                        help="The input folder containing the audio files.",
                        type=str)
    parser.add_argument("-o", "--output_folder",
                        required=True,
                        help="Where the output folder will be saved",
                        type=str)
    parser.add_argument("-s", "--shard_size",
                        required=False,
                        default=200,
                        help="The maximum of files allowed in a folder.",
                        type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    save_timestamps(Path(args.input_folder), Path(args.output_folder), args.shard_size)
    
if __name__ == "__main__":
    main()