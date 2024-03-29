# Load the libraries
import json
from itertools import tee
from pathlib import Path
from argparse import ArgumentParser
from inaSpeechSegmenter import Segmenter
from tqdm import tqdm
import uuid

def parse_args():
    parser = ArgumentParser(description="Perform VAD to get only speech segment (without the music, noise, silence)")
    parser.add_argument("-d", "--data",
                        help="Path to the folder containing the audios.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_folder",
                        help="Where the timestamps files will be stored.",
                        required=True,
                        type=str)
    parser.add_argument("-b", "--batch_size",
                        help="The batch size to use. Have to be set low if no GPUs are available on the machine.",
                        required=False,
                        default=32,
                        type=int)
    return parser.parse_args()

def save_timestamps(output_path: Path, timestamps):
    with open(output_path, "w") as output_file:
        for item in timestamps:
          output_file.write(json.dumps(item + (str(uuid.uuid1()),)) + "\n")

def main():
    args = parse_args()
    segmenter = Segmenter(vad_engine="smn",
                          detect_gender=False,
                          batch_size=args.batch_size)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    for_total, input_files = tee(Path(args.data).glob("*.wav"), 2)
    total = sum(1 for _ in for_total)
    for audio in tqdm(input_files, total=total):
        filename = audio.stem
        timestamps = segmenter(audio)
        save_timestamps(output_folder / f"{filename}.jsonl", timestamps)

if __name__ == "__main__":
    main()