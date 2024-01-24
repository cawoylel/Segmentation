import soundfile as sf
from argparse import ArgumentParser
from pathlib import Path
from itertools import tee
from tqdm import tqdm

def parse_args():
    parser = ArgumentParser(description="Generate segmentation audio chunks files in .wav format")
    parser.add_argument("-d", "--data",
                        help="Path to the folder containing the audios.",
                        required=True,
                        type=str)
    
    parser.add_argument("-s", "--segments",
                        help="Path to the folder containing the jsonl timestamps.",
                        required=True,
                        type=str)
    
    parser.add_argument("-o", "--output_folder",
                        help="Where the segmented audio files will be stored.",
                        required=True,
                        type=str)
 
    return parser.parse_args()


def generate_audio_chunks():
  """
  Generate audio segments from jsonl timestamps files in .wav format
  """
  args = parse_args()
  output_folder = Path(args.output_folder)
  data = Path(args.data)

  output_folder.mkdir(exist_ok=True, parents=True)

  for_total, input_files = tee(Path(args.segments).glob("*.jsonl"), 2)
  total = sum(1 for _ in for_total)

  for jsonl in tqdm(input_files, total=total):
    basename = jsonl.stem
    audio, sr = sf.read(data / f"{basename}.wav")

    with open(jsonl, "r") as segmentation:
        for line in segmentation:
            label, onset, offset, id = eval(line)
            if "speech" in label:
              output_file = output_folder / f"{basename}__{id}.wav"
              onset = int(onset * sr)
              offset = int(offset * sr)
              utterance = audio[onset:offset]
              sf.write(output_file, utterance, sr)

if __name__ == "__main__":
    generate_audio_chunks()