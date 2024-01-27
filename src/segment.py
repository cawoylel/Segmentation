import soundfile as sf
from argparse import ArgumentParser
from pathlib import Path
from itertools import tee
from tqdm import tqdm
import os
import torch

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
 
    parser.add_argument("-m", "--min_speech_duration_ms",
                        help="Minimum segment speech duration in miniseconds",
                        required=True,
                        default=3000,
                        type=int)
  
    parser.add_argument("-n", "--max_speech_duration_s",
                        help="Maximum segment speech duration in seconds",
                        required=True,
                        default=30,
                        type=int) 
      
    parser.add_argument("-r", "--sampling_rate",
                        help="Sampling rate",
                        required=False,
                        default=16000,
                        type=int)     

    parser.add_argument("-f", "--max_files_per_shard",
                        help="The maximum number of files in each shard",
                        required=True,
                        default=10,
                        type=int)   
     
    return parser.parse_args()


def create_shard_folder(corpus, id_jsonl, shard_id):
  shard_folder = Path("SEGMENTED", corpus, id_jsonl, str(shard_id))
  shard_folder.mkdir(exist_ok=True, parents=True)
  return shard_folder


def generate_audio_chunks():
  """
  Generate audio segments from jsonl timestamps files in .wav format
  """
  args = parse_args()
  data = Path(args.data)

  max_files_per_shard = args.max_files_per_shard

  for_total, input_files = tee(Path(args.segments).glob("*.jsonl"), 2)
  total = sum(1 for _ in for_total)

  shard_id = 1

  for jsonl in tqdm(input_files, total=total):
    basename = jsonl.stem
    audio, sr = sf.read(data / f"{basename}.wav")

    current_shard_folder = create_shard_folder(args.output_folder, basename, shard_id)
    with open(jsonl, "r") as segmentation:
        for line in segmentation:
            label, onset, offset, id = eval(line)
            threshold = offset - onset
            if "speech" in label:
              
              onset = int(onset * sr)
              offset = int(offset * sr)
              utterance = audio[onset:offset]

              if len(os.listdir(current_shard_folder)) >= max_files_per_shard:
                  shard_id += 1
                  current_shard_folder = create_shard_folder(args.output_folder, basename, shard_id)
              
              

              if threshold <= 30:
                output_file = current_shard_folder / f"{id}.wav"
                sf.write(output_file, utterance, sr)

              else:
                torch.set_num_threads(1)
                model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                force_reload=True,
                                                onnx=True)

                (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils


                speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=args.sampling_rate,
                                                            min_speech_duration_ms=args.min_speech_duration_ms,
                                                            max_speech_duration_s=args.max_speech_duration_s)
                for i, segment in enumerate(speech_timestamps):
                  output_file = current_shard_folder / f"{id}__{i}.wav"
                  save_audio(output_file,
                  collect_chunks([segment], audio), sampling_rate=args.sampling_rate) 

if __name__ == "__main__":
    generate_audio_chunks()