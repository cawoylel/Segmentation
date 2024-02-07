# Voice Activity Detection and Audio Segmentation
This repository provides a two-step process designed for Voice Activity Detection (VAD) and audio segmentation to facilitate the preprocessing of audio data for Automatic Speech Recognition (ASR) model training or speech model pretraining. 
It contains code to segment audio files into shorter segments, remove non-speech elements like music and noise, and ensure clean voice data for efficient ASR model training.

# Getting Started

## Prerequisites
You will also need to install the required dependencies

Install ffmpeg:
```bash
sudo apt-get install ffmpeg
```
Then, install the dependencies on a python environment (`conda` or `venv`):

```bash
pip install -r requirements.txt
```

# Voice Activity Detection
The first step in processing your audio files is to run the Voice Activity Detection. This step utilizes **[inaSpeechSegmenter](https://github.com/ina-foss/inaSpeechSegmenter/tree/master)**, a CNN-based audio segmentation toolkit that detects speech, music, noise, and speaker gender.

To perform VAD on your audio files, execute the following command:

```bash
python vad.py --data="Folder containing audio files" \
              --output_folder="Where the timestamps files will be stored" \
              --batch_size="The batch size to use. Set lower if no GPUs are available"

```

This process will generate a `.jsonl` file containing timestamps for each tag (speech, music, noise), the start and end position for the tag, along with an ID we generated and added for easy file identification

```bash
["music", 0.0, 11.200000000000001, "d012f188-b9c6-11ee-9717-42010a800005"]
["speech", 11.200000000000001, 126.5, "d012f3ea-b9c6-11ee-9717-42010a800005"]
["music", 126.5, 129.0, "d012f48a-b9c6-11ee-9717-42010a800005"]
["speech", 129.0, 221.16, "d012f4ee-b9c6-11ee-9717-42010a800005"]
["noEnergy", 221.16, 221.56, "d012f53e-b9c6-11ee-9717-42010a800005"]
["speech", 221.56, 331.0, "d012f584-b9c6-11ee-9717-42010a800005"]
["noEnergy", 331.0, 331.44, "d012f5d4-b9c6-11ee-9717-42010a800005"]
["speech", 331.44, 896.4, "d012f610-b9c6-11ee-9717-42010a800005"]
["music", 896.4, 900.04, "d012f656-b9c6-11ee-9717-42010a800005"]

```

# Further Segmentation with Silero VAD
For audio segments longer than 30 seconds, we utilize **[Silero VAD](https://github.com/snakers4/silero-vad)** for further segmentation into shorter clips. Silero VAD is a pre-trained, enterprise-grade Voice Activity Detector.

Execute the following command for further segmentation:

```bash
python segment.py --data="Path to the folder containing the audios" \
                  --segments="Path to the folder containing the jsonl timestamps, generated in previous steps" \
                  --output_folder="Where the segmented audio files will be stored" \
                  --max_files_per_shard="The maximum number of files in each shard" \
                  --min_speech_duration_ms="Minimum segment speech duration in milliseconds" \
                  --max_speech_duration_s="Maximum segment speech duration in seconds"


```

The output is organized into shards for efficient storage and handling, with a maximum number of files specified by `max_files_per_shard` parameter. The structure is as follows: `Corpus/AudioID/ShardID/.wav files.`

# Additional Notes:

- Refer to the provided links for more information on inaSpeechSegmenter and Silero VAD.
- Consider adjusting script parameters based on your specific requirements and data characteristics.
