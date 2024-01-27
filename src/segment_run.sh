python /content/Segmentation/src/segment.py \
    --data="RESAMPLED/WhatsappAudio" \
	--segments="SPEECH_TIMESTAMPS/WhatsappAudio" \
    --output_folder="WhatsappAudio" \
    --max_files_per_shard="6" \
	--min_speech_duration_ms="3000" \
    --output_folder="WhatsappAudio" \
    --max_speech_duration_s="30" 