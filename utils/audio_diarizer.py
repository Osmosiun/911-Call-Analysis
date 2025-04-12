import os
import pandas as pd
import pympi
import logging
import assemblyai as aai
import shutil
aai.settings.api_key = "83df3680877f4567a3ada5b60fa61d1b"

def elan_to_dataframe(folder_path):
    
    file_names = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file) for file in file_names if file.endswith(".eaf")]
    
    data = []

    for eaf_file_path in file_paths:

        call_name = eaf_file_path.split("/")[-1].split(".")[0]

        eaf_obj = pympi.Elan.Eaf(eaf_file_path)

        for tier_name in eaf_obj.get_tier_names():
            annotations = eaf_obj.get_annotation_data_for_tier(tier_name)
            for (start, end, _) in annotations:
                data.append({
                    "channel": tier_name,
                    "startutt": start,
                    "stoputt": end,
                    "CallName": call_name
                })

    df = pd.DataFrame(data)
    return df

def transcribe_audio_to_dataframe(folder_path, log_dir):
    
    # Set up logging
    log_file_path = os.path.join(log_dir, 'transcribe_audio_to_dataframe.txt')
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('Starting transcription process.')

    # # Delete the processed_audio_dir if it exists, then create it
    # if os.path.exists(processed_audio_dir):
    #     logging.info(f'Deleting existing directory: {processed_audio_dir}')
    #     shutil.rmtree(processed_audio_dir)
    
    # os.makedirs(processed_audio_dir)
    # logging.info(f'Created new directory: {processed_audio_dir}')

    file_names = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file) for file in file_names]

    file_num = 1

    df_word_level = pd.DataFrame(columns=['CallName', 'filenum', 'channel', 'startutt', 'stoputt', 'duration', 'content'])

    for idx, file_path in enumerate(file_paths):

        if not file_path.endswith(".wav"): continue

        audio_url = file_path

        config = aai.TranscriptionConfig(speaker_labels=True)

        logging.info(f'Transcribing file: {file_names[idx]}')
        transcript = aai.Transcriber().transcribe(audio_url, config)

        # Get the CallName by finding the last dot and omitting everything after it
        call_name = file_names[idx][:file_names[idx].rfind(".")]

        # print(tra)

        for utterance in transcript.utterances:

            for utterance_word in utterance.words:
                new_row = {
                    'CallName': call_name,
                    'startutt': utterance_word.start,
                    'stoputt': utterance_word.end,
                    'content': utterance_word.text
                }

                df_word_level.loc[len(df_word_level)] = new_row

        file_num += 1

    logging.info('Transcription process completed and dataframes saved.')

    return df_word_level

def assign_words_to_speakers_by_call(df_speaker, df_word):
    # Ensure 'CallName' is present
    assert 'CallName' in df_speaker.columns and 'CallName' in df_word.columns, \
        "'CallName' column must exist in both DataFrames"

    merged_rows = []

    # Process each call separately
    call_names = df_speaker['CallName'].unique()

    for call in call_names:
        df_s = df_speaker[df_speaker['CallName'] == call].sort_values(by="startutt").reset_index(drop=True)
        df_w = df_word[df_word['CallName'] == call].sort_values(by="startutt").reset_index(drop=True)

        i, j = 0, 0
        while i < len(df_s) and j < len(df_w):
            speaker = df_s.iloc[i]
            speaker_start = speaker['startutt']
            speaker_end = speaker['stoputt']
            speaker_name = speaker['channel']

            cur_speaker_transcription = []

            while j < len(df_w):
                word = df_w.iloc[j]
                word_start = word['startutt']
                word_end = word['stoputt']
                word_text = word['content']

                if word_end <= speaker_start:
                    j += 1
                    continue

                if word_start >= speaker_end:
                    break 
            
                if word_start < speaker_end and word_end > speaker_start:
                    cur_speaker_transcription.append(word_text)

                j += 1 
            
            merged_rows.append({
                'CallName': call,
                'channel': speaker_name,
                'startutt': speaker_start,
                'stoputt': speaker_end,
                'content': " ".join(cur_speaker_transcription)
            })
            # print(merged_rows)
            i += 1

    df_merged = pd.DataFrame(merged_rows)
    return df_merged



def transcribe_and_diarize(folder_path, processed_audio_dir, log_dir):
    df_word_level = transcribe_audio_to_dataframe(folder_path, log_dir)
    df_speaker = elan_to_dataframe(folder_path)
    df_merged = assign_words_to_speakers_by_call(df_speaker, df_word_level)

    # Delete the processed_audio_dir if it exists, then create it
    if os.path.exists(processed_audio_dir):
        logging.info(f'Deleting existing directory: {processed_audio_dir}')
        shutil.rmtree(processed_audio_dir)
    
    os.makedirs(processed_audio_dir)
    logging.info(f'Created new directory: {processed_audio_dir}')

    destination = os.path.join(processed_audio_dir, "sentence_level_transcription.csv")
    df_merged['startutt']/=1000
    df_merged['stoputt']/=1000

    df_merged['content'].fillna("", inplace=True)
    df_merged.to_csv(destination)

