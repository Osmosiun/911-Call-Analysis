import assemblyai as aai
import os
import pandas as pd
import logging

def transcribe_audio_to_dataframe(folder_path, log_dir):
    """
    Transcribes audio files in the specified folder and generates two dataframes: 
    one at the sentence level and another at the word level.
    
    The output files will have the same names as the input audio files but without their extensions.

    Args:
        folder_path (str): The path to the folder containing audio files.
        log_dir (str): The directory where the log file will be saved.

    Returns:
        pd.DataFrame: DataFrame containing sentence-level transcription data.
        pd.DataFrame: DataFrame containing word-level transcription data.
    """
    
    # Set up logging
    log_file_path = os.path.join(log_dir, 'transcribe_audio_to_dataframe.txt')
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info('Starting transcription process.')
    
    file_names = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file) for file in file_names]

    file_num = 1

    df_sentence_level = pd.DataFrame(columns=['CallName', 'filenum', 'channel', 'startutt', 'stoputt', 'duration', 'content'])
    df_word_level = pd.DataFrame(columns=['CallName', 'filenum', 'channel', 'startutt', 'stoputt', 'duration', 'content'])

    for idx, file_path in enumerate(file_paths):
        audio_url = file_path
        config = aai.TranscriptionConfig(speaker_labels=True)

        logging.info(f'Transcribing file: {file_names[idx]}')
        transcript = aai.Transcriber().transcribe(audio_url, config)

        # Get the CallName by finding the last dot and omitting everything after it
        call_name = file_names[idx][:file_names[idx].rfind(".")]

        for utterance in transcript.utterances:
            new_row = {
                'CallName': call_name,
                'filenum': file_num,
                'channel': utterance.speaker,
                'startutt': utterance.start / 1000,
                'stoputt': utterance.end / 1000,
                'duration': utterance.end / 1000 - utterance.start / 1000,
                'content': utterance.text
            }
            df_sentence_level.loc[len(df_sentence_level)] = new_row

            for utterance_word in utterance.words:
                new_row = {
                    'CallName': call_name,
                    'filenum': file_num,
                    'channel': utterance_word.speaker,
                    'startutt': utterance_word.start / 1000,
                    'stoputt': utterance_word.end / 1000,
                    'duration': utterance_word.end / 1000 - utterance_word.start / 1000,
                    'content': utterance_word.text
                }

                df_word_level.loc[len(df_word_level)] = new_row

        file_num += 1

    logging.info('Transcription process completed.')
    return df_sentence_level, df_word_level
