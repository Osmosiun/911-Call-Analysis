import os
import shutil
import pandas as pd
import pympi  # Make sure this is installed
import logging
import mimetypes

def to_milliseconds(time_in_seconds):
    """Convert time from seconds to milliseconds."""
    return int(float(time_in_seconds) * 1000)

def generate_elan_files(audio_folder_path, elan_files_dir,
                        diarized_sentence_level_file_path, diarized_human_file_path=None,
                        diarized_word_level_file_path=None, log_dir=None):
    """
    Generates ELAN files from audio files and their associated transcriptions.

    This function processes audio files located in a specified folder, 
    creating an ELAN file for each audio file by incorporating sentence-level, 
    word-level, and human transcription data. If a folder for ELAN files already exists, 
    it will be deleted and recreated.

    Args:
        audio_folder_path (str): The path to the folder containing audio files.
        elan_files_dir (str): The path where the generated ELAN files will be saved.
        diarized_sentence_level_file_path (str): Path to the CSV file containing sentence-level transcription data.
        diarized_human_file_path (str, optional): Path to the CSV file containing human transcription data.
        diarized_word_level_file_path (str, optional): Path to the CSV file containing word-level transcription data.
        log_dir (str, optional): Directory where the log file will be saved. The log file will be named after the function.

    Returns:
        None: The function generates ELAN files and saves them in the specified folder.

    Raises:
        Exception: If there are issues accessing the input files or writing the output files.

    Logging:
        The function logs the progress and any issues encountered during execution to a specified log file.
    """
    
    # Set up logging if a log directory is provided
    if log_dir:
        log_file_path = os.path.join(log_dir, 'generate_elan_files.txt')
        logging.basicConfig(filename=log_file_path, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Starting ELAN file generation process.')

    # Create ELAN file folder
    if os.path.exists(elan_files_dir):
        shutil.rmtree(elan_files_dir)
    os.makedirs(elan_files_dir, exist_ok=True)

    # Load transcription data
    df_sentence_level = pd.read_csv(diarized_sentence_level_file_path) if diarized_sentence_level_file_path else None
    df_word_level = pd.read_csv(diarized_word_level_file_path) if diarized_word_level_file_path else None
    df_human = pd.read_csv(diarized_human_file_path) if diarized_human_file_path else None

    # Log if any dataframes are empty
    if df_sentence_level is not None and df_sentence_level.empty:
        logging.warning('The sentence level transcription data is empty.')
    if df_word_level is not None and df_word_level.empty:
        logging.warning('The word level transcription data is empty.')
    if df_human is not None and df_human.empty:
        logging.warning('The human transcription data is empty.')

    audio_files = [os.path.join(audio_folder_path, file) for file in os.listdir(audio_folder_path) if file.endswith(('.wav', '.mp3', '.flac'))]
    
    count = 0

    for cur_audio_file_path in audio_files:
        cur_file_name = os.path.basename(cur_audio_file_path).split('.')[0]
        output_elan_file_path = os.path.join(elan_files_dir, cur_file_name + '.eaf')

        # Fetching the transcription for current audio
        df_sentence_level_cur_audio = df_sentence_level[df_sentence_level['CallName'] == cur_file_name].reset_index(drop=True) if df_sentence_level is not None else None
        df_word_level_cur_audio = df_word_level[df_word_level['CallName'] == cur_file_name].reset_index(drop=True) if df_word_level is not None else None
        df_human_cur_audio = df_human[df_human['CallName'] == cur_file_name].reset_index(drop=True) if df_human is not None else None

        count += 1

        # Create an EAF object
        eaf = pympi.Elan.Eaf()

        # Adding AI-generated sentences to the ELAN file
        if df_sentence_level_cur_audio is not None:
            for idx, row in df_sentence_level_cur_audio.iterrows():
                start_time = to_milliseconds(row['startutt'])
                end_time = to_milliseconds(row['stoputt'])
                who_spoke = row['channel']
                what_spoke = row['content']

                if who_spoke not in eaf.get_tier_names():
                    eaf.add_tier(who_spoke)

                eaf.add_annotation(who_spoke, start_time, end_time, what_spoke)

        # Adding AI-generated words to the ELAN file
        if df_word_level_cur_audio is not None:
            for idx, row in df_word_level_cur_audio.iterrows():
                if type(row['content']) != str: 
                    continue
                start_time = to_milliseconds(row['startutt'])
                end_time = to_milliseconds(row['stoputt'])
                who_spoke = row['channel'] + "_word_level"
                what_spoke = row['content']

                if who_spoke not in eaf.get_tier_names():
                    eaf.add_tier(who_spoke)

                eaf.add_annotation(who_spoke, start_time, end_time, what_spoke)

        # Adding human sentences to the ELAN file
        if df_human_cur_audio is not None:
            for idx, row in df_human_cur_audio.iterrows():
                start_time = to_milliseconds(row['startutt'])
                end_time = to_milliseconds(row['stoputt'])
                who_spoke = row['channel']
                what_spoke = row['content']

                if who_spoke not in eaf.get_tier_names():
                    eaf.add_tier(who_spoke)

                eaf.add_annotation(who_spoke, start_time, end_time, what_spoke)

        # Adding the audio file to the ELAN file
        logging.info(f'Processing audio file: {cur_audio_file_path}')
        eaf.add_linked_file(cur_audio_file_path, mimetype='audio/wav')

        # Save the ELAN file
        eaf.to_file(output_elan_file_path)
        logging.info(f'Saved ELAN file: {output_elan_file_path}')

    logging.info('ELAN file generation process completed.')
