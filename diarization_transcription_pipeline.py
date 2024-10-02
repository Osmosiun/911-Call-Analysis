from utils.audio_diarizer import transcribe_audio_to_dataframe
from utils.elan_file_generator import generate_elan_files
import os

def diarization_transcription_pipeline(audio_folder_path, log_dir,
                                       processed_audio_dir, elan_files_dir,
                                       diarized_human_file_path=None):

    transcribe_audio_to_dataframe(
        folder_path=audio_folder_path, processed_audio_dir = processed_audio_dir, log_dir=log_dir
        )
    
    diarized_sentence_level_file_path = os.path.join(processed_audio_dir, "sentence_level_transcription.csv")
    diarized_word_level_file_path = os.path.join(processed_audio_dir, "word_level_transcription.csv")

    diarized_sentence_level_file_path = diarized_sentence_level_file_path if os.path.exists(diarized_sentence_level_file_path) else None
    diarized_word_level_file_path = diarized_word_level_file_path if os.path.exists(diarized_word_level_file_path) else None

    generate_elan_files(
            audio_folder_path=audio_folder_path, elan_files_dir=elan_files_dir,
            diarized_sentence_level_file_path=diarized_sentence_level_file_path, 
            diarized_human_file_path=diarized_human_file_path,
            diarized_word_level_file_path=diarized_word_level_file_path, log_dir=log_dir
            )

if __name__ == "__main__":

    audio_folder_path = '/Users/manavgarg/Downloads/Temp Audios'
    log_dir = "logs"
    processed_audio_dir = '/Users/manavgarg/Downloads/processed_audio'
    elan_files_dir = '/Users/manavgarg/Downloads/elan_dir'
    diarized_human_file_path=None

    diarization_transcription_pipeline(audio_folder_path, log_dir,
                                       processed_audio_dir, elan_files_dir,
                                       diarized_human_file_path)

