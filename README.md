# Diarization and Transcription Pipeline

This project provides a pipeline for diarizing, transcribing, and converting 911 call audio files into ELAN files, along with speaker role identification and metric calculation (WER, DER, JER). The pipeline integrates multiple components to process audio files, including speaker identification and transcription evaluation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Directory Structure](#directory-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
    1. [Running Diarization and Transcription Pipeline](#running-diarization-and-transcription-pipeline)
    2. [Speaker Identification](#speaker-identification)
    3. [Calculating Metrics](#calculating-metrics)
6. [Logging](#logging)
7. [Future Work](#future-work)

## Project Overview

The main goal of this project is to automate the process of diarizing and transcribing 911 call audio files, identifying the roles of speakers (caller or operator), and calculating key metrics like Word Error Rate (WER), Diarization Error Rate (DER), and Jaccard Error Rate (JER). The pipeline generates both sentence-level and word-level transcriptions and outputs ELAN files for further analysis.

## Directory Structure

```
.
├── diarization_transcription_pipeline.py   # Main pipeline file
├── utils
│   ├── audio_diarizer.py                   # Audio diarization and transcription
│   ├── elan_file_generator.py              # ELAN file generation
│   ├── speaker_identification.py           # Speaker role identification
│   ├── calculate_metrics.py                # WER, DER, JER calculation
├── logs                                    # Directory for log files
├── requirements.txt                        # Dependencies
```

## Requirements

This project uses several libraries for audio processing, transcription, and evaluation. Install the dependencies using the `requirements.txt` file.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. [Optional] You will need an API key for **AssemblyAI** for transcription:
   - Replace the placeholder API key in `audio_diarizer.py` with your actual API key:
     ```python
     aai.settings.api_key = "your-assemblyai-api-key"
     ```

## Usage

### 1. Running Diarization and Transcription Pipeline

This script processes a folder of audio files, performs diarization, transcription, and generates corresponding ELAN files.

To run the main pipeline:

```bash
python diarization_transcription_pipeline.py
```

**Arguments:**
- `audio_folder_path`: Path to the folder containing audio files.
- `log_dir`: Directory to store log files.
- `processed_audio_dir`: Directory where processed audio and CSVs will be saved.
- `elan_files_dir`: Directory where the generated ELAN files will be saved.

The script will output sentence-level and word-level CSV transcriptions, along with ELAN files.

### 2. Speaker Identification

Before calculating metrics, the speakers must be identified using the `speaker_identification.py` script.

Run the speaker identification script on the transcribed CSV files to label speakers as either 'caller' or 'operator':

```bash
python utils/speaker_identification.py
```

**Arguments:**
- `csv_file`: Path to the CSV file containing 911 call data.
- `num_speaker_occurrence`: Number of occurrences for each speaker to be analyzed.

This script currently uses the **Ollama** model for speaker role identification, but you may replace it with **Groq** in the future for better GPU performance.

### 3. Calculating Metrics

Once the speaker roles have been identified, you can calculate key metrics like WER, DER, and JER using the `calculate_metrics.py` script:

```bash
python utils/calculate_metrics.py
```

**Arguments:**
- `ref_file`: Path to the reference CSV file (human transcription).
- `hyp_file`: Path to the hypothesis CSV file (AI-generated transcription).

This will calculate and output the following metrics:
- Word Error Rate (WER)
- Diarization Error Rate (DER)
- Jaccard Error Rate (JER)
- Detection Error Rate

### 4. Logging

Logs are created for each component in the respective log directories. For example:
- `transcribe_audio_to_dataframe.txt` logs the audio transcription process.
- `generate_elan_files.txt` logs the ELAN file generation process.
- Other scripts will log their actions under the main `logs/` directory.

### Future Work

- **Groq Integration**: To replace **Ollama** for more efficient GPU usage in speaker identification.
- **Additional Metrics**: Expanding the types of metrics calculated for transcription quality.
