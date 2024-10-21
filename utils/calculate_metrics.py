import werpy
import pandas as pd
import logging
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.metrics.detection import DetectionErrorRate
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_annotation(cur_df):
    """
    Creates an Annotation object from a DataFrame containing speaker segments.

    Parameters:
    cur_df (DataFrame): DataFrame with 'startutt', 'stoputt', and 'channel' columns.

    Returns:
    Annotation: Pyannote Annotation object.
    """
    annotation = Annotation()
    for index, row in cur_df.iterrows():
        annotation[Segment(float(row["startutt"]), float(row["stoputt"]))] = row["channel"]
    return annotation

def calculate_metrics(ref_file, hyp_file):
    """
    Calculate Word Error Rate (WER), Diarization Error Rate (DER), Jaccard Error Rate (JER),
    and Detection Error Rate based on reference and hypothesis files.

    Parameters:
    ref_file (str): Path to the reference CSV file.
    hyp_file (str): Path to the hypothesis CSV file.

    Returns:
    dict: A dictionary containing average WER, DER, JER, and detection error rates.
    """
    # Load CSV files
    ref = pd.read_csv(ref_file)
    hyp = pd.read_csv(hyp_file)

    # Identify unique calls
    unique_calls = set(hyp['CallName'].unique()).intersection(set(ref['CallName'].unique()))
    logging.info(f"Found {len(unique_calls)} unique calls.")

    # Initialize metrics
    wer_scores = []
    ders = []
    jers = []
    detection_error_rates = []

    for call in unique_calls:
        logging.info(f"Processing call: {call}")
        cur_ref = ref[ref['CallName'] == call].sort_values(by='startutt')
        cur_hyp = hyp[hyp['CallName'] == call].sort_values(by='startutt')

        # Calculate WER
        cur_ref_transcript = werpy.normalize(' '.join(cur_ref['content']))
        cur_hyp_transcript = werpy.normalize(' '.join(cur_hyp['content']))
        wer_score = werpy.wer(cur_ref_transcript, cur_hyp_transcript)
        wer_scores.append(wer_score)

        # Create annotation for DER, JER, and detection error rates
        reference = create_annotation(cur_ref)
        hypothesis = create_annotation(cur_hyp)

        # Calculate Diarization Error Rate (DER)
        der = DiarizationErrorRate()
        der_value = der(reference, hypothesis)
        ders.append(der_value)
        logging.info(f'Diarization Error Rate (DER) for {call}: {der_value:.4f}')

        # Calculate Jaccard Error Rate (JER)
        jer = JaccardErrorRate()
        jer_value = jer(reference, hypothesis)
        jers.append(jer_value)
        logging.info(f'Jaccard Error Rate (JER) for {call}: {jer_value:.4f}')

        # Calculate Detection Error Rate
        detection_error_rate = DetectionErrorRate()
        detection_result = detection_error_rate(reference, hypothesis)
        detection_error_rates.append(detection_result)
        logging.info(f'Detection Error Rate for {call}: {detection_result:.4f}')

    # Compute average metrics
    average_metrics = {
        'Average WER': sum(wer_scores) / len(wer_scores) if wer_scores else float('nan'),
        'Average DER': sum(ders) / len(ders) if ders else float('nan'),
        'Average JER': sum(jers) / len(jers) if jers else float('nan'),
        'Average Detection Error Rate': sum(detection_error_rates) / len(detection_error_rates) if detection_error_rates else float('nan'),
    }

    logging.info("Calculation completed.")
    return average_metrics

if __name__ == "__main__":
    # Example usage
    ref_file_path = "/Users/manavgarg/Documents/Nicholas Duran Project/911-Call-Analysis/Sample Files/sample_human_file.csv"
    hyp_file_path = "/Users/manavgarg/Documents/Nicholas Duran Project/911-Call-Analysis/Sample Files/sample_ai_file.csv"
    
    metrics = calculate_metrics(ref_file_path, hyp_file_path)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")