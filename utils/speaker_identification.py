import logging
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_speakers_in_calls(csv_file: str, num_speaker_occurence: int) -> pd.DataFrame:
    """
    Analyze the speakers in 911 call conversations from a CSV file,

    Args:
        csv_file (str): The path to the CSV file containing 911 call data.
        num_speaker_occurence(int): Number of conversation of each speaker per call.

    Returns:
        pd.DataFrame: A DataFrame containing call names, speaker names, and their corresponding roles.
    """
    # Initialize the language model
    llm = Ollama(model="llama3")

    # Define a prompt template for the chatbot
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Please respond to the questions accurately."),
            ("user", "Question:{question}")
        ]
    )

    # Create a chain that combines the prompt and the Ollama model
    chain = prompt | llm

    # Read the CSV file
    logging.info(f'Reading data from {csv_file}')
    df = pd.read_csv(csv_file)

    # Initialize an empty list to store results
    results = []

    # Iterate through unique call names
    for call_name in df['CallName'].unique():
        logging.info(f'Processing call: {call_name}')

        temp_df = df[df['CallName'] == call_name]

        # Limit each speaker to a maximum of 5 occurrences
        limited_df = temp_df.groupby('channel').head(5)

        # Prepare the conversation string
        limited_df['final_conversation'] = limited_df['channel'] + ': ' + limited_df['content']
        unique_speakers = limited_df['channel'].unique()
        whole_conversation = '\n'.join(limited_df['final_conversation'])

        # Prepare the input prompt with strict instructions
        input_prompt = f'''
        Below is the whole conversation.
        {whole_conversation}
        Each line represents '<speaker name>: <what speaker spoke>'.
        This is a 911 call conversation. The caller describes the incident while the operator assists them with questions. There can be multiple operators and callers.

        I want to determine the role of each speaker (either 'operator' or 'caller').
        Speaker list: {', '.join(unique_speakers)}

        Please respond only with the following CSV format:
        speaker_name, speaker_role
        (without any extra text)

        The header should be:
        speaker_name, speaker_role

        Only use the values 'operator' or 'caller' for speaker_role.
        '''

        logging.info(f'Generating response for call: {call_name}')
        output = chain.invoke({"question": input_prompt})

        # Process the output
        for line in output.splitlines():
            line = line.strip()
            if line and not line.startswith('speaker_name'):
                # Attempt to split only on the first comma to avoid too many values error
                if line.count(',') == 1:
                    speaker_name, speaker_role = line.split(',')
                    results.append({'call_name': call_name, 'speaker_name': speaker_name.strip(), 'speaker_role': speaker_role.strip()})
                else:
                    logging.warning(f"Unexpected output format for call {call_name}: {line}")

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    logging.info('Processing complete.')

    return results_df
