import logging
import pandas as pd
import os
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall, answer_correctness
from ragas import RunConfig, evaluate

# Constants
OPENAI_API = os.getenv("OPENAI_API_KEY")
MAX_LENGTH = 1000  # Limit for long text fields

# Define root and storage paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_PATH = os.path.join(ROOT_DIR, "data/dataset/")
os.makedirs(STORAGE_PATH, exist_ok=True)

data_path = os.path.join(STORAGE_PATH, "evaluation_dataset.xlsx")  # Input file
output_path = os.path.join(STORAGE_PATH, "evaluation_results.xlsx")  # Output file

logging.basicConfig(
    level=logging.INFO,  # Reduced log level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load dataset
logging.info("Loading dataset...")
df = pd.read_excel(data_path, dtype=str)
logging.info("Dataset loaded successfully with %d entries.", len(df))

# Ensure the dataset contains the required fields
required_columns = ["query", "ground_truth", "retrieved_contexts", "answer"]
if not all(column in df.columns for column in required_columns):
    raise ValueError(f"Dataset must contain the following columns: {required_columns}")
logging.info("Dataset validated successfully.")

# Replace NaN values and convert "retrieved_contexts" to lists if needed
df = df.fillna({
    "query": "",
    "ground_truth": "",
    "retrieved_contexts": "",
    "answer": ""
})

def clean_retrieved_contexts(contexts):
    if isinstance(contexts, str):
        return [context.strip() for context in contexts.split("|||") if context.strip()]
    elif isinstance(contexts, list):
        return [str(context).strip() for context in contexts if isinstance(context, str)]
    return []

df["retrieved_contexts"] = df["retrieved_contexts"].apply(clean_retrieved_contexts)

# Truncate long text fields to avoid processing issues
for col in ["ground_truth", "answer"]:
    df[col] = df[col].apply(lambda x: x[:MAX_LENGTH] if isinstance(x, str) else "")

# Validate dataset for RAGAS compatibility
def validate_dataset(df):
    errors = []
    for i, row in df.iterrows():
        if not isinstance(row["retrieved_contexts"], list):
            errors.append(f"Row {i}: 'retrieved_contexts' should be a list.")
        if not isinstance(row["query"], str):
            errors.append(f"Row {i}: 'query' should be a string.")
        if not isinstance(row["ground_truth"], str):
            errors.append(f"Row {i}: 'ground_truth' should be a string.")
        if not isinstance(row["answer"], str):
            errors.append(f"Row {i}: 'answer' should be a string.")
    if errors:
        for error in errors:
            logging.error(error)
        raise ValueError("Dataset validation errors encountered.")

validate_dataset(df)

# Prepare evaluation inputs
data = {
    "question": df["query"].tolist(),
    "ground_truth": df["ground_truth"].tolist(),
    "retrieved_contexts": df["retrieved_contexts"].tolist(),
    "answer": df["answer"].tolist(),
}
dataset = Dataset.from_dict(data)

# RAGAS runtime settings
run_config = RunConfig(max_workers=4, max_wait=180)

# Define metrics for evaluation
metrics = [faithfulness, answer_relevancy, context_recall, answer_correctness]

# Enhanced Debugging for Evaluation Process
logging.info("Starting evaluation...")
try:
    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=run_config,
        raise_exceptions=False,
    )

    # Process results into separate tables
    # Process results into separate tables
# Process results into separate tables
    if hasattr(results, "__dict__"):
        raw_data = results.__dict__

        # Extract metrics for each row
        scores_list = raw_data.get("scores", [])
        average_scores = raw_data.get("_scores_dict", {})

        # Create detailed DataFrame for scores
        scores_df = pd.DataFrame(scores_list)
        scores_df["query"] = df["query"]

        # Handle average scores
        try:
            # Compute the average for each metric
            processed_average_scores = {metric: sum(scores) / len(scores) for metric, scores in average_scores.items()}

            # Create DataFrame from the processed average scores
            summary_df = pd.DataFrame.from_dict(processed_average_scores, orient="index", columns=["Average Score"])
            summary_df.reset_index(inplace=True)
            summary_df.columns = ["Metric", "Average Score"]
        except ValueError as e:
            logging.error("Error processing average scores: %s", e)
            logging.error("Raw average_scores: %s", average_scores)
            raise

        # Save both to the Excel file
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            scores_df.to_excel(writer, index=False, sheet_name="Detailed Scores")
            summary_df.to_excel(writer, index=False, sheet_name="Average Scores")
        logging.info(f"Evaluation results saved to {output_path}")
    else:
        raise ValueError("Unexpected results format.")

except Exception as e:
    logging.error("Error during evaluation:")
    logging.exception(e)
