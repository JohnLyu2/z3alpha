# create_embeddings.py
#
# A standalone script to generate text embeddings for SMT benchmark files.
# It recursively finds all .smt2 files, scrapes relevant info from each file,
# combines the info into a descriptive string, and uses a pre-trained BERT
# model to create a vector representation.
#
# Usage:
# python create_embeddings.py --benchmark-dir /path/to/your/benchmarks --output-file embeddings.pkl
#

import argparse
import pickle
import sys
import re
from pathlib import Path
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm # For a nice progress bar

def extract_description_from_file(file_path: Path) -> str:
    """
    Reads an .smt2 file and extracts key information to form a descriptive string.

    Scrapes the following fields if present:
    - (set-logic ...)
    - Generator
    - Application
    - Target solver
    - Description

    Args:
        file_path: A Path object pointing to the .smt2 file.

    Returns:
        A concatenated string of the found information, or an empty string if
        the file cannot be read or contains no relevant info.
    """
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        
        # --- Use regular expressions to find the information ---
        
        # 1. Extract set-logic
        logic_match = re.search(r'\(set-logic\s+([^\)]+)\)', content)
        logic = logic_match.group(1) if logic_match else ""

        # 2. Extract the entire multi-line :source block
        source_block_match = re.search(r'\(set-info :source \|([\s\S]*?)\|\)', content)
        source_block = source_block_match.group(1) if source_block_match else ""
        
        # 3. Extract individual fields from the source block
        generator_match = re.search(r'Generator:\s*(.*)', source_block)
        generator = generator_match.group(1).strip() if generator_match else ""

        application_match = re.search(r'Application:\s*(.*)', source_block)
        application = application_match.group(1).strip() if application_match else ""

        target_solver_match = re.search(r'Target solver:\s*(.*)', source_block)
        target = target_solver_match.group(1).strip() if target_solver_match else ""
        
        description_match = re.search(r'Description:\s*(.*)', source_block)
        description = description_match.group(1).strip() if description_match else ""

        # --- 4. Combine the extracted parts into a single string ---
        # We only add parts that were actually found to keep the text clean.
        parts = []
        if logic: parts.append(f"Logic: {logic}.")
        if generator: parts.append(f"Generator: {generator}.")
        if application: parts.append(f"Application: {application}.")
        if target: parts.append(f"Target: {target}.")
        if description: parts.append(f"Description: {description}.")
        
        return " ".join(parts)

    except Exception as e:
        print(f"Warning: Could not process file {file_path}. Error: {e}", file=sys.stderr)
        return ""


def main():
    """Main function to parse arguments and generate embeddings."""
    parser = argparse.ArgumentParser(
        description="Generate text embeddings for SMT benchmark files by scraping their content."
    )
    parser.add_argument(
        "--benchmark-dir", type=str, required=True,
        help="Directory containing the .smt2 benchmark files."
    )
    parser.add_argument(
        "--output-file", type=str, default="description_embeddings.pkl",
        help="Path to save the output pickle file."
    )
    parser.add_argument(
        "--model-name", type=str, default="bert-base-uncased",
        help="Name of the pre-trained model from Hugging Face."
    )
    args = parser.parse_args()

    # --- 1. Find all benchmark files ---
    benchmark_dir = Path(args.benchmark_dir)
    if not benchmark_dir.is_dir():
        print(f"Error: Directory not found at {args.benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for .smt2 files in '{benchmark_dir}'...")
    all_benchmark_paths = list(benchmark_dir.rglob("*.smt2"))

    if not all_benchmark_paths:
        print(f"Error: No .smt2 files found in {args.benchmark_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(all_benchmark_paths)} benchmark files.")

    # --- 2. Load pre-trained model ---
    print(f"Loading pre-trained model '{args.model_name}'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name).to(device)
    model.eval()
    embedding_dim = model.config.hidden_size

    # --- 3. Generate and store embeddings ---
    benchmark_embeddings = {}
    print(f"Generating {embedding_dim}-dimensional embeddings from file content...")

    for path_obj in tqdm(all_benchmark_paths, desc="Processing benchmarks"):
        absolute_path = str(path_obj.resolve())
        # The key step: extract description directly from the file
        description_text = extract_description_from_file(path_obj)

        if description_text:
            inputs = tokenizer(
                description_text, return_tensors='pt', padding=True,
                truncation=True, max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
            
            sentence_vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten().tolist()
            benchmark_embeddings[absolute_path] = sentence_vector
        else:
            # If no info was scraped, use a vector of zeros.
            benchmark_embeddings[absolute_path] = [0.0] * embedding_dim

    # --- 4. Save the embeddings to a file ---
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(benchmark_embeddings, f)

    print("\n✅ Success!")
    print(f"Saved {len(benchmark_embeddings)} embeddings to '{output_path}'")


if __name__ == "__main__":
    main()