import pickle

# --- 1. Define the path to your file ---
file_path = 'description_embeddings.pkl' # Make sure this is the correct path

# --- 2. Load the data from the pickle file ---
try:
    with open(file_path, 'rb') as f:
        # The pickle.load() function reads the pickled data
        embeddings_data = pickle.load(f)

    # --- 3. Debugging: Inspect the loaded data ---

    # Check the type of the data structure (it should be a dictionary)
    print(f"✅ Successfully loaded '{file_path}'")
    print(f"Type of loaded data: {type(embeddings_data)}\n")

    # Check how many benchmark embeddings are in the dictionary
    if isinstance(embeddings_data, dict):
        print(f"Total number of embeddings found: {len(embeddings_data)}\n")

        # Get the list of all keys (file paths)
        all_paths = list(embeddings_data.keys())

        # --- 4. Print the first 3 items for a quick look ---
        print("--- Displaying the first 3 items for debugging ---")
        for i in range(min(3, len(all_paths))):
            path = all_paths[i]
            embedding = embeddings_data[path]

            print(f"\nItem {i+1}:")
            print(f"  File Path (Key): {path}")
            
            # Print the embedding vector itself
            # We'll only show the first 5 and last 5 dimensions to keep it readable
            embedding_preview = f"[{', '.join(map(str, embedding[:5]))}, ..., {', '.join(map(str, embedding[-5:]))}]"
            print(f"  Embedding (Value): {embedding_preview}")
            print(f"  Embedding Dimension: {len(embedding)}")


except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")