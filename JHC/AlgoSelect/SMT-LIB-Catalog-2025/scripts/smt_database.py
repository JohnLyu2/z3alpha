import requests
import tarfile
import zstandard as zstd

"""
url = "https://zenodo.org/records/16290040/files/SMT-LIB-Catalog-2025.tar.zst?download=1"
output_file = "SMT-LIB-Catalog-2025.tar.zst"

response = requests.get(url, stream=True)
response.raise_for_status()  # Check for HTTP errors

with open(output_file, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

print(f"Downloaded {output_file}")
"""

# Decompress .zst file
with open("SMT-LIB-Catalog-2025.tar.zst", "rb") as compressed:
    dctx = zstd.ZstdDecompressor()
    with open("SMT-LIB-Catalog-2025.tar", "wb") as decompressed:
        dctx.copy_stream(compressed, decompressed)

# Extract tar file
with tarfile.open("SMT-LIB-Catalog-2025.tar", "r") as tar:
    tar.extractall(".")

print("Extraction complete!")
