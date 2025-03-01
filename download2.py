import gdown
import os
import zipfile

def download_and_unzip(file_id, output_path, extract_path):
    """Downloads a file from Google Drive using its ID and unzips it.

    Args:
        file_id: The Google Drive file ID.
        output_path: The path where the downloaded file should be saved.
        extract_path: The path where the contents of the zip should be extracted.
    """
    url = f'https://drive.google.com/uc?id={file_id}'  # Construct URL from ID
    if not os.path.exists(output_path):
        print(f"Downloading {url} to {output_path}...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"File already exists: {output_path}")

    try:
        with zipfile.ZipFile(output_path, 'r') as zip_ref:
            print(f"Extracting {output_path} to {extract_path}...")
            zip_ref.extractall(extract_path)  # Extract to the desired location
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"Error: {output_path} is not a valid zip file, or is corrupted.")
        os.remove(output_path)  # Delete the corrupted file
    except FileNotFoundError:
        print(f"Error: {output_path} not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    file_id = '1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5'  # Use the FILE ID
    output_file = "dataset.zip"
    extract_path = "data/Specific_test_2"

    download_and_unzip(file_id, output_file, extract_path)
    os.remove(output_file)  # Remove the zip file