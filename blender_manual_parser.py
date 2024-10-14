
import os
import json
import numpy as np
from bs4 import BeautifulSoup
import openai  # Ensure you have your API key setup
import time

from akkodis_clients import client_ada_002

client, model = client_ada_002()


def parse_blender_manual_html(html_content, file_path):
    """
    Parse Blender Manual HTML content to extract structured data for embeddings by header level.

    Args:
        html_content (str): HTML content of the Blender manual.
        file_path (str): Path of the HTML file for metadata tracking.

    Returns:
        dict: Parsed data with headers and associated text content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    data = {
        'file_path': file_path,
        'headers': {}
    }

 
    current_h1 = None

    # Iterate over all header tags (h1, h2, h3, h4, etc.) and collect content under each
    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4','h5','h6','h7','h8','h9']):
        if tag.name == 'h1':
            current_h1 = tag.get_text(strip=True)
            data['headers'][current_h1] = {}

        elif current_h1:  # If there is an active h1, gather h2, h3, etc., under it
            header_text = tag.get_text(strip=True)
            data['headers'][current_h1][header_text] = ''

            # Collect all content (paragraphs, lists, <dl> structures) under this header
            next_siblings = tag.find_next_siblings()
            for sibling in next_siblings:
                if sibling.name in [ 'h2', 'h3', 'h4','h5','h6','h7','h8','h9']:
                    data['headers'][current_h1][header_text] += sibling.get_text(strip=True) + ' : '

                if sibling.name == 'p':  # Paragraphs
                    data['headers'][current_h1][header_text] += sibling.get_text(strip=True) + ' '


                if sibling.name == 'p':  # Paragraphs
                    data['headers'][current_h1][header_text] += sibling.get_text(strip=True) + ' '

                if sibling.name in ['ul', 'ol']:  # Lists
                    for li in sibling.find_all('li'):
                        data['headers'][current_h1][header_text] += '- ' + li.get_text(strip=True) + ' '

                if sibling.name == 'dl':  # <dl> for definitions and parameters
                    for dt in sibling.find_all('dt'):
                        dt_text = dt.get_text(strip=True)
                        dd_text = dt.find_next('dd').get_text(strip=True) if dt.find_next('dd') else ''
                        data['headers'][current_h1][header_text] += f"\n{dt_text}: {dd_text}"

                if sibling.name in ['h1']:  # Stop at next header
                    break

    return data


def create_embeddings_and_metadata(html_data, batch_size=1000, output_path="./v_blender_manual_embeds.npy", metadata_output_path="./blender_manual_metadata.json"):
    """
    Create and save embeddings for the parsed manual data using OpenAI's embedding API.
    
    Args:
        html_data (list): List of dictionaries containing parsed HTML data for embedding.
        batch_size (int): Number of texts to process in each batch.
        output_path (str): Path to save the embeddings (.npy format).
        metadata_output_path (str): Path to save the metadata (.json format).
    
    Returns:
        None
    """
    all_embeddings = []
    metadata = []

    for i in range(0, len(html_data), batch_size):
        batch = html_data[i:i + batch_size]
        texts_to_embed = []

        # Prepare embedding texts
        for entry in batch:
            for h1, subheaders in entry['headers'].items():
                # Embed the h1 along with each subheader (h2, h3, etc.)
                for subheader, content in subheaders.items():
                    embedding_input = f"{h1}\n{subheader}\n{content}"
                    texts_to_embed.append(embedding_input)
                    metadata.append({
                        'h1': h1,
                        'subheader': subheader,
                        'content': content
                    })

        # Send batch to OpenAI for embeddings
        while True:
            try:
                embeddings = client.embeddings.create(
                    input=texts_to_embed,
                    model=model
                ).data
                embeddings = np.array([e.embedding for e in embeddings])
                all_embeddings.extend(embeddings)

                print(f"Processed batch {i // batch_size + 1}/{len(html_data) // batch_size + 1}")
                break
            except Exception as e:
                print(f"Error: {str(e)}, retrying after delay...")
                time.sleep(10)

    # Save the embeddings and metadata
    np.save(output_path, np.vstack(all_embeddings))
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Embeddings saved to {output_path}")
    print(f"Metadata saved to {metadata_output_path}")


def read_and_process_html_folder(folder_path):
    """
    Read all HTML files in the folder (including subfolders) and process them for embeddings.

    Args:
        folder_path (str): Root folder containing Blender manual HTML files.

    Returns:
        list: List of parsed HTML data ready for embedding.
    """
    html_data = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    parsed_data = parse_blender_manual_html(html_content, file_path)
                    html_data.append(parsed_data)

    return html_data


# Example usage:
if __name__ == "__main__":
    folder_path = './blender-manual/build/html/'  # Replace with your actual folder path
    parsed_html_data = read_and_process_html_folder(folder_path)
    
    # Create embeddings and metadata
    create_embeddings_and_metadata(parsed_html_data, batch_size=100)
