import os
from bs4 import BeautifulSoup
import openai
import numpy as np
import time
import json
from pathlib import Path

from akkodis_clients import client_ada_002

client, model = client_ada_002()

def parse_blender_api_html(html_content):
    """
    Parse the HTML content to extract Blender API function definitions along with their parameters and descriptions.

    Args:
        html_content (str): The HTML content of the Blender API documentation.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing function names, parameters, and descriptions.
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all 'py function' elements, which hold function definitions and descriptions
    functions_data = []

    for dl in soup.find_all('dl', class_='py function'):
        function_data = {}

        # Extract the function ID, function name, and parameters from the appropriate elements
        function_name_element = dl.find('dt', class_='sig sig-object py')
        if function_name_element:
            function_id = function_name_element['id']
            function_signature = function_name_element.get_text(strip=True)
            function_data['function_id'] = function_id
            function_data['signature'] = function_signature

        # Extract the function description (usually found in a <dd><p> tag)
        description_element = dl.find('dd').find('p')
        if description_element:
            function_data['description'] = description_element.get_text(strip=True)

        # Extract parameters (contained in 'sig-param' classes)
        params = []
        param_elements = dl.find_all('em', class_='sig-param')
        for param_element in param_elements:
            param_text = param_element.get_text(strip=True)
            params.append(param_text)

        function_data['parameters'] = params

        # Extract parameter descriptions if available
        param_desc_element = dl.find('dl', class_='field-list simple')
        if param_desc_element:
            param_descriptions = []
            for desc_item in param_desc_element.find_all('li'):
                param_desc = desc_item.get_text(strip=True)
                param_descriptions.append(param_desc)
            function_data['parameter_descriptions'] = param_descriptions

        functions_data.append(function_data)

    

    return functions_data



def read_all_html_files_in_folder(folder_path):
    """
    Read all HTML files in a folder and parse them for Blender function data.

    Args:
        folder_path (str): Path to the folder containing HTML files.

    Returns:
        List[Dict[str, Any]]: A list of parsed functions from all the HTML files.
    """
    all_parsed_functions = []
    fail_counter = 0
    total_counter = 0
    for filename in os.listdir(folder_path):

        if filename.endswith(".html"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                total_counter = total_counter + 1

            # Parse the HTML to extract function data
            try:
                parsed_functions = parse_blender_api_html(html_content)
                all_parsed_functions.extend(parsed_functions)  # Add the parsed functions to the overall list
            except:
                fail_counter = fail_counter + 1
                print(fail_counter, total_counter)

    print(fail_counter, total_counter)
    return all_parsed_functions



def create_bulk_embeddings(functions_data, batch_size=1000, output_path="blender_api_embeds.npy", metadata_output_path="blender_api_metadata.json"):
    """
    Create and save embeddings for the parsed functions data in bulk using OpenAI's embedding API.
    Saves embeddings as a numpy .npy file and corresponding metadata as a .json file for easy retrieval in a RAG system.

    Args:
        functions_data (List[Dict[str, Any]]): List of parsed function definitions and descriptions.
        batch_size (int): Number of texts to process in each batch (default is 1000).
        model (str): OpenAI embedding model to use (default is "text-embedding-ada-002").
        output_path (str): Path to the output file where embeddings will be saved (as .npy).
        metadata_output_path (str): Path to the file where metadata will be saved (as .json).

    Returns:
        None
    """

    # Initialize empty lists for embeddings and metadata
    all_embeddings = []
    metadata = []

    # Process embeddings in batches
    for i in range(0, len(functions_data), batch_size):
        batch_functions = functions_data[i:i + batch_size]
        texts_to_embed = []

        # Prepare embedding input texts for the batch
        for function in batch_functions:
            if not function:
                continue

            # Create the embedding input text dynamically
            embedding_input = f"{function.get('signature', '')}\n"
            
            if 'description' in function:
                embedding_input += f"{function['description']}\n"

            if 'parameters' in function:
                embedding_input += f"Parameters:\n{', '.join(function['parameters'])}\n"

            if 'parameter_descriptions' in function:
                embedding_input += f"Descriptions:\n{', '.join(function['parameter_descriptions'])}\n"

            texts_to_embed.append(embedding_input)
            metadata.append(function)  # Store the metadata for this function

        # Ensure retries if rate limit or network error occurs
        while True:
            try:
                embeddings = client.embeddings.create(
                                    model=model,
                                    input=texts_to_embed
                                    ).data
                embeddings = np.array([e.embedding for e in embeddings])
                all_embeddings.extend(embeddings)

                print(f"Processed batch {i // batch_size + 1}/{len(functions_data) // batch_size + 1}")
                break
            except Exception as e:
                print(f"Error occurred: {str(e)}. Retrying after a delay...")
                time.sleep(10)

    # Convert all embeddings to a numpy array and save
    all_embeddings_np = np.array(all_embeddings)
    v_embeds_np = np.vstack(all_embeddings)
    # Save embeddings as a .npy file
    np.save(output_path, all_embeddings_np)
    print(f"Embeddings saved to {output_path}")

    np.save("v_" + output_path, v_embeds_np)

    # Save metadata as a .json file
    with open(metadata_output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Metadata saved to {metadata_output_path}")

    return all_embeddings_np



# Example usage:
if __name__ == "__main__":

    functions_data = read_all_html_files_in_folder('blender_python_reference_4_2')
    # Call the bulk embedding creation function and save embeddings in .npy format
    embeddings_data = create_bulk_embeddings(functions_data, batch_size=1000)

