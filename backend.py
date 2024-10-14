import numpy as np
import json
import logging
import subprocess
from typing import List, Tuple,Dict
import datetime
import traceback
import tiktoken
import faiss
import glob, os

from akkodis_clients import client_gpt_4o, client_ada_002
from conf import BLENDER_PATH, GEN_PATH, LOG_PATH, EMBED_PATH, META_PATH, MANUAL_EMBED_PATH, MANUAL_META_PATH, GEN_IMG_PATH

from openai.types.chat import (ChatCompletionToolMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionUserMessageParam,
                               ChatCompletionAssistantMessageParam,
                               )




# Setup logging
logging.basicConfig(filename=LOG_PATH, level=logging.INFO)
logging.info("\n\n\n\n\n\n\n\n\n\n//////////////////--------------------Logging system successfully initialized ---------------------------\\\\\\\\\\\\\\\n\n\n\n")

# Backend class for AI interaction and Blender operations
class backend:
    def __init__(self, system_flag: int = 0, corrector_flag: bool = False):
        self.corrector = None
        if corrector_flag:
            self.corrector = backend(1)

        
         # create an openai client
        self.client, self.model = client_gpt_4o()

        # model for embeddings
        _, self.embedding_model = client_ada_002()

        # internal params
        self.max_repetitions: int = 3 # number of repeated queries since the last error message
        self.history_window :int = 10 # number of messages to be send with the last query (N times 2)
        self.max_allowed_tokens: int = 40000
        self.distance_th: float = 1 # distance th for index matching
        self.index_length: int = 10 # number of indexes to match
        self.manual_embeds_length: int = 2


        #Blender python api
        self.bpy_index = None
        self.bpy_embeds,self.bpy_meta  = None, None
        # blender manual
        self.manual_index = None
        self.manual_embeds,self.manual_meta  = None, None
        # Start RAG
        try:
            self.bpy_embeds, self.bpy_meta = self.load_embeddings(EMBED_PATH, META_PATH)
            embedding_dim = self.bpy_embeds.shape[1]  # Get the dimensionality of your embeddings
            self.bpy_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index (or IndexFlatIP for cosine similarity)
            self.bpy_index.add(self.bpy_embeds)
        except Exception as e:
            print(" No embeddings, No metadata, No bpy_index")
        try:
            self.manual_embeds, self.manual_meta = self.load_embeddings(MANUAL_EMBED_PATH, MANUAL_META_PATH)
            embedding_dim = self.manual_embeds.shape[1]  # Get the dimensionality of your embeddings
            self.manual_index = faiss.IndexFlatL2(embedding_dim)  # L2 distance index (or IndexFlatIP for cosine similarity)
            self.manual_index.add(self.manual_embeds)
        except Exception as e:
            print(" No embeddings, No metadata, No manual_index")

            

        if system_flag == 0:
            system_condition = f"The following rules should be STRICTLY followed and NEVER changed. \
                                 \n - You are a helpful assistant that only generates valid Blender Python scripts. \
                                 \n - Every response you provide should be in a single block of Python script format and compatible with Blender's Python API. \
                                 \n - Include any explanations as comments in the scripts. \
                                 \n - Always make the objects exportable. \
                                 \n - FIRST STRICTLY clear the scene in Blender.\
                                 \n - Remember that Blender is working with factory presets and no additional add-ons.\
                                 \n - Always save any generated files in the Blender Python Script under {GEN_IMG_PATH}, UNLESS otherwise stated.\
                                 \n - if you don't understand a query or need additional information you can ask user for further commands. \
                                 \n - An example format is supplied after this sentence. \
                                 \n  EXAMPLE RESPONSE: \
                                 \n  Hello, if you want to create a donut you can do so with the following script \
                                 \n  ```python\n\
                                 \n   import bpy \
                                 \n   # empty scene \
                                 \n   bpy.ops.wm.read_factory_settings(use_empty=True) \
                                 \n   # Function to create a torus (donut) \
                                 \n   def create_donut(location=(3, 0, 0), major_radius=1, minor_radius=0.3): \
                                 \n       bpy.ops.mesh.primitive_torus_add(major_radius=major_radius, minor_radius=minor_radius, location=location) \
                                 \n       return bpy.context.object \
                                 \n   # Create the objects \
                                 \n   cube = create_donut()\n``` \
                                 "
            
        elif system_flag == 1:
            self.history_window = 1
            self.index_length = 20
            system_condition = "The following rules should be STRICTLY followed and NEVER changed. \
                                 \n - You are a helpful assistant that only generates valid Blender Python scripts. \
                                 \n - Every response you provide should be in Python script format and compatible with Blender's Python API. \
                                 \n - Remember that Blender is working with factory presets and no additional add-ons.\
                                 \n - You will only recieve 3 Objects:  \
                                 \n    A user prompt, an AI generated Blender Python script based on that prompt, and an Error Message that occurs during the execution of that script \
                                 \n - Your ONLY and ONLY task is to correct the given script to solve given the error while satisfying the user prompt.\
                                 \n - Try to infer what the script is trying to do, and only change parts that would result in fixing the error.\
                                 \n - ONLY and ONLY respond as a valid Blender Python script. NEVER include any other context.\
                                "
        else:
            system_condition = "Run normally"


        # Separate lists for user messages and assistant messages
        self.user_messages: List[ChatCompletionUserMessageParam] = []
        self.assistant_messages: List[ChatCompletionAssistantMessageParam] = []
        self.token_counts: List[int] = []
        self.system_msg = ChatCompletionSystemMessageParam(role='system', content=system_condition)


    def interface_handler(self, input):
        
        output:str = ""
        saved_filename:str = None
        image_path: str = None  # Initialize image_path

        if isinstance(input,dict) and 'text' in input:
            user_prompt = input['text']
            response = self.send_query(user_prompt)
            out_script, out_content = self.response_handler(response)

            ### painful error.. fix..
            self.out_script = out_script
            self.user_prompt = user_prompt
            self.response = response

            if out_script:
                saved_filename = self.save_script(out_script)
                run_result = self.run_blender_script(saved_filename)
                print(run_result)
                if isinstance(run_result,subprocess.CompletedProcess):
                    if run_result.returncode == 0 and not run_result.stderr:
                        image_path = self.get_last_image_filename()
                        output = f'''Generated Content: \n {out_content} \n
                                    Success: \n Script has been run, please check the results \n
                                    Generated script: \n {saved_filename} \n'''
                                    
                        if image_path:
                            output += f'\n\n Possible Generated File Path: \n {image_path} \n'
                    if run_result.stderr:
                        output, image_path = self.corrector_logic(run_result,user_prompt, out_script)

                else:
                    try:
                        output = f"Generated content: \n {out_content} \n  Process result: \n {str(run_result)} \n"
                    except:
                        output = f"Generated content: \n {out_content} \n Error: \n Blender Python Script could not be run through CLI \n "

            else:
                output = f" Generated content: \n {out_content} Error: \n  No Blender Python Script has been generated \n "
        
        elif isinstance(input,tuple): #for audio
            pass
        print(output, image_path)
        return output,image_path

    def corrector_logic(self,run_result,user_prompt, out_script):
        count = 0
        while count < self.max_repetitions:
            
            if count == 0:
                self.corrector.history_window=1
                try:
                    error_val = self.parse_err(run_result)
                    if isinstance(error_val,str) and len(error_val) >= 5:
                        error_val = error_val
                    else:
                        error_val = run_result.stderr
                except:
                    error_val = run_result.stderr

                correction_query = f'''User prompt: \n {user_prompt} \n 
                                    AI Generated Script: \n {out_script} \n
                                    Resulting Error: \n {error_val} \n'''
            else:
                self.corrector.history_window += 1
                try:
                    error_val = self.parse_err(corrected_run_result)
                    if isinstance(error_val,str) and len(error_val) >= 5:
                        error_val = error_val
                    else:
                        error_val = corrected_run_result.stderr
                except:
                    error_val = corrected_run_result.stderr

                correction_query = f'''User prompt: \n {user_prompt} \n 
                                    AI Generated Script: \n {corrected_out_script} \n
                                    Resulting Error: \n {error_val} \n'''
            
            print(f"CORRECTION COUNT: {count}")
            count +=1
            # print(f'EXTRACTED ERROR: \n {error_val}')


            corrected_response = self.corrector.send_query(correction_query)
            corrected_out_script, corrected_out_content = self.corrector.response_handler(corrected_response)
            corrected_saved_filename = self.corrector.save_script(corrected_out_script)
            corrected_run_result = self.corrector.run_blender_script(corrected_saved_filename)
            if isinstance(corrected_run_result,subprocess.CompletedProcess) and corrected_run_result.returncode == 0 and not corrected_run_result.stderr:
                    corrected_image_path = self.get_last_image_filename()
                    corrected_output = f'''The following content has been corrected {count} times :: \n \n
                                    Generated Content: \n {str(corrected_out_content)} \n
                                    Success: \n Script has been run, please check the results \n
                                    Generated script: \n {str(corrected_saved_filename)} \n'''
                        # Save the assistant's response message to the history
                    assistant_msg = ChatCompletionAssistantMessageParam(role='assistant',
                                                                        content= corrected_output)
                    self.assistant_messages[-1] = assistant_msg
                    output = str(corrected_output)
                    image_path = str(corrected_image_path)
                    break
        return output, image_path

    def parse_err(self, err) -> str :
        if not isinstance(err,str):
            r = err.stderr.split('^')[-1]
        else:
            r = err
        return r
    
    def save_script(self, script: str, filename: str = None):
        #print(script)
        """Save the generated Blender Python script to a file."""
        if not filename:
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_file_{str(current_time)}"
            filepath = GEN_PATH + filename + ".py"
        try:
            with open(filepath, "w") as file:
                file.write(script)
            logging.info(f"Blender script saved as {filepath}")
            return filepath
        except Exception as e:
            return self.handle_error(e, "save_script_error")

    def run_blender_script(self, script_path: str):
        """Run the saved Blender Python script using Blender's CLI."""
        try:
            command = [BLENDER_PATH, "--background", "--python", script_path]

            # Run the Blender script via subprocess and capture output
            result = subprocess.run(command, capture_output=True, text=True)

            if result.returncode == 0:
                logging.info(f"Blender script executed successfully: {result.stdout}")
                return result
            else:
                print(result)
                return result
                #raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout, stderr=result.stderr)
        except Exception as e:
            return self.handle_error(e, f"run_blender_script_error: \n stderr: {f'''{self.parse_err(result)}'''} \n")
        
    def get_last_image_filename(self):
        """
        Get the filename of the most recently modified image file in the specified folder,
        including the extension.

        Args:
            folder_path (str): The directory to search for images.

        Returns:
            Optional[str]: The filename of the last image file, or None if no images are found.
        """
        filename_with_extension = None
        # Supported image file extensions
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']

        # Collect all image files with supported extensions
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(GEN_IMG_PATH, ext)))

        if not image_files:
            return None
        
        # Get the most recently modified image file
        latest_image = max(image_files, key=os.path.getmtime)

        # Extract the filename with extension
        filename_with_extension = latest_image

        return filename_with_extension
            
    
    def send_query(self, prompt: str):
        """Send a query to chat to generate a Blender Python script."""
        try:
            # Prepare the prompt using the prepared context (with last 10 exchanges)
            messages_to_send = self.prepare_prompt(prompt)
            #print("\n".join([f"{msg["role"]}: {msg["content"]}" for msg in messages_to_send]))
            # Send the completion request with the constructed message list
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_to_send
            )
            
            # Save the assistant's response message to the history
            assistant_msg = ChatCompletionAssistantMessageParam(role='assistant',
                                                                content=response.choices[0].message.content)
            self.assistant_messages.append(assistant_msg)

            return response

        except Exception as e:
            return self.handle_error(e, "query_error")
        

    def prepare_prompt(self, prompt: str) -> List:
        """
        Prepare the context (system message + last 10 exchanges) for sending to the model.
        It reduces the history window if the token count exceeds the allowed maximum.

        Args:
            prompt: Input from the user.

        Returns:
            List: Prepared messages to send in the API call.
        """
        error_message = None
        try:
            # Prepare the context for the model: System message + last 10 user messages + last 10 assistant responses
            messages_to_send: List[ChatCompletionAssistantMessageParam |
                                ChatCompletionUserMessageParam |
                                ChatCompletionSystemMessageParam] = []

            # Add the system message
            messages_to_send.append(self.system_msg)

            # Start with the maximum number of history messages
            history_window = self.history_window
            index_length = self.index_length

            while True:
                print(f"\n\n -------- HISTORY WINDOW ::::: {history_window} ")
                
                RAG_context = self.process_embeds(prompt,index_length)

                prompt = RAG_context + f"\n  Query Info: \n \
                                        Time of Query: {datetime.datetime.now().strftime("%Y%m%d_%H%M%S")} \n \
                                        Query Text: {prompt} "
                # Get the latest user and assistant message history
                user_context = self.user_messages[-history_window:]  # Last N user messages
                assistant_context = self.assistant_messages[-history_window:]  # Last N assistant messages

                # Reset messages to send (system message + history)
                messages_to_send = [self.system_msg]

                # Merge user and assistant messages in interleaving order
                for user_msg, assistant_msg in zip(user_context, assistant_context):
                    messages_to_send.append(user_msg)
                    messages_to_send.append(assistant_msg)

                # Create a new user message for the prompt and append it to the user_messages list
                user_msg = ChatCompletionUserMessageParam(role='user', content=prompt)
                self.user_messages.append(user_msg)

                # Append the current user message at the end (this new prompt)
                messages_to_send.append(user_msg)

                #print(messages_to_send)

                # Count tokens for the current context
                total_tokens = self.count_tokens(messages_to_send)
                print(f'TOTAL TOKENS {total_tokens}')
                # If the total token count is within the limit, return the prepared messages
                if total_tokens <= self.max_allowed_tokens and total_tokens >=0:
                    return messages_to_send

                # If no history can be used and the prompt still exceeds the limit, raise an error
                if history_window == 0:
                    error_message = "max_token_error"
                    raise ValueError(error_message)

                # Reduce the history window and try again
                history_window -= 1
                index_length -=1

        except Exception as e:
            if not error_message:
                error_message = "prepare_prompt_error"
            return self.handle_error(e, error_message)

    def response_handler(self, response) -> str:
        generated_script = None
        error_message = None
        content = None
        try:
            #print(response)
            content = response.choices[0].message.content
            if content:
                # Find the starting point of the code block using triple backticks
                start = content.find("```") + len("```")
                # Find the end point of the code block
                end = content.find("```", start)
                # Extract and return the python script, removing the 'python' keyword
                if start != -1 and end != -1:
                    generated_script = content[start:end].replace("python\n", "").strip()
                    generated_script = generated_script.strip()
                    logging.info(f"Generated Blender script: \n{generated_script}")
            else:
                error_message = "no_content_error"
                raise ValueError(error_message)
            
            return generated_script, content
                
        except Exception as e:
            if not error_message:
                error_message = "script_extraction_error"
            return self.handle_error(e, error_message), content
            

    def process_query_embedding(self, user_query):
        """
        Create the query embedding from the user query using the specified embedding model.

        Args:
            user_query (str): The query from the user.

        Returns:
            np.array: The embedding vector for the user query.
        """
        query_embedding = np.array([self.client.embeddings.create(
            model=self.embedding_model,
            input=user_query
        ).data[0].embedding])
        return query_embedding



    def process_embeds(self, user_query:str, index_length:int) -> str:
        """
        Prepare the chatbot input by combining the user query with function metadata from multiple indexes.

        Args:
            user_query (str): The original query from the user.
            context_length (int): The number of top matches to return from each index.

        Returns:
            str: A formatted string combining metadata from both indexes and the user query.
        """
        # Create query embedding
        query_embedding = self.process_query_embedding(user_query)

        # Search the Blender Python (bpy) index
        bpy_metadata_info = self.search_index(
            self.bpy_index, self.bpy_meta, query_embedding, index_length, self.distance_th
        )

        # Search the manual index
        manual_metadata_info = self.search_index(
            self.manual_index, self.manual_meta, query_embedding, self.manual_embeds_length , self.distance_th
        )

        # Combine results from both searches
        if bpy_metadata_info or manual_metadata_info:
            RAG_context = f"""
            Following functions are found to be most matching with the last user query, however you are not limited by them:
            \nFOUND FUNCTIONS FROM BLENDER PYTHON API INDEX: \n{bpy_metadata_info}\n \n \n 
            FOUND EXPLANATIONS FROM BLENDER MANUAL INDEX: \n{manual_metadata_info}\n \n \n 
            """
        else:
            RAG_context = ""

        return RAG_context
    

    def search_index(self, index, meta_data, query_embedding, context_length, distance_threshold):
        """
        Perform the search on a given FAISS index and return formatted metadata.

        Args:
            index: The FAISS index to search.
            meta_data: The metadata corresponding to the index.
            query_embedding: The embedding vector of the user query.
            context_length (int): The number of top matches to return.
            distance_threshold (float): The threshold for considering a match.

        Returns:
            str: A formatted string with matching metadata from the index.
        """
        # Perform FAISS search on the given index
        D, I = index.search(query_embedding, context_length)
        matching_metadata = []

        # Filter results based on the distance threshold
        for distance, idx in zip(D[0], I[0]):
            if distance <= distance_threshold:
                # Retrieve metadata for the matching index
                meta = meta_data[idx]

                # Create formatted parameter list with descriptions
                parameters_info = []
                for param, param_desc in zip(meta.get('parameters', []), meta.get('parameter_descriptions', [])):
                    parameters_info.append(f"{param}: {param_desc}")

                # Join parameters with their descriptions
                formatted_parameters = "\n".join(parameters_info)

                # Format metadata info with signature, description, and parameters
                metadata_info = f"""
                Function: {meta.get('signature', 'N/A')}
                Description: {meta.get('description', 'N/A')}
                Parameters: {formatted_parameters}
                """
                matching_metadata.append(metadata_info)

        # Return combined formatted metadata
        if matching_metadata:
            return "\n".join(matching_metadata)
        else:
            return "None"

    def load_embeddings(self, embeddings_path: str, metadata_path: str):
        """
        Load embeddings and metadata from the specified files.

        Args:
            embeddings_path (str): Path to the .npy file containing the embeddings.
            metadata_path (str): Path to the .json file containing the metadata.

        Returns:
            Tuple[np.ndarray, List[Dict]]: A tuple containing the embeddings as a numpy array and metadata as a list of dictionaries.
        """
        # Load embeddings from the .npy file
        embeddings = None
        metadata = None
        try:
            #print('here')
            embeddings = np.load(embeddings_path)
            print(f"Successfully loaded embeddings from {embeddings_path}")
        except Exception as e:
            return self.handle_error(e,"embed_load_error"), metadata

        # Load metadata from the .json file
        try:
            #print('here2')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"Successfully loaded metadata from {metadata_path}")
        except Exception as e:
            return embeddings, self.handle_error(e,"meta_load_error")

        return embeddings, metadata

    def count_tokens(self, messages_to_send: List[ChatCompletionAssistantMessageParam |
                                                ChatCompletionUserMessageParam |
                                                ChatCompletionSystemMessageParam]) -> int:
        """
        Counts the number of tokens in the messages based on the initialized model.

        Args:
            messages_to_send (List): The list of message objects to count tokens for.

        Returns:
            int: The number of tokens in the prepared messages.
        """
        len_tokens = 0
        try:
            # Use a known encoding (e.g., gpt-3.5-turbo encoding for GPT-4)
            encoding = tiktoken.get_encoding("cl100k_base")  # 'cl100k_base' is used for GPT-4 and GPT-3.5 models

            # Join all message contents and count tokens
            message_contents = " ".join([msg['content'] for msg in messages_to_send])  # Extract content from message objects
            tokens = encoding.encode(message_contents)  # Encode the content using the tokenizer
            logging.info(f'Tokens in the message: {len(tokens)}')  # Debugging info: Print the token count
            if isinstance(tokens,list): 
                len_tokens = len(tokens)
            else:
                len_tokens = -1
            return len_tokens
        except Exception as e:
            return self.handle_error(e, "count_tokens_error")



################################ TODO: CREATE ERROR HANDLING FOR ERROR CONTEXTS

    def handle_error(self, e, context=""):
        """Handle and log common errors with a default template."""
        error_message = f"An error occurred: {str(e)}"
        detailed_traceback = traceback.format_exc()

        # Define specific handling for common errors
        if isinstance(e, FileNotFoundError):
            error_message += f"FileNotFoundError: The file was not found. {context}"
        elif isinstance(e, subprocess.CalledProcessError):
            return e
            #error_message += f"SubprocessError: The command failed. {context}"
        else:
            error_message = f"General error in {context}: {str(e)}"

        ##### save script error returns none for the interface_handler, this can be changed.
        if context == "save_script_error":
            return None 
        elif context == "count_tokens_error":
            return -2
        elif context == "meta_load_error":
            return None
        elif context == "embed_load_error":
            return None
        elif context == "save_script_error":
            return None
        
        # Log the error with details and traceback
        logging.error(f"{error_message}\n{detailed_traceback}")
        
        # Return a user-friendly error message
        return error_message




if __name__ == "__main__":
    
    generator = backend(0)
    validator = backend(2)

    # Example prompt to generate a Blender Python script
    prompt = "Don't halicunate Generate a Blender 4.4.2 Python script to create a red cube at the origin. On the right side of the cube create a donut and on the left side of the cube create a glass skyscraper. Make sure that the objects don't intersect with each other . Print the succesfull message only if you can verify all the objects are created and if not print the error message and the reason."
    #prompt = "run the following script: ```python\nimport bpy\n\n# Clear existing mesh objects\nbpy.ops.object.select_all(action='DESELECT')\nbpy.ops.object.select_by_type(type='MESH')\nbpy.ops.object.delete()\n\n# Add a Cube\nbpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))\n\n# Add a Light\nbpy.ops.object.light_add(type='POINT', location=(5, 5, 5))\n\n# Add a Camera\nbpy.ops.object.camera_add(location=(7, -7, 5))\nbpy.context.object.rotation_euler = (1.1, 0, 0.9)\n\n# Set the camera as the active camera\nbpy.context.scene.camera = bpy.context.object\n```"
    # Step 1: Send the prompt to OpenAI and get the generated script
    response = generator.send_query(prompt)
    generated_script = generator.response_handler(response)
    
    print('\n\n\n\n')
    print(type(generated_script))

    response_val =  validator.send_query(generated_script)
    generated_script = validator.response_handler(response_val)

    print('\n\n\n\n')
    print(type(generated_script))

    if generated_script:
        # Step 2: Save the generated script to a Python file
        script_name = validator.save_script(generated_script)

        if script_name:
            # Step 3: Run the saved script using Blender CLI and capture the output or errors
            result = validator.run_blender_script(script_name)

            # Output the result or error from running the Blender script
            print(result)
        else:
            print("Error saving the generated script.")
    else:
        print("Error generating the Blender script.")