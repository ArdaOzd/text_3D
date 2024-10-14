import gradio as gr
from backend import backend
from typing import List
import numpy as np
from PIL import Image
from conf import GEN_IMG_PATH


class ChatInterface:
    def __init__(self, backend):
        """Initialize the Chat Interface with a backend class."""
        self.backend = backend
        self.interface = gr.Blocks(fill_height=True)
        self.image = []

    def handle_input(self, input=None, history: List = []):
        output_message = gr.ChatMessage(role='assistant', content="")
        image_path = None  # Initialize the image output

        if isinstance(input, dict):
            # Process text input
            response, image_path = self.backend.interface_handler(input)
            print(response)
            output_message.content = str(f'''{str(response)}''')
            if image_path:
                self.image = self.handle_image(image_path)
                history.append( dict(role="assistant",
                        content=self.image))
            return output_message

        elif isinstance(input, tuple):
            # Process other types of input if necessary
            response, image_path = self.backend.interface_handler(input)
            output_message.content = str(response)
            if image_path:
                return output_message
            else:
                return output_message

        else:
            return output_message

    def handle_image(self, image_path = None):
        if image_path is None:
            return None
        try:
            with Image.open(image_path) as img:
                self.image = np.array(img)
            return self.image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def create_interface(self):
        """Create the Gradio interface."""
        with self.interface:
            gr.Markdown("## Blender Bot")

            # Create the Chatbot component
            chatbot = gr.Chatbot(
                type='messages',
                placeholder="<strong>Your Personal Yes-Man</strong><br>Ask Me Anything",
                height=800,
                max_height=1000,
                elem_id="dynamic-chatbox"
            )


            # Define the ChatInterface
            chatinterface = gr.ChatInterface(
                fn=self.handle_input,
                type='messages',
                chatbot= chatbot,
                examples=[
                    {"text": "Create donut and save as png"},
                    {"text": "Place a donut in the origin and rotate it."}
                ],
                multimodal=True
            )

            ''' 
            # Create the Image component to display the output image
            image_output = gr.Image(value=self.handle_image,
                                    label="Generated Image", 
                                    visible=True,
                                    inputs=self.image
                                    )
            
            # Set up the audio input if needed
            audio_input = gr.Microphone(
                sources="microphone",
                type="numpy",
                label="Record Audio",
                interactive=True
            )
            audio_btn = gr.Button("Submit Audio")

            # Handle audio input
            audio_btn.click(
                self.handle_input,
                inputs=audio_input,
                outputs=[chatinterface.chatbot, image_output]
            )'''

        # Add custom CSS for dynamic sizing of the chatbot and auto-scroll
        self.interface.css = """
            #dynamic-chatbox {
                overflow-y: auto;
                max-height: 80vh; /* Chatbox takes 80% of the viewport height */
                width: 100%;
                height: 100%;
            }

            #dynamic-chatbox .wrap {
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
                height: 100%;
            }
        """
    

    def launch(self):
        """Launch the Gradio UI."""
        self.create_interface()
        self.interface.launch()

# Example of how you can launch the Gradio UI
if __name__ == "__main__":
    # Create an instance of the backend
    generator = backend(0,True)

    # Create the chat interface
    chat_ui = ChatInterface(generator)

    # Launch the interface
    chat_ui.launch()



'''

build 2 towers make a bridge between them at 5 metres, create 10 people infront of the towers, create 2 cars 1 blue 1 red, make a pyramid. Don't intersect any of the object. Place a sun far away for diffuse light, render the scene. Place the sun to 50,50,50, make the sun really bright, and create environment lighting too,build 2 towers make a bridge between them at 5 metres, create 10 people infront of the towers, create 2 cars 1 blue 1 red, make a pyramid. Don't intersect any of the object. Place a sun far away for diffuse light, render the scene. Place the sun to 50,50,50, make the sun really bright, and create environment lighting too,build 2 towers make a bridge between them at 5 metres, create 10 people infront of the towers, create 2 cars 1 blue 1 red, make a pyramid. Don't intersect any of the object. Place a sun far away for diffuse light, render the scene. Place the sun to 50,50,50, make the sun really bright, and create environment lighting too
RATATATATATATATA, ciaoaicoiaaa


'''