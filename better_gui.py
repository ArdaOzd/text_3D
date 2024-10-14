import gradio as gr
from backend import backend
from typing import List
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData
import os
import asyncio

class ChatInterface:
    def __init__(self, backend):
        """Initialize the Chat Interface with a backend class."""
        self.backend = backend
        self.interface = gr.Blocks()

    async def handle_input(self, user_input_text, history):
        """
        Handle the input from the user and update the conversation history.
        """
        if history is None:
            history = []

        # Prepare user message
        user_msg = {
            "text": user_input_text,
            "files": []
        }

        # Display the user's message immediately
        history = history + [[user_msg, None]]
        yield gr.update(value=history), history, gr.update(value="")

        # Prepare input dict for backend
        input_dict = {'text': user_input_text}

        # Process the assistant's response asynchronously
        response, image_path = await asyncio.to_thread(self.backend.interface_handler, input_dict)
        print(response)

        # Prepare assistant message
        if image_path and os.path.exists(image_path):
            # Use FileData to include the image
            assistant_msg = {
                "text": response,
                "files": [{"file": FileData(path=image_path)}]
            }
        else:
            assistant_msg = {
                "text": response,
                "files": []
            }

        # Update the last conversation turn with the assistant's response
        history[-1][1] = assistant_msg

        # Return the updated history
        yield gr.update(value=history), history, gr.update(value="")
    
    def create_interface(self):
        """Create the Gradio interface using MultimodalChatbot."""
        with self.interface:
            gr.Markdown("## Blender Bot")


            # Add custom CSS to align user messages to the right
            self.interface.css = """
            /* Align user messages to the right */
            #chatbot .user {
                text-align: right
            }

            """ 

            # MultimodalChatbot component
            chatbot = MultimodalChatbot(height=800)

            # Textbox for user input
            user_input = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False
            )

            # State to hold conversation history
            state = gr.State([])

            # Define the interaction
            async def on_submit(user_input_text, history):
                # Call the async handle_input function
                async for chatbot_update, state_update, input_update in self.handle_input(user_input_text, history):
                    yield chatbot_update, state_update, input_update

            # When the submit button is clicked or Enter is pressed, handle input and update outputs
            submit_btn = gr.Button("Submit")

            # Handle button click
            submit_btn.click(
                on_submit,
                inputs=[user_input, state],
                outputs=[chatbot, state, user_input]
            )

            # Handle Enter key press
            user_input.submit(
                on_submit,
                inputs=[user_input, state],
                outputs=[chatbot, state, user_input]
            )

    def launch(self):
        """Launch the Gradio UI."""
        self.create_interface()
        self.interface.launch()

if __name__ == "__main__":
    # Create an instance of the backend
    generator = backend(0, True)

    # Create the chat interface
    chat_ui = ChatInterface(generator)

    # Launch the interface
    chat_ui.launch()
