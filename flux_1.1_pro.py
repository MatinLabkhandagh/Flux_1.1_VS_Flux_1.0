import os
import gradio as gr
from together import Together
import base64
from PIL import Image
import io

def generate_image(api_key, prompt):
    # Use the provided API key or fall back to the environment variable
    api_key = api_key or os.environ.get('TOGETHER_API_KEY')
    
    if not api_key:
        return None, "Please provide a valid Together API key"
    
    try:
        # Initialize the Together client with the API key
        client = Together(api_key=api_key)
        
        response = client.images.generate(
            prompt=prompt,
            model="black-forest-labs/FLUX.1.1-pro",
            width=1024,
            height=768,
            steps=1,
            n=1,
            response_format="b64_json"
        )
        
        # Decode the base64 image
        image_data = base64.b64decode(response.data[0].b64_json)
        image = Image.open(io.BytesIO(image_data))
        
        return image, "Image generated successfully!"
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(type="password", label="Together API Key"),
        gr.Textbox(lines=3, placeholder="Enter your image prompt here...", label="Prompt")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="Status")
    ],
    title="Image Generation with FLUX.1.1-pro",
    description="Generate images using the FLUX.1.1-pro model via the Together API. You can provide your API key here"
)

# Launch the interface
iface.launch()