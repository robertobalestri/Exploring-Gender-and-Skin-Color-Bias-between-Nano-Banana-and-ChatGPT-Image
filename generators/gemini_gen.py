from google import genai
from google.genai import types
import base64
import os

from settings import TEMPERATURE, TOP_P

class GeminiGenerator:
    def __init__(self, model_name="gemini-2.5-flash-image"):
        self.model_name = model_name
        self.api_key = os.environ.get("GOOGLE_CLOUD_API_KEY")
        if not self.api_key:
            print("Warning: GOOGLE_CLOUD_API_KEY environment variable not set.")
        
        # Initialize the client. 
        self.client = genai.Client(
            api_key=self.api_key
        )

    def generate_image(self, prompt, output_file="gemini_image.png"):
        print(f"Generating image with Gemini ({self.model_name}) for prompt: '{prompt}'")
        model = self.model_name
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt)
                ]
            )
        ]

        if self.model_name == "gemini-3-pro-image-preview":
            generate_content_config = types.GenerateContentConfig(
                temperature = TEMPERATURE,
                top_p = TOP_P,
                max_output_tokens = 8192,
                response_modalities = ["IMAGE"],
                system_instruction="",
                image_config=types.ImageConfig(
                    aspect_ratio="1:1",
                    image_size="1K"
                ),
                safety_settings = [types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                )],
            )
        else:
            # Simple config for Flash or other models
            generate_content_config = types.GenerateContentConfig(
                temperature = TEMPERATURE,
                top_p = TOP_P,
                max_output_tokens = 8192,
                response_modalities = ["IMAGE"],
                # system_instruction might be supported but safer to omit if not needed
                # image_config NOT supported by Flash
                safety_settings = [types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE"
                )],
            )

        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            
            if not response.candidates:
                print(f"No candidates returned. Prompt feedback: {response.prompt_feedback}")
                return None

            candidate = response.candidates[0]
            if not candidate.content:
                print(f"Blocked. Finish Reason: {candidate.finish_reason}")
                return None
                
            for part in candidate.content.parts:
                if part.text:
                    print(f"Gemini Text Response: {part.text}")
                
                if part.inline_data:
                    # The SDK usually returns raw bytes in inline_data.data
                    img_data = part.inline_data.data
                    
                    # If it happens to be a base64 string (legacy), we might need to decode.
                    # But based on 1.9MB size in debug, it's likely raw bytes.
                    # We can check if it's bytes.
                    if isinstance(img_data, str):
                         # If string, assumption is base64
                         img_data = base64.b64decode(img_data)
                    
                    with open(output_file, "wb") as f:
                        f.write(img_data)
                    print(f"Gemini Image saved to {output_file}")
                    return output_file
            
            print("No image found in Gemini response.")
            return None

        except Exception as e:
            print(f"Error generating image with Gemini: {e}")
            with open("error_log.txt", "w") as f:
                f.write(str(e))
            return None

# Subclasses for specific models
class GeminiFlashGenerator(GeminiGenerator):
    def __init__(self):
        super().__init__(model_name="gemini-2.5-flash-image")

class GeminiProGenerator(GeminiGenerator):
    def __init__(self):
        super().__init__(model_name="gemini-3-pro-image-preview")