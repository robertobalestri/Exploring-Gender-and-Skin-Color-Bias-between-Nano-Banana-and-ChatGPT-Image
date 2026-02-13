import os
import requests
import base64
from PIL import Image
from io import BytesIO
from azure.identity import DefaultAzureCredential

class AzureGenerator:
    def __init__(self):
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.api_version = os.getenv("OPENAI_API_VERSION")
        self.api_key = os.getenv("AZURE_API_KEY")
        
        self.credential = None
        if not self.api_key:
            try:
                self.credential = DefaultAzureCredential()
            except Exception as e:
                print(f"Error initializing Azure credential: {e}")

    def get_token(self):
        if self.credential:
            token_response = self.credential.get_token("https://cognitiveservices.azure.com/.default")
            return token_response.token
        return None

    def decode_and_save_image(self, b64_data, output_filename):
        try:
            image = Image.open(BytesIO(base64.b64decode(b64_data)))
            image.save(output_filename)
            print(f"Azure Image saved to: '{output_filename}'")
        except Exception as e:
            print(f"Error saving image: {e}")

    def generate_image(self, prompt, output_file="azure_image.png"):
        print(f"Generating image with Azure (deployment: {self.deployment}) for prompt: '{prompt}'")
        
        headers = {
            'Content-Type': 'application/json',
        }

        if self.api_key:
            headers['api-key'] = self.api_key
        else:
            try:
                token = self.get_token()
                headers['Authorization'] = 'Bearer ' + token
            except Exception as e:
                print(f"Failed to get Azure token: {e}")
                return None

        base_path = f'openai/deployments/{self.deployment}/images'
        params = f'?api-version={self.api_version}'

        if not self.endpoint:
             print("Error: AZURE_OPENAI_ENDPOINT not set.")
             return None

        generation_url = f"{self.endpoint.rstrip('/')}/{base_path}/generations{params}"
        
        generation_body = {
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024",
            "quality": "medium", 
            "output_format": "png"
        }
        
        try:
            response = requests.post(
                generation_url,
                headers=headers,
                json=generation_body
            )
            
            if response.status_code != 200:
                print(f"Azure API Error: {response.status_code} - {response.text}")
                return None
                
            generation_response = response.json()
            
            if 'data' in generation_response and len(generation_response['data']) > 0:
                item = generation_response['data'][0]
                if 'b64_json' in item:
                    self.decode_and_save_image(item['b64_json'], output_file)
                    return output_file
                elif 'url' in item:
                    print("Received URL instead of b64_json. Downloading...")
                    img_resp = requests.get(item['url'])
                    with open(output_file, 'wb') as f:
                        f.write(img_resp.content)
                    print(f"Saved to {output_file}")
                    return output_file
            
            print(f"Unexpected response format: {generation_response}")
            return None

        except Exception as e:
            print(f"Error generating image with Azure: {e}")
            return None


if __name__ == "__main__":
    gen = AzureGenerator()
    gen.generate_image("A futuristic city")
