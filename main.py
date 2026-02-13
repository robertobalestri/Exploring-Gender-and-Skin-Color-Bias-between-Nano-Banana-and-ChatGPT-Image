import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from generators.gemini_gen import GeminiGenerator
from generators.azure_gen import AzureGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate images using Gemini and Azure OpenAI.")
    parser.add_argument("prompt", nargs="?", default="A photorealistic shot of a futuristic bicycle", help="The prompt for image generation")
    parser.add_argument("--provider", choices=["all", "gemini", "azure"], default="all", help="Which provider to use")
    
    args = parser.parse_args()
    
    print(f"Starting generation for prompt: '{args.prompt}'")
    
    # Gemini
    if args.provider in ["all", "gemini"]:
        print("\n--- Gemini Generation ---")
        if not os.environ.get("GOOGLE_CLOUD_API_KEY"):
             print("SKIPPING Gemini: GOOGLE_CLOUD_API_KEY not set.")
        else:
            gemini = GeminiGenerator()
            gemini.generate_image(args.prompt, "gemini_result.png")

    # Azure
    if args.provider in ["all", "azure"]:
        print("\n--- Azure OpenAI Generation ---")
        # specific check for Azure vars could be added, but the class handles it gracefully
        azure = AzureGenerator()
        azure.generate_image(args.prompt, "azure_result.png")

    print("\nDone.")

if __name__ == "__main__":
    main()
