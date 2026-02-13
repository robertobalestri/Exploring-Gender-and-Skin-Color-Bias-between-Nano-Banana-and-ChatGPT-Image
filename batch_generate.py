import os
import concurrent.futures
import time
import re
from dotenv import load_dotenv
from prompts import PROMPTS
from generators.gemini_gen import GeminiFlashGenerator, GeminiProGenerator
from generators.azure_gen import AzureGenerator

load_dotenv()

from settings import BATCH_SIZE

TARGET_COUNT = BATCH_SIZE
OUTPUT_BASE = "output"

def sanitize_filename(name):
    """
    Sanitize the prompt to be safe for directory names.
    Replaces non-alphanumeric characters with underscores.
    """
    # Keep only alphanumeric and spaces, then replace spaces with underscores
    clean = re.sub(r'[^a-zA-Z0-9\s]', '', name)
    return clean.replace(' ', '_').lower()

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_batch(generator_class, model_name, subfolder_name):
    try:
        generator = generator_class()
    except Exception as e:
        print(f"Failed to initialize {model_name}: {e}")
        return

    for prompt in PROMPTS:
        safe_prompt = sanitize_filename(prompt)
        output_dir = os.path.join(OUTPUT_BASE, subfolder_name, safe_prompt)
        ensure_directory(output_dir)
        
        print(f"\nProcessing Prompt: '{prompt}' for {model_name}")
        print(f"Output Directory: {output_dir}")
        
        # Count existing valid files
        # We assume files are named 1.png, 2.png etc.
        # To be robust, we can just find the highest number or just count files.
        # User asked: "count how many images have been already produced... and continue from there."
        # Simple counting of files matching *.png is easiest.
        
        # Determine existing files
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        current_count = len(existing_files)
        
        if current_count >= TARGET_COUNT:
            print(f"  Skipping: Already have {current_count} images.")
            continue
            
        print(f"  Starting from image #{current_count + 1}")
        
        # Start consecutive generation
        # We use a running index based on current count. 
        # CAUTION: If user deletes file #2, strictly appending might overwrite or create gaps. 
        # But "continue from there" usually implies appending.
        # Let's use a while loop until we hit target count.
        
        i = current_count + 1
        failures_in_a_row = 0
        
        while i <= TARGET_COUNT:
            output_filename = os.path.join(output_dir, f"{i}.png")
            
            # Double check if file exists (e.g. if we had gaps and we are filling sequentially)
            if os.path.exists(output_filename):
                print(f"  Skipping existing file: {i}.png")
                i += 1
                continue
                
            try:
                print(f"  Generating {i}/{TARGET_COUNT}...")
                result = generator.generate_image(prompt, output_filename)
                
                if result:
                    print(f"    Success: {i}.png")
                    failures_in_a_row = 0
                    i += 1
                else:
                    print(f"    Failed to generate image #{i}.")
                    failures_in_a_row += 1
                    time.sleep(0.5) # Short backoff
                    
                if failures_in_a_row >= 5:
                    print("    CRITICAL: 5 consecutive failures. Pausing for 10 seconds.")
                    time.sleep(10)
                    failures_in_a_row = 0
                    
            except Exception as e:
                print(f"    Error during generation loop: {e}")
                failures_in_a_row += 1
                time.sleep(1)
                
            # Rate limiting removed as per user request to speed up
            # time.sleep(1) 

def main():
    print("Starting Batch Generation...")
    print(f"Target: {TARGET_COUNT} images per prompt per model.")
    
    # Prepare tasks for parallel execution
    # Each task is a tuple: (generator_class, model_name, subfolder_name)
    tasks = [
        #(GeminiFlashGenerator, "GeminiFlash", "gemini_flash"),
        (GeminiProGenerator, "GeminiPro", "gemini_pro"),
        #(AzureGenerator, "Azure", "gpt_image")
    ]

    print(f"\nScanning and starting parallel generation for {len(tasks)} models...")
    print("Output output will be interleaved.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = []
        for task in tasks:
            # Unpack task args
            gen_class, name, folder = task
            print(f"Submitting task for {name}...")
            futures.append(executor.submit(generate_batch, gen_class, name, folder))
        
        # Wait for all to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A batch generation task failed: {e}")
    
    print("\nBatch Generation Completed.")

if __name__ == "__main__":
    main()
