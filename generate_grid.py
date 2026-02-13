
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Configuration
OUTPUT_WORK_DIR = "output"
OUTPUT_NORM_DIR = "output_normalized_balanced_norm"
GRID_OUTPUT_PATH = "reports/comparison_grid.jpg"

MODELS = ["gemini_flash", "gpt_image"]
PROMPTS = [
    "a_human_being_photorealistic",
    "a_person_photorealistic", 
    "an_individual_photorealistic",
    "someone_photorealistic"
]

SAMPLES_PER_GROUP = 5
IMG_SIZE = (256, 256)

def generate_grid(use_normalized=False):
    """
    Generates a grid of images:
    Rows: Prompts
    Cols: Models (5 samples each)
    """
    
    # Image source directory
    base_dir = OUTPUT_NORM_DIR if use_normalized else OUTPUT_WORK_DIR
    title_suffix = " (Normalized)" if use_normalized else " (Original)"
    
    # Create the figure
    # 4 rows, 10 columns
    # We'll use a slightly complex layout or just pure numpy concatenation
    
    # Let's use numpy concatenation for the image grid and cv2/PIL to save, 
    # but adding text is easier with matplotlib or PIL. 
    # Let's use matplotlib subplots.
    
    n_rows = len(PROMPTS)
    n_cols = len(MODELS) * SAMPLES_PER_GROUP # 10 columns
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    
    # Set headers
    # We want big headers for Models at the top
    # And row labels for Prompts on the left
    
    for row_idx, prompt in enumerate(PROMPTS):
        prompt_clean = prompt.replace("_photorealistic", "").replace("_", " ").title()
        
        # Add Row Label (Prompt) - simplified, just print it on the first plot or use fig text
        # We'll use the y-label of the first column
        axes[row_idx, 0].set_ylabel(prompt_clean, fontsize=12, rotation=90, labelpad=10)
        
        col_cursor = 0
        for model in MODELS:
            model_dir = os.path.join(base_dir, model, prompt)
            
            # Get images
            if os.path.exists(model_dir):
                files = [f for f in os.listdir(model_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                random.shuffle(files)
                selected_files = files[:SAMPLES_PER_GROUP]
            else:
                selected_files = []
            
            # Pad if not enough
            while len(selected_files) < SAMPLES_PER_GROUP:
                selected_files.append(None)
                
            for i, filename in enumerate(selected_files):
                ax = axes[row_idx, col_cursor]
                
                # Column titles (Model Names) only on first row
                if row_idx == 0 and i == 2: # Centered roughly
                    ax.set_title(model.replace("_", " ").upper(), fontsize=14, fontweight='bold')
                
                if filename:
                    img_path = os.path.join(model_dir, filename)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, IMG_SIZE)
                        ax.imshow(img)
                    else:
                        ax.text(0.5, 0.5, "Error", ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, "N/A", ha='center', va='center')
                
                ax.set_xticks([])
                ax.set_yticks([])
                
                col_cursor += 1

    # Main Title
    fig.suptitle(f"Random Samples by Prompt and Model{title_suffix}", fontsize=16, y=0.98)
    
    os.makedirs(os.path.dirname(GRID_OUTPUT_PATH), exist_ok=True)
    plt.savefig(GRID_OUTPUT_PATH, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Grid saved to {GRID_OUTPUT_PATH}")

if __name__ == "__main__":
    # Default to Original as per user preference discussion
    generate_grid(use_normalized=False)
