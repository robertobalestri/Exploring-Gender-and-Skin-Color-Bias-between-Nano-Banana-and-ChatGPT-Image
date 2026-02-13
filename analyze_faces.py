import os
import json
import cv2
import pandas as pd
import numpy as np
from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage import color as skcolor  # For CIEDE2000

# Configuration
OUTPUT_WORK_DIR = "output"
MODEL_PATH = "weights/face_landmarker.task"

# Result CSVs
RESULTS_ORIGINAL_CSV = "analysis_results_original.csv"
RESULTS_BALANCED_CSV = "analysis_results_balanced_norm.csv"

# Output Folders for Normalized Images
OUTPUT_NORM_ORIGINAL_DIR = "output_normalized_original"
OUTPUT_NORM_BALANCED_DIR = "output_normalized_balanced_norm"

# Output Folders for Face Crops
OUTPUT_FACES_ORIGINAL_DIR = "output_faces_original"
OUTPUT_FACES_BALANCED_DIR = "output_faces_balanced_norm"

# --- LANDMARK GROUPS ---
try:
    with open('landmarks_groups.json', 'r') as f:
        LANDMARK_GROUPS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load landmarks_groups.json: {e}")
    LANDMARK_GROUPS = {}

# --- MEDIAPIPE CONSTANTS ---
# Landmark indices for precise skin extraction are handled in extract_precise_mask via landmarks_groups.json

# Monk Skin Tone (MST) Scale Base Colors
MST_COLORS = [
    "#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
    "#a07e56", "#825c43", "#604134", "#3a312a", "#292420"
]

# PERLA Skin Tone Scale Colors
PERLA_COLORS = [
    "#373028", "#422811", "#513B2E", "#6F503C", 
    "#81654F", "#9D7A54", "#BEA07E", "#E5C8A6", 
    "#E7C1B8", "#F3DAD6", "#FBF2F3"
]

# Fitzpatrick Skin Type (FST) Scale - RGB values (Type I-VI)
FST_PALETTE_RGB = np.array([
    [255, 224, 196],  # FST I - Very light/pale
    [241, 194, 168],  # FST II - Light
    [224, 172, 141],  # FST III - Medium light
    [198, 134, 103],  # FST IV - Medium
    [141, 85, 62],    # FST V - Medium dark
    [75, 47, 35],     # FST VI - Dark
], dtype=np.uint8)

# Pre-convert FST palette to standard CIELAB for CIEDE2000 matching
FST_PALETTE_LAB = skcolor.rgb2lab(FST_PALETTE_RGB.reshape(1, -1, 3).astype(np.float64) / 255.0).reshape(-1, 3)

# Pre-convert palettes to LAB for Delta-E matching
# IMPORTANT: skimage uses standard L*a*b* (L: 0-100, a/b: -128 to +127)
# We must convert from OpenCV's 0-255 scaled LAB to standard LAB for CIEDE2000
def _hex_list_to_lab_standard(hex_list):
    """Convert list of hex colors to standard CIELAB array (L: 0-100)."""
    rgb_array = np.array([[int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] for h in hex_list], dtype=np.uint8)
    # skimage expects RGB 0-1 float, outputs L: 0-100, a/b: -128 to 127
    rgb_float = rgb_array.astype(np.float64) / 255.0
    return skcolor.rgb2lab(rgb_float.reshape(1, -1, 3)).reshape(-1, 3)

MST_PALETTE_LAB = _hex_list_to_lab_standard(MST_COLORS)
PERLA_PALETTE_LAB = _hex_list_to_lab_standard(PERLA_COLORS)

def hex_to_rgb(hex_color):
    """Converts hex string (e.g., '#ffffff') to rgb tuple (255, 255, 255)."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def bgr_to_hex(bgr):
    """Convert BGR tuple to hex string."""
    return "#{:02x}{:02x}{:02x}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))

def get_mst_index(target_hex):
    """Returns 1-10 MST index using CIEDE2000 (industry standard)."""
    target_rgb = np.array(hex_to_rgb(target_hex), dtype=np.uint8).reshape(1, 1, 3)
    target_lab = skcolor.rgb2lab(target_rgb.astype(np.float64) / 255.0).reshape(1, 3)
    # Calculate CIEDE2000 distance to each palette color
    distances = np.array([skcolor.deltaE_ciede2000(target_lab, p.reshape(1, 3))[0] for p in MST_PALETTE_LAB])
    return int(np.argmin(distances)) + 1

def get_perla_index(target_hex):
    """Returns 1-11 PERLA index using CIEDE2000."""
    target_rgb = np.array(hex_to_rgb(target_hex), dtype=np.uint8).reshape(1, 1, 3)
    target_lab = skcolor.rgb2lab(target_rgb.astype(np.float64) / 255.0).reshape(1, 3)
    distances = np.array([skcolor.deltaE_ciede2000(target_lab, p.reshape(1, 3))[0] for p in PERLA_PALETTE_LAB])
    return int(np.argmin(distances)) + 1

def get_nearest_mst_color(target_hex):
    """Finds the nearest MST color using Delta-E in LAB color space."""
    idx = get_mst_index(target_hex) - 1
    return MST_COLORS[idx]

def get_nearest_perla_color(target_hex):
    """Finds the nearest PERLA color using Delta-E in LAB color space."""
    idx = get_perla_index(target_hex) - 1
    return PERLA_COLORS[idx]

def run_aida_kmeans(pixels_bgr, k=4, threshold=0.36):
    """
    AIDA Method: K-Means clustering in LAB color space with palette matching.
    
    Uses weighted mean of clusters that cumulatively represent at least 50% of pixels.
    
    Returns dict with:
        - 'skin_hex': representative skin color hex
        - 'mst_index': 1-10 MST index
        - 'mst_hex': nearest MST color hex
        - 'perla_index': 1-11 PERLA index
        - 'perla_hex': nearest PERLA color hex
        - 'fst_type': 1-6 Fitzpatrick Type
        - 'delta_e': Delta-E distance to FST
        - 'n_clusters': number of clusters used to reach 50% threshold
    """
    if len(pixels_bgr) < k:
        return None
    
    # Convert BGR -> RGB -> LAB
    pixels_rgb = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
    pixels_lab = cv2.cvtColor(pixels_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    
    # NEW: Trim outliers by luminance within the mask (Ignore bottom 10% and top 5% pixels)
    # This prevents deep shadows and specular highlights from skewing the final average.
    L_vals = pixels_lab[:, 0]
    p_low = np.percentile(L_vals, 10)
    p_high = np.percentile(L_vals, 95)
    trimmed_mask = (L_vals >= p_low) & (L_vals <= p_high)
    trimmed_pixels = pixels_lab[trimmed_mask]
    
    if len(trimmed_pixels) < k:
        data = np.float32(pixels_lab)
    else:
        data = np.float32(trimmed_pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    try:
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    except:
        return None
    
    # Calculate cluster proportions and sort by size
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)
    proportions = counts / total_pixels
    sorted_indices = np.argsort(proportions)[::-1]  # Descending order
    
    # Accumulate clusters until we reach 50% threshold
    cumulative = 0.0
    selected_clusters = []
    selected_weights = []
    
    for idx in sorted_indices:
        selected_clusters.append(idx)
        selected_weights.append(proportions[idx])
        cumulative += proportions[idx]
        if cumulative >= threshold:
            break
    
    n_clusters = len(selected_clusters)
    
    # IMPROVED: Weighted average of the dominant clusters AFTER luminance trimming.
    # This is more robust than "Max L" or a simple average of the whole mask.
    # NOTE: At this point, centers are in OpenCV LAB (L: 0-255). We need to convert to standard LAB (L: 0-100).
    weights = np.array(selected_weights)
    weights = weights / weights.sum()
    final_lab_cv = np.zeros(3, dtype=np.float32)
    for i, cluster_idx in enumerate(selected_clusters):
        final_lab_cv += weights[i] * centers[cluster_idx]
    
    # Convert OpenCV LAB (L: 0-255) to standard CIELAB (L: 0-100) for CIEDE2000
    final_lab = np.array([final_lab_cv[0] * 100 / 255, final_lab_cv[1] - 128, final_lab_cv[2] - 128])
    
    # Match to FST palette using CIEDE2000 (industry standard)
    distances = np.array([skcolor.deltaE_ciede2000(final_lab.reshape(1, 3), p.reshape(1, 3))[0] for p in FST_PALETTE_LAB])
    fst_idx = np.argmin(distances)
    fst_type = fst_idx + 1  # FST I-VI (1-indexed)
    delta_e = distances[fst_idx]
    
    # Convert back to BGR/HEX (use OpenCV LAB for cv2 conversion)
    weighted_rgb = cv2.cvtColor(np.uint8([[final_lab_cv]]), cv2.COLOR_LAB2RGB)[0][0]
    weighted_bgr = weighted_rgb[::-1]  # RGB -> BGR
    skin_hex = bgr_to_hex(weighted_bgr)
    
    # Get MST and PERLA matches
    mst_hex = get_nearest_mst_color(skin_hex)
    perla_hex = get_nearest_perla_color(skin_hex)
    
    return {
        'skin_hex': skin_hex,
        'mst_index': get_mst_index(skin_hex),
        'mst_hex': mst_hex,
        'perla_index': get_perla_index(skin_hex),
        'perla_hex': perla_hex,
        'fst_type': fst_type,
        'delta_e': round(delta_e, 2),
        'n_clusters': n_clusters,
    }

def white_balance(img):
    """Simple Gray World white balancing."""
    result = img.copy()
    avg_b = np.average(result[:, :, 0])
    avg_g = np.average(result[:, :, 1])
    avg_r = np.average(result[:, :, 2])
    if avg_b == 0: avg_b = 1
    if avg_g == 0: avg_g = 1
    if avg_r == 0: avg_r = 1
    avg = (avg_b + avg_g + avg_r) / 3
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg / avg_b), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg / avg_g), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg / avg_r), 0, 255)
    return result

def robust_white_balance(img):
    """Robust White Patch normalization (Percentile-based)."""
    result = img.copy()
    for i in range(3):
        v_max = np.percentile(result[:,:,i], 98) 
        if v_max == 0: v_max = 255
        scale = 255.0 / v_max
        result[:,:,i] = np.clip(result[:,:,i] * scale, 0, 255)
    return result

def histogram_normalization(img):
    """CLAHE on L channel of LAB color space - reduced aggression."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Reduced clipLimit from 2.0 to 1.2 to preserve more global skin tone differences
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result

# --- MEDIAPIPE HELPERS ---

def get_landmark_points(face_landmarks, indices, img_w, img_h):
    points = []
    for idx in indices:
        lm = face_landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        points.append([x, y])
    return np.array(points)

    h, w = image_bgr.shape[:2]
    
    # Get face oval points
    oval_pts = get_landmark_points(face_landmarks, FACE_OVAL, w, h)
    if len(oval_pts) < 3:
        return np.zeros((h, w), dtype=np.uint8)
    
    # Create base mask from face oval
    mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(oval_pts)
    cv2.fillConvexPoly(mask, hull, 255)
    
    # Function to subtract a region
    def subtract_region(indices):
        if len(indices) < 3:
            return
        region_pts = get_landmark_points(face_landmarks, indices, w, h)
        region_hull = cv2.convexHull(region_pts)
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(temp_mask, region_hull, 255)
        temp_mask = cv2.dilate(temp_mask, np.ones((5,5), np.uint8), iterations=2)
        nonlocal mask
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(temp_mask))
    
    # Subtract exclusion zones
    subtract_region(LEFT_EYE)
    subtract_region(RIGHT_EYE)
    subtract_region(LEFT_EYEBROW)
    subtract_region(RIGHT_EYEBROW)
    subtract_region(LIPS_OUTER)
    
    # Cut off at philtrum level (landmark 164)
    philtrum_y = int(face_landmarks[164].y * h)
    mask[philtrum_y:, :] = 0
    
    return mask

def extract_precise_mask(img_bgr, face_landmarks):
    """
    New Precise Method: 
    1. Convex Hull of (Include - Ignore) points.
    2. Subtract Convex Hulls of Exclude* groups.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Constants for Tapering
    FOREHEAD_LANDMARKS = [10, 338, 297, 151, 67, 109]
    
    # 1. Base Mask (Include - Ignore)
    include_set = set(LANDMARK_GROUPS.get('Include', []))
    ignore_set = set(LANDMARK_GROUPS.get('Ignore', []))
    base_indices = list(include_set - ignore_set)
    
    if not base_indices:
        return mask
        
    base_pts = get_landmark_points(face_landmarks, base_indices, w, h)
    if len(base_pts) >= 3:
        hull = cv2.convexHull(base_pts)
        cv2.fillConvexPoly(mask, hull, 255)
        
        # --- Tapered Forehead Expansion ---
        # Ref Height: Dist from Nose Tip (1) to Forehead Center (10)
        p10 = get_landmark_points(face_landmarks, [10], w, h)[0]
        p1 = get_landmark_points(face_landmarks, [1], w, h)[0]
        ref_h = abs(p1[1] - p10[1])
        max_offset = 0.2 * ref_h
        
        f_pts = get_landmark_points(face_landmarks, FOREHEAD_LANDMARKS, w, h)
        if len(f_pts) >= 1:
            cx = p10[0]
            # Find max distance from center x to determine tapering width
            max_dx = max([abs(p[0] - cx) for p in f_pts]) if len(f_pts) > 1 else 1.0
            if max_dx == 0: max_dx = 1.0
            
            expanded_pts = []
            for p in f_pts:
                dx = abs(p[0] - cx)
                # Curved (Elliptical) tapering for a convex shape
                normalized_dx = dx / max_dx
                weight = np.sqrt(max(0, 1.0 - normalized_dx**2))
                offset = int(max_offset * weight)
                expanded_pts.append([p[0], max(0, p[1] - offset)])
            
            # Add expanded points to mask
            all_f_pts = np.vstack([f_pts, np.array(expanded_pts)])
            f_hull = cv2.convexHull(all_f_pts.astype(np.int32))
            cv2.fillConvexPoly(mask, f_hull, 255)
            
    # 2. Subtract Exclude Regions
    for key, indices in LANDMARK_GROUPS.items():
        if key.startswith("Exclude"):
            excl_pts = get_landmark_points(face_landmarks, indices, w, h)
            if len(excl_pts) >= 3:
                excl_hull = cv2.convexHull(excl_pts)
                cv2.fillConvexPoly(mask, excl_hull, 0)
                
    return mask

def extract_skin_tone_mediapipe(image_bgr, detector):
    """
    Extracts skin tone using ONLY the Precise Mask (landmark-based anatomical regions).
    
    Returns: (results_dict, visualization_image, success_bool)
    """
    h, w = image_bgr.shape[:2]
    viz_image = image_bgr.copy()
    
    try:
        rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        detection = detector.detect(mp_image)
        
        if not detection.face_landmarks:
            return None, viz_image, False
            
        face_landmarks = detection.face_landmarks[0]
        
        # --- Precise Mask ---
        precise_mask = extract_precise_mask(image_bgr, face_landmarks)

        # --- AIDA Analysis ---
        pixels_pr = image_bgr[precise_mask == 255]
        analysis_result = run_aida_kmeans(pixels_pr, k=4) if len(pixels_pr) >= 5 else None
        
        results = {'precise': analysis_result}
        
        # Visualization
        combined_viz = np.zeros((h, w, 3), dtype=np.uint8)
        combined_viz[precise_mask > 0] = [255, 100, 255]   # Magenta for precise
        
        masked_viz = cv2.addWeighted(image_bgr, 0.6, combined_viz, 0.4, 0)
        
        if analysis_result is None:
            return None, viz_image, False
        
        return results, masked_viz, True
        
    except Exception as e:
        return None, viz_image, False

def get_dominant_color_kmeans(pixels, k=5):
    """Simple K-Means dominant color extraction."""
    if isinstance(pixels, np.ndarray) and pixels.ndim == 3:
        # Flatten if full image passed
        pixels = pixels.reshape(-1, 3)
        
    data = np.float32(pixels)
    if len(data) < k: return "#000000"
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        unique_labels, counts = np.unique(labels, return_counts=True)
        dominant_label = unique_labels[np.argmax(counts)]
        dominant_color = centers[dominant_label]
        return bgr_to_hex(dominant_color)
    except:
        return "#000000"

# --- GLOBAL SETTINGS ---
CSV_COLUMNS = [
    "File Path", "Model", "Prompt", "Filename", "Age", "Gender", "Race", "Emotion",
    # Precise Mask AIDA Results
    "P_Skin_Hex", "P_MST_Index", "P_MST_Hex", "P_PERLA_Index", "P_PERLA_Hex", "P_FST_Type", "P_FST_DeltaE", "P_N_Clusters"
]

# --- MAIN LOGIC ---

def process_analysis_pass(norm_img_path, original_file_path, model_name, prompt_name, 
                          face_output_dir, csv_file, img_norm_bgr, detector, pass_name="UNKNOWN"):
    try:
        # 1. Detect and Align Face (DeepFace)
        faces = DeepFace.extract_faces(
            img_path=norm_img_path, 
            detector_backend='retinaface', 
            align=True,
            enforce_detection=True
        )
        
        if not faces: return

        # Process largest face
        faces = sorted(faces, key=lambda x: x['facial_area']['w'] * x['facial_area']['h'], reverse=True)
        face_obj = faces[0]
        
        # DeepFace returns face in RGB 0-1, convert to BGR 0-255
        aligned_face_norm = face_obj['face']
        aligned_face_bgr = cv2.cvtColor((aligned_face_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 2. Skin Tone Extraction (All Methods)
        results, viz_image, success = extract_skin_tone_mediapipe(aligned_face_bgr, detector)
        
        # Get individual result sets

        
        # Default empty result for failed extraction
        empty = {'skin_hex': None, 'mst_index': None, 'mst_hex': None, 'perla_index': None, 'perla_hex': None, 'fst_type': None, 'delta_e': None, 'n_clusters': None}
        
        if not success or results is None:
            # Fallback to a simple ellipse mask if Mediapipe fails
            h, w = aligned_face_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, int(h*0.45)), (int(w*0.3), int(h*0.2)), 0, 0, 360, 255, -1)
            pixels = aligned_face_bgr[mask > 0]
            fallback = run_aida_kmeans(pixels, k=4) if len(pixels) > 0 else None
            pr = fallback if fallback else empty
            viz_image = aligned_face_bgr
        else:
            pr = results.get('precise') or empty

        # Save Visualization
        rel_path = os.path.relpath(original_file_path, OUTPUT_WORK_DIR)
        face_output_path = os.path.join(face_output_dir, rel_path)
        os.makedirs(os.path.dirname(face_output_path), exist_ok=True)
        cv2.imwrite(face_output_path, viz_image)
            
        # 3. Attributes Analysis (DeepFace)
        objs = DeepFace.analyze(
            img_path=aligned_face_bgr,
            actions=['age', 'gender', 'race', 'emotion'],
            enforce_detection=False, 
            detector_backend='skip', 
            align=False, 
            silent=True
        )
        
        if not objs: return
        face_data = objs[0]
        
        age = face_data.get('age')
        gender_preds = face_data.get('gender')
        gender = max(gender_preds, key=gender_preds.get) if isinstance(gender_preds, dict) else face_data.get('dominant_gender')
        race = face_data.get('dominant_race')
        emotion = face_data.get('dominant_emotion')

        result_row = {
            "File Path": original_file_path,  # Renamed from Original_File for CSV consistency
            "Model": model_name,
            "Prompt": prompt_name,
            "Pass": pass_name,
            "Filename": os.path.basename(original_file_path),
            "Age": age,
            "Gender": gender,
            "Race": race,
            "Emotion": emotion,
            # Precise Mask AIDA Results
            "P_Skin_Hex": pr['skin_hex'],
            "P_MST_Index": pr['mst_index'],
            "P_MST_Hex": pr['mst_hex'],
            "P_PERLA_Index": pr['perla_index'],
            "P_PERLA_Hex": pr['perla_hex'],
            "P_FST_Type": pr['fst_type'],
            "P_FST_DeltaE": pr['delta_e'],
            "P_N_Clusters": pr['n_clusters'],
        }
        
        print(f"    [{pass_name}]: Age={age}, Gender={gender}, P_MST={pr['mst_index']}")
        
        # Ensure exact column order matching header
        df = pd.DataFrame([result_row], columns=CSV_COLUMNS)
        
        if not os.path.exists(csv_file):
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode='a', header=False, index=False)
        print(f"    -> Written to {os.path.basename(csv_file)}")

    except Exception as e:
        print(f"    Error in {pass_name} pass: {e}")
        import traceback
        traceback.print_exc()
        pass

def analyze_images():
    print(f"Initializing MediaPipe FaceLandmarker from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("ERROR: Model file not found! Please check weights path.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    print("MediaPipe Initialized. Scanning directory...")

    print("MediaPipe Initialized. Scanning directory...")

    # Initialize CSVs with new column structure
    for csv_path in [RESULTS_ORIGINAL_CSV, RESULTS_BALANCED_CSV]:
        if not os.path.exists(csv_path):
            pd.DataFrame(columns=CSV_COLUMNS).to_csv(csv_path, index=False)
    
    processed_orig = set()
    processed_balanced = set()
    
    for csv_path in [RESULTS_ORIGINAL_CSV, RESULTS_BALANCED_CSV]:
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Check if 'Race' column exists and filter invalid rows
                if 'Race' in df.columns:
                    initial_len = len(df)
                    # Keep rows where Race is not null/empty
                    df_valid = df[df['Race'].notna() & (df['Race'] != '')]
                    
                    if len(df_valid) < initial_len:
                        print(f"Removing {initial_len - len(df_valid)} incomplete rows (missing Race) from {csv_path} to re-process.")
                        df_valid.to_csv(csv_path, index=False)
                    
                    if csv_path == RESULTS_ORIGINAL_CSV:
                        processed_orig = set(df_valid['File Path'].tolist())
                    else:
                        processed_balanced = set(df_valid['File Path'].tolist())
                else:
                    # If Race column is missing entirely, we might need to re-process all (or just keep as is if old format)
                    # But for now, if column missing, let's assume we treat all as valid or let the script append?
                    # Safer: If 'Race' column missing in header, the script initialization (line 554) would have handled it 
                    # if file didn't exist. If it exists but no Race column, likely old CSV.
                    pass
            except Exception as e:
                print(f"Error reading/cleaning {csv_path}: {e}")

    for root, dirs, files in os.walk(OUTPUT_WORK_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                file_path = os.path.join(root, file)
                
                needs_orig = file_path not in processed_orig
                needs_balanced = file_path not in processed_balanced
                
                if not needs_orig and not needs_balanced: continue

                print(f"Processing: {file}")
                
                try:
                    rel_dir = os.path.relpath(root, OUTPUT_WORK_DIR)
                    parts = rel_dir.split(os.sep)
                    model_name = parts[0] if len(parts) >= 1 else "unknown"
                    prompt_name = parts[1] if len(parts) >= 2 else "unknown"
                except:
                    model_name, prompt_name = "unknown", "unknown"

                img_raw = cv2.imread(file_path)
                if img_raw is None: continue
                
                # --- Pipeline 1: Original ---
                if needs_orig:
                    try:
                        rel_path = os.path.relpath(file_path, OUTPUT_WORK_DIR)
                        norm_orig_path = os.path.join(OUTPUT_NORM_ORIGINAL_DIR, rel_path)
                        os.makedirs(os.path.dirname(norm_orig_path), exist_ok=True)
                        cv2.imwrite(norm_orig_path, img_raw)
                        
                        process_analysis_pass(
                            norm_img_path=norm_orig_path,
                            original_file_path=file_path,
                            model_name=model_name,
                            prompt_name=prompt_name,
                            face_output_dir=OUTPUT_FACES_ORIGINAL_DIR,
                            csv_file=RESULTS_ORIGINAL_CSV,
                            img_norm_bgr=img_raw,
                            detector=detector,
                            pass_name="ORIGINAL"
                        )
                    except Exception as e: pass

                # --- Pipeline 2: Balanced Normalization (New: 50% Orig + 50% Hist -> Bg-Only WB) ---
                if needs_balanced:
                    try:
                        # 1. Blend 50/50 Original and Hist
                        norm_hist = histogram_normalization(img_raw)
                        blend_50_50 = cv2.addWeighted(img_raw, 0.5, norm_hist, 0.5, 0)
                        
                        # 2. Detect Face to create Background Mask
                        # Run detection on the blended image
                        rgb_blend = cv2.cvtColor(blend_50_50, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_blend)
                        detection = detector.detect(mp_img)
                        
                        img_balanced = blend_50_50.copy()
                        
                        if detection.face_landmarks:
                            lm = detection.face_landmarks[0]
                            h, w = img_raw.shape[:2]
                            
                            # Create Face Hull Mask
                            all_landmarks_idx = list(range(468)) 
                            face_pts = get_landmark_points(lm, all_landmarks_idx, w, h)
                            face_hull = cv2.convexHull(face_pts)
                            
                            mask_face = np.zeros((h, w), dtype=np.uint8)
                            cv2.fillConvexPoly(mask_face, face_hull, 255)
                            # Dilate to cover edges/hair
                            mask_face = cv2.dilate(mask_face, np.ones((30,30), np.uint8), iterations=1)
                            mask_bg = cv2.bitwise_not(mask_face)
                            
                            # 3. Apply White Balance (Top 1% of BACKGROUND)
                            for i in range(3):
                                # Extract background pixels for this channel
                                bg_vals = img_balanced[:,:,i][mask_bg > 0]
                                
                                if len(bg_vals) > 500: # Ensure enough pixels
                                     v_max = np.percentile(bg_vals, 99)
                                else:
                                     # Fallback to global if no background 
                                     v_max = np.percentile(img_balanced[:,:,i], 99)
                                     
                                if v_max == 0: v_max = 255
                                scale = 255.0 / v_max
                                img_balanced[:,:,i] = np.clip(img_balanced[:,:,i] * scale, 0, 255)
                        else:
                            # Fallback: Global WB if no face detected in this pre-pass
                            for i in range(3):
                                v_max = np.percentile(img_balanced[:,:,i], 99)
                                if v_max == 0: v_max = 255
                                scale = 255.0 / v_max
                                img_balanced[:,:,i] = np.clip(img_balanced[:,:,i] * scale, 0, 255)

                        rel_path = os.path.relpath(file_path, OUTPUT_WORK_DIR)
                        norm_balanced_path = os.path.join(OUTPUT_NORM_BALANCED_DIR, rel_path)
                        os.makedirs(os.path.dirname(norm_balanced_path), exist_ok=True)
                        cv2.imwrite(norm_balanced_path, img_balanced)
                        
                        process_analysis_pass(
                            norm_img_path=norm_balanced_path,
                            original_file_path=file_path,
                            model_name=model_name,
                            prompt_name=prompt_name,
                            face_output_dir=OUTPUT_FACES_BALANCED_DIR,
                            csv_file=RESULTS_BALANCED_CSV,
                            img_norm_bgr=img_balanced,
                            detector=detector,
                            pass_name="BALANCED_NORM"
                        )
                    except Exception as e: pass

    print("Analysis complete.")

if __name__ == "__main__":
    analyze_images()
