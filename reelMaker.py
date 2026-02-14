import os
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import sys
import requests
from transformers import pipeline
import spacy
from moviepy.editor import (
    ColorClip,
    ImageClip,
    CompositeVideoClip,
    VideoClip,
    VideoFileClip,
    AudioFileClip,
    vfx 
)
import numpy as np
from scipy.ndimage import map_coordinates
import sqlite3
import re
from random import randrange

def add_audio_to_video(video_path, audio_path, output_path):
    """
    Adds an audio track to the given video and saves the result.

    :param video_path: Path to the input video file.
    :param audio_path: Path to the audio file (e.g., audio.mp3).
    :param output_path: Path to save the output video with audio.
    """

    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Check if the audio file exists
    if not os.path.isfile(audio_path):
        print(f"Error: Audio file '{audio_path}' not found.")
        return

    try:
        # Load the video clip
        video_clip = VideoFileClip(video_path)

        # Load the audio clip
        audio_clip = AudioFileClip(audio_path)

        # Ensure the audio is not longer than the video
        if audio_clip.duration > video_clip.duration:
            audio_clip = audio_clip.subclip(0, video_clip.duration)
            print("Audio was trimmed to match the video duration.")
        elif audio_clip.duration < video_clip.duration:
            # Well if you want, you can loop the audio to match the video duration for more flexible projects :)
            # Uncomment the following lines if you want to loop the audio
            # n_loops = int(video_clip.duration // audio_clip.duration) + 1
            # audio_clip = concatenate_audioclips([audio_clip] * n_loops).subclip(0, video_clip.duration)
            # print("Audio was looped to match the video duration.")
            pass  # If you find this comment useful leave a 5 star on the repo xD

        # Set the audio of the video clip
        final_video = video_clip.set_audio(audio_clip)

        # Write the result to the output file
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            bitrate="2000k",
            fps=video_clip.fps,
            preset="medium",
            threads=4,
            logger='bar'  # remove this if you are autistic
        )

        print(f"Audio added successfully. Saved to '{output_path}'.")

    except Exception as e:
        print(f"An error occurred while adding audio: {e}")

    finally:
        # Close the clips to release resources
        video_clip.close()
        audio_clip.close()
        if 'final_video' in locals():
            final_video.close()



# reel maker helper functions lets go
#=================================================================================
def create_reels_folder():
    # Define the path to the "reels" folder in the current directory
    reels_folder = os.path.join(os.getcwd(), "reels/")

    # Check if the "reels" folder exists, and remove it if it does (poor coding but effective)
    if os.path.exists(reels_folder):
        shutil.rmtree(reels_folder)  

    # Create a new, empty "reels" folder
    os.makedirs(reels_folder)
    return reels_folder

def resize_and_crop(img, target_width, target_height):
    img_width, img_height = img.size
    img_ratio = img_width / img_height
    target_ratio = target_width / target_height

    if img_ratio > target_ratio:
        # Image is wider than target; crop width
        new_height = target_height
        new_width = int(new_height * img_ratio)
    else:
        # Image is taller than target; crop height
        new_width = target_width
        new_height = int(new_width / img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Now crop the image to the target size
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = left + target_width
    bottom = top + target_height

    img = img.crop((left, top, right, bottom))

    return img

def create_caption(result_labels):
    caption = ""
    for result in result_labels[:10]:
        caption = caption + "#" + result.replace(" ", "_") + " "

    return caption

def clean_string(text):
    # Regular expression to keep alphabets, numbers, and spaces only
    text = text.replace(" ", "_")
    cleaned_text = re.sub(r'[^A-Za-z0-9_\s]', '', text)
    return cleaned_text.strip()

def check_if_quote_is_unique(quote):
    conn = sqlite3.connect("used_quote.db")
    cursor = conn.cursor()
    
    # Create table with an auto-incrementing ID if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS used_quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT, 
        quote TEXT UNIQUE
    )''')

    try:
        cursor.execute("INSERT INTO used_quotes (quote) VALUES (?)", (quote,))
        conn.commit()
        print(f"{quote} added to the used quotes list.")
        return True
    except sqlite3.IntegrityError:
        print(f"{quote} has already been used.")
        return False


def generate_quote_caption():
    # for video_number in range(videos_number):
    while True:
        response = requests.get('https://zenquotes.io/api/random')
        if check_if_quote_is_unique(response.json()[0]['q']):
            break


    # Initialize the zero-shot classification pipeline
    # print(response.text[0]['q'])
    # exit()
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Define the text and the candidate categories
    author = response.json()[0]['a']
    quote = response.json()[0]['q']

    quote_categories = [
        "Inspirational",
        "Motivational",
        "Love",
        "Life",
        "Success",
        "Happiness",
        "Friendship",
        "Wisdom",
        "Humor",
        "Philosophy",
        "Education",
        "Change",
        "Courage",
        "Hope",
        "Dreams",
        "Art",
        "Leadership",
        "Health",
        "Business",
        "Time",
        "Creativity",
        "Kindness",
        "Gratitude",
        "Positive Thinking",
        "Spirituality",
        "Family",
        "Nature",
        "Mindfulness",
        "Self-Esteem",
        "Relationships",
        "Fear",
        "Failure",
        "Travel",
        "Beauty",
        "Memories",
        "Perseverance",
        "Respect",
        "Ambition",
        "Passion",
        "Forgiveness",
        "Peace",
        "Aging",
        "Equality",
        "Empathy",
        "Environment",
        "Faith",
        "Freedom",
        "Growth",
        "Justice",
        "Knowledge",
        "Learning",
        "Mindset",
        "Optimism",
        "Patience",
        "Persistence",
        "Reflection",
        "Resilience",
        "Teamwork",
        "Trust",
        "Understanding",
        "Self-Improvement",
        "Motivation",
        "Inner Strength",
        "Overcoming Challenges",
        "Balance",
        "Positivity",
        "Self-Discovery",
        "Self-Love",
        "Vision",
        "Opportunity",
        "Innovation",
        "Achievement",
        "Empowerment",
        "Simplicity",
        "Joy",
        "Dedication",
        "Integrity",
        "Responsibility",
        "Adaptability",
        "Curiosity",
        "Determination",
        "Focus",
        "Growth Mindset",
        "Hustle",
        "Inspiration",
        "Potential",
        "Purpose",
        "Strength",
        "Transformation",
        "Wellness",
        "Wonder",
        "Zeal"
    ]

    # Perform classification
    result = classifier(quote, quote_categories)
    print(quote)
    
    caption = create_caption(result['labels'])
    print(caption)

    return quote, author, caption

def extract_keywords_spacy(text, top_n=10):
    """
    Extracts keywords from the given text using spaCy.

    :param text: The input text from which to extract keywords.
    :param top_n: The maximum number of keywords to return.
    :return: A list of extracted keywords.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Extract nouns and proper nouns as potential keywords
    keywords = [token.text for token in doc if token.pos_ in ("NOUN", "PROPN")]

    # Filter out stop words and punctuation
    keywords = [word for word in keywords if not nlp.vocab[word].is_stop and word.isalpha()]

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word.lower() not in seen:
            seen.add(word.lower())
            unique_keywords.append(word)

    # Return top_n keywords
    return unique_keywords[:top_n]



def generate_quote_image_with_bold_keywords(
    quote,
    author,
    referal,
    image_width,
    image_height,
    output_path,
    font_regular_path,
    font_bold_path,
    keywords,
    line_spacing=3.0  # May interest you if you want wider line spaces 
):
    """
    Generates an image with the given quote and author, bolding the specified keywords and coloring the author name.
    Includes customizable line spacing and referral text at 3/4 of the image height.

    :param quote: The quote text.
    :param author: The author of the quote.
    :param referal: The referral text.
    :param image_width: Width of the image in pixels.
    :param image_height: Height of the image in pixels.
    :param output_path: Path to save the generated image.
    :param font_regular_path: Path to the regular font file.
    :param font_bold_path: Path to the bold font file.
    :param keywords: List of keywords to be bolded.
    :param line_spacing: Factor to control the spacing between lines. Default is 3.0.  # CHANGED
    """
    # Define margins based on the original script
    horizontal_padding = 200  # 200 pixels total padding (100 on each side)
    # vertical_padding = 50      # Adjusted to 50 pixels for better vertical spacing

    # Create an empty image with a transparent background
    img = Image.new('RGBA', (image_width, image_height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Define font properties
    font_size = 45
    font_size_bold = 52  # Bold font is larger
    # referral_font_size = 30  # Font size for referral
    try:
        font_regular = ImageFont.truetype(font_regular_path, font_size)
        font_bold = ImageFont.truetype(font_bold_path, font_size_bold)
        font_bold_author = ImageFont.truetype(font_bold_path, font_size)
    except IOError:
        print(f"Font file not found. Please update the font paths.")
        return

    # Prepare the quote and author texts in the desired format
    # Format: "- AUTHOR\n\n\"QUOTE\""
    full_author = f"- {author}"
    full_quote = f'" {quote} "'

    # Combine author and quote for wrapping
    combined_text = f"{full_quote}\n\n{full_author}"

    # Define the maximum width for the text box
    box_width = image_width - 2 * horizontal_padding  # Total horizontal padding

    # Split the combined_text into lines considering newlines
    lines = combined_text.split('\n')

    # Function to wrap text within a given width
    def wrap_line(line, font, max_width):
        words = line.split()
        wrapped = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if draw.textlength(test_line, font=font) <= max_width:
                current_line = test_line
            else:
                if current_line:  # Avoid adding empty lines
                    wrapped.append(current_line.strip())
                current_line = word + " "
        if current_line:
            wrapped.append(current_line.strip())
        return wrapped

    # Wrap each line separately
    wrapped_lines = []
    for line in lines:
        if line.strip() == "":
            wrapped_lines.append("")  # Preserve empty lines for spacing
            continue
        # Use regular font for wrapping
        wrapped = wrap_line(line, font_regular, box_width)
        wrapped_lines.extend(wrapped)

    # Calculate the total height of the text block with line spacing
    line_heights = []
    for line in wrapped_lines:
        if line.strip() == "":
            # Assume double line spacing for empty lines
            line_heights.append((font_size // 2) * line_spacing)  
            continue
        # Use maximum font size for line height
        line_heights.append(font_size_bold * line_spacing) 

    total_text_height = sum(line_heights)

    # Calculate the starting y-position to center the text block vertically
    y = (image_height - total_text_height) / 2  

    # Define shadow properties
    shadow_color = (0, 0, 0, 255)  # Black shadow
    shadow_offsets = [(-2, -2), (-2, 2), (2, -2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)]  

    # Define the orange color for the author
    orange_color = (232, 113, 2, 255)  

    # Prepare a list of author words for coloring
    author_words = [w.lower().strip('.,!?;"\'') for w in author.split()]
    # Prepare a list of keyword words for bolding
    keyword_words = [kw.lower() for kw in keywords]

    # Define y-shift for bold words to align baselines
    bold_y_shift = -7  

    # Draw each line with shadow and handle bold keywords and colored author
    for idx, line in enumerate(wrapped_lines):
        if line.strip() == "":
            y += line_heights[idx]
            continue

        words = line.split()
        # Calculate the total width of the line with appropriate fonts
        line_width = 0
        word_fonts = []
        word_colors = []
        for word in words:
            word_clean = word.strip('.,!?;\'').lower()
            # Determine the styling based on whether the word is part of the author or a keyword
            if word_clean in author_words or word_clean == '"' or word_clean == "-":
                font = font_bold_author
                text_color = orange_color  # Orange color for author
            elif word_clean in keyword_words:
                font = font_bold
                text_color = (255, 255, 255, 255)  # White for keywords
            else:
                font = font_regular
                text_color = (255, 255, 255, 255)  # White for regular text
            word_fonts.append(font)
            word_colors.append(text_color)
            # Calculate word width
            word_width = draw.textlength(word + " ", font=font)
            line_width += word_width

        # Calculate the starting x-position to center the line horizontally
        x = (image_width - line_width) / 2

        # Draw each word in the line
        for word, font, text_color in zip(words, word_fonts, word_colors):
            # Calculate word width
            word_width = draw.textlength(word + " ", font=font)

            # Determine y-offset based on font type
            if font == font_bold:
                current_y = y + bold_y_shift  
            else:
                current_y = y  # Regular words use the base y position

            # Shadow rendering
            for offset in shadow_offsets:
                shadow_x = x + offset[0]
                shadow_y = current_y + offset[1]
                draw.text((shadow_x, shadow_y), word + " ", font=font, fill=shadow_color)

            # Draw the main text
            draw.text((x, current_y), word + " ", font=font, fill=text_color)

            # Move x position for next word
            x += word_width

        # Move to the next line
        y += line_heights[idx]

    # Draw the referral text at 3/4 of the image height
    referral_y = image_height * 3 / 4  # Position at 3/4 of the height
    referral_width = draw.textlength(referal, font=font_bold)
    referral_x = (image_width - referral_width) / 2  # Center the referral horizontally
    draw.text((referral_x, referral_y), referal, font=font_bold, fill=(255, 255, 255, 255))  # White color

    # Save the image
    img.save(output_path)
    print(f"Image saved to {output_path}")



def apply_waving_effect(clip, max_amplitude_x=3, variation_x=1.0, max_amplitude_y=3, variation_y=1.0):
    """
    Apply a waving effect to the clip by distorting both x and y coordinates based on wave functions.
    The amplitude of the wave is higher along the diagonals from the corners to the center, forming an 'X' pattern.
    
    Args:
        clip (VideoClip): The clip to apply the wave effect to.
        max_amplitude_x (float): The maximum horizontal displacement in pixels along the diagonals.
        variation_x (float): Controls the frequency of the horizontal wave.
        max_amplitude_y (float): The maximum vertical displacement in pixels along the diagonals.
        variation_y (float): Controls the frequency of the vertical wave.
    
    Returns:
        VideoClip: The clip with the waving effect applied.
    """
    def wave_function(get_frame, t):
        frame = get_frame(t)
        h, w, _ = frame.shape  # Ensure frame has 3 channels

        # Create coordinate grids
        x_coords = np.arange(w)
        y_coords = np.arange(h)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Compute distances from the two diagonals
        diag1 = np.abs(x_grid + y_grid - (w + h) / 2)
        diag2 = np.abs(x_grid - y_grid)
        
        # Normalize distances to range from 0 at the diagonals to 1 at the furthest points
        max_diag_distance = np.sqrt(w**2 + h**2) / 2
        normalized_distance = (diag1 + diag2) / (2 * max_diag_distance)  # Average the distances
        normalized_distance = np.clip(normalized_distance, 0, 1)

        # Amplitude scaling is maximum along the diagonals
        amplitude_scaling = 1 - normalized_distance  # 1 at diagonals, decreasing towards edges

        # Generate wave offsets for x and y
        phase_x = t * variation_x * 2 * np.pi
        phase_y = t * variation_y * 2 * np.pi

        frequencies = np.linspace(0.5, 2.0, num=5)

        x_offsets = np.zeros_like(x_grid, dtype=np.float32)
        y_offsets = np.zeros_like(y_grid, dtype=np.float32)

        for freq in frequencies:
            x_offsets += (max_amplitude_x * amplitude_scaling) * np.sin(freq * (x_grid / w * 2 * np.pi) + phase_x)
            y_offsets += (max_amplitude_y * amplitude_scaling) * np.sin(freq * (y_grid / h * 2 * np.pi) + phase_y)

        # Apply offsets
        x_new = x_grid + x_offsets
        y_new = y_grid + y_offsets

        # Clip coordinates to valid range
        x_new = np.clip(x_new, 0, w - 1).astype(np.float32)
        y_new = np.clip(y_new, 0, h - 1).astype(np.float32)

        # Use map_coordinates to remap the pixels
        distorted_frame = np.zeros_like(frame)
        for i in range(3):  # For each color channel
            distorted_frame[:,:,i] = map_coordinates(frame[:,:,i], [y_new.ravel(), x_new.ravel()], order=1).reshape(h, w)

        return distorted_frame

    return clip.fl(wave_function, apply_to=['video'])  # Apply only to 'video'

def interpolate_vertices(t, keyframes):
    """
    Interpolates the positions of the pentagon's vertices at time t.
    """
    times = sorted(keyframes.keys())
    for i, time in enumerate(times):
        if t == time:
            return keyframes[time]
        if t < time:
            t0, t1 = times[i-1], time
            vertices_t0, vertices_t1 = keyframes[t0], keyframes[t1]
            ratio = (t - t0) / (t1 - t0)
            interpolated_vertices = [
                (x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio)
                for (x0, y0), (x1, y1) in zip(vertices_t0, vertices_t1)
            ]
            return interpolated_vertices
    return keyframes[times[-1]]  # Return the final positions if time exceeds the keyframes

def create_dynamic_pentagon_mask(width, height, duration, fps, keyframes, blur_radius=50):
    """
    Creates a mask function that moves the pentagon's vertices based on predefined positions at specific times.
    """
    def mask_frame(t):
        if t < 0:
            t = 0
        elif t > duration:
            t = duration

        new_vertices = interpolate_vertices(t, keyframes)
        mask_image = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.polygon(new_vertices, fill=255)
        blurred_mask = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
        mask = np.array(blurred_mask) / 255.0
        return mask
    return mask_frame

def zoom_in_out_end(get_frame, t):
    """
    Apply zoom-in and zoom-out at the end of the video with specific scaling behavior.
    
    Zoom in to 1.3x from 5s to 5.3s,
    Zoom out to 1.2x from 5.3s to 5.5s,
    Zoom in to 1.5x from 5.5s to 6s.
    """
    if 7 <= t < 8:
        scale = 1.0 + (0.3 * (t - 7) / 0.3)  
    # elif 5.3 <= t < 5.5:
    #     scale = 1.3 - (0.1 * (t - 5.3) / 0.2)  
    # elif 5.5 <= t <= 6:
    #     scale = 1.2 + (0.3 * (t - 5.5) / 0.5) 
    else:
        scale = 1.0
    
    frame = get_frame(t)
    pil_image = Image.fromarray(frame)
    new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
    pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Paste resized image onto a black background
    background = Image.new('RGB', (frame.shape[1], frame.shape[0]), (0, 0, 0))
    paste_position = ((frame.shape[1] - new_size[0]) // 2, (frame.shape[0] - new_size[1]) // 2)
    background.paste(pil_image, paste_position)
    
    return np.array(background)

def create_reversed_pentagon_mask(width, height, duration, fps, keyframes, blur_radius=50):
    """
    Creates a reversed pentagon mask function for the end of the video.
    The mask appears to shrink instead of expand.
    """
    def mask_frame(t):
        # Reverse the timing by flipping the progress
        t_reversed = duration - t
        return create_dynamic_pentagon_mask(width, height, duration, fps, keyframes, blur_radius)(t_reversed)
    
    return mask_frame

def create_thumbnail(background_image_path, quote_image_path, output_thumbnail_path, desaturation_level=0.1):
    """
    Creates a thumbnail by desaturating the background image and overlaying the quote image on top.

    :param background_image_path: Path to the background image (e.g., "background_resized.jpg").
    :param quote_image_path: Path to the quote image with shadow (e.g., "quote_image_with_shadow.png").
    :param output_thumbnail_path: Path to save the generated thumbnail image (e.g., "thumbnail.png").
    :param desaturation_level: Level of desaturation (0.0 to 1.0, where 0.0 is grayscale and 1.0 is original color).
    """
    from PIL import Image, ImageEnhance
    import os

    # Check if input files exist
    if not os.path.isfile(background_image_path):
        print(f"Error: Background image '{background_image_path}' not found.")
        return
    if not os.path.isfile(quote_image_path):
        print(f"Error: Quote image '{quote_image_path}' not found.")
        return

    try:
        # Open and desaturate the background image
        background = Image.open(background_image_path).convert('RGBA')
        enhancer = ImageEnhance.Color(background)
        background_desaturated = enhancer.enhance(desaturation_level)
        print(f"Desaturated background image with desaturation level {desaturation_level}.")
        
        # **Debug Step**: Save the desaturated background to verify
        debug_desaturated_path = "background_desaturated_debug.png"
        background_desaturated.save(debug_desaturated_path)
        print(f"Desaturated background saved for debugging at '{debug_desaturated_path}'.")
    
    except Exception as e:
        print(f"Error processing background image '{background_image_path}': {e}")
        return

    try:
        # Open the quote image
        quote = Image.open(quote_image_path).convert('RGBA')
        print(f"Loaded quote image '{quote_image_path}' with size {quote.size}.")
    
    except Exception as e:
        print(f"Error loading quote image '{quote_image_path}': {e}")
        return

    try:
        # Calculate position to place the quote image at the top center
        bg_width, bg_height = background_desaturated.size
        quote_width, quote_height = quote.size

        padding_top = 50  # Pixels from the top

        position = (
            (bg_width - quote_width) // 2,
            padding_top
        )
        print(f"Pasting quote image at position {position}.")

        # Paste the quote image onto the desaturated background using alpha channel as mask
        background_desaturated.paste(quote, position, quote)
        print("Successfully pasted quote image onto background.")
    
    except Exception as e:
        print(f"Error pasting quote image onto background: {e}")
        return

    try:
        # Save the combined image as the thumbnail
        background_desaturated.save(output_thumbnail_path)
        print(f"Thumbnail saved successfully to '{output_thumbnail_path}'.")
    
    except Exception as e:
        print(f"Error saving thumbnail to '{output_thumbnail_path}': {e}")
        return

#=================================================================================

def reel_maker():
    reels_folder = create_reels_folder()
    # Configuration
    TARGET_WIDTH = 1080
    TARGET_HEIGHT = 1920
    VIDEO_DURATION = 8          # Total duration in seconds
    BLACK_DURATION = 0          # Duration of the initial black screen
    REVEAL_DURATION = 7.2         # Duration of the reveal effect
    STATIC_DURATION = VIDEO_DURATION - BLACK_DURATION - REVEAL_DURATION  # Remaining duration
    FPS = 24                     # Frames per second (increase it if you are rich and have good hardware)

    quote, author, caption = generate_quote_caption()

    
    chosen = randrange(22)
    BACKGROUND_IMAGE = f"Background/{chosen}.jpeg"  # Name of the variable is self explanatory
    QUOTE_IMAGE = "quote_image_with_shadow.png"  # The image you want to overlay
    Video_ID = clean_string(quote) + ".mp4"    
    reels_folder = "./reels"
    
    OUTPUT_VIDEO = reels_folder + "/" + Video_ID
    # Load and process the background image
    try:
        pil_image = Image.open(BACKGROUND_IMAGE)
    except FileNotFoundError:
        print(f"Error: '{BACKGROUND_IMAGE}' not found. Please ensure the image is in the current directory.")
        sys.exit(1)

    # Resize and crop the background image to the target dimensions
    pil_image = resize_and_crop(pil_image, TARGET_WIDTH, TARGET_HEIGHT)

    # Save the resized background image to a temporary file
    BACKGROUND_IMAGE_RESIZED = "background_resized.jpg"
    pil_image.save(BACKGROUND_IMAGE_RESIZED)

    width, height = TARGET_WIDTH, TARGET_HEIGHT

    # Define the positions of the pentagon vertices at specific times (in seconds)
    keyframes = {
        0: [(width * 0.5, height * 0.5)] * 5,  # t = 0s
        0.25: [(width * 0.5, height * 0.5)] * 5,  # t = 0.25s
        0.5: [
            (width * 0.40, height * 0.45),
            (width * 0.55, height * 0.4),
            (width * 0.65, height * 0.45),
            (width * 0.55, height * 0.58),
            (width * 0.30, height * 0.55)
        ],  # t = 0.5s
        0.75: [
            (width * 0.05, height * 0.42),
            (width * 0.55, height * 0.3),
            (width * 0.9, height * 0.4),
            (width * 0.8, height * 0.60),
            (width * 0.1, height * 0.6)
        ],  # t = 0.75s
        1: [
            (width * 0.05, height * 0.42),
            (width * 0.55, height * 0.3),
            (width * 0.9, height * 0.4),
            (width * 0.8, height * 0.60),
            (width * 0.1, height * 0.6)
        ],  # t = 1s
        2.75: [
            (width * 0.05, height * 0.05),
            (width * 0.50, height * 0.05),
            (width * 0.95, height * 0.05),
            (width * 0.95, height * 0.95),
            (width * 0.05, height * 0.95)
        ],  # t = 2.75s
    }

    # Define the positions of the pentagon vertices at specific times (in seconds)
    keyframes_reverse = {
        0: [(width * 0.5, height * 0.5)] * 5,  # t = 0s
        0.15: [
            (width * 0.40, height * 0.45),
            (width * 0.55, height * 0.4),
            (width * 0.65, height * 0.45),
            (width * 0.55, height * 0.58),
            (width * 0.30, height * 0.55)
        ],  # t = 0.5s
        0.25: [
            (width * 0.05, height * 0.42),
            (width * 0.55, height * 0.3),
            (width * 0.9, height * 0.4),
            (width * 0.8, height * 0.60),
            (width * 0.1, height * 0.6)
        ],  # t = 0.75s
        0.3: [
            (width * 0.05, height * 0.42),
            (width * 0.55, height * 0.3),
            (width * 0.9, height * 0.4),
            (width * 0.8, height * 0.60),
            (width * 0.1, height * 0.6)
        ],  # t = 1s
        0.7: [
            (width * 0.05, height * 0.05),
            (width * 0.50, height * 0.05),
            (width * 0.95, height * 0.05),
            (width * 0.95, height * 0.95),
            (width * 0.05, height * 0.95)
        ],  # t = 2.75s
    }
    # Initialize spaCy model once
    # nlp = spacy.load("en_core_web_sm")

    # TODO: Bro remove the coppies it is a waste of fucking ressources 
    # quote, author, caption = generate_quote_caption()

    sample_quote = quote
    sample_author = author
    referal = "@Astonishing.Quote"

    # Extract keywords
    keywords = extract_keywords_spacy(sample_quote)
    print("Extracted Keywords:", keywords)

    # Generate the quote image with bold keywords and colored author
    generate_quote_image_with_bold_keywords(
        sample_quote,
        sample_author,
        referal,
        width,
        height,
        QUOTE_IMAGE,
        "/usr/share/fonts/X11/Type1/NimbusRoman-Italic.pfb",  # Update as needed
        "/usr/share/fonts/X11/Type1/NimbusRoman-BoldItalic.pfb",  # Update as needed
        keywords,
        line_spacing=1.2,
    )
    # Create the Thumbnail
    create_thumbnail(BACKGROUND_IMAGE_RESIZED, QUOTE_IMAGE, reels_folder + "/thumbnail.png")
    # Create the background and quote image clips
    background_clip = ImageClip(BACKGROUND_IMAGE_RESIZED).set_duration(VIDEO_DURATION)
    quote_image_clip = ImageClip(QUOTE_IMAGE).set_duration(VIDEO_DURATION).set_position("center")
    
    # Apply Black-and-White Effect to the Background Clip with a Degree of Saturation
    desaturation_level = 0.2  # A degree between 0 (black-and-white) and 1 (full color)
    background_clip = background_clip.fx(vfx.colorx, desaturation_level)

    # Composite the background and the quote into a single clip
    fused_clip = CompositeVideoClip([background_clip, quote_image_clip])

    # Apply the waving effect to the fused clip
    # Adjust max_amplitude and variation as needed for desired effect
    fused_clip = apply_waving_effect(fused_clip, max_amplitude_x=3, variation_x=0.5, max_amplitude_y=3, variation_y=0.5)

    # Define the mask for the reveal effect
    mask_func = create_dynamic_pentagon_mask(width, height, REVEAL_DURATION, FPS, keyframes)
    mask_clip = VideoClip(mask_func, duration=REVEAL_DURATION).set_fps(FPS)
    mask_clip = mask_clip.set_duration(REVEAL_DURATION)
    mask_clip.ismask = True  # Indicate that this clip is a mask

    # Apply the mask to the fused clip (background + quote)
    fused_with_mask = fused_clip.set_start(BLACK_DURATION).set_mask(mask_clip)

    # Create a black background
    black_clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=VIDEO_DURATION)

    # Composite the black background with the fused clip (with mask applied)
    final_video = CompositeVideoClip([black_clip, fused_with_mask])

    # Set the duration of the final video
    final_video = final_video.set_duration(VIDEO_DURATION)

    # ========== Enhancements Start Here ==========

    # **Enhancement 1: Add a Zoom-Out Effect for One Second at the Beginning**

    def zoom_out(get_frame, t):
        if t < 1:
            scale = 1.3 - 0.3 * t  # Scale from 1.3x to 1.0x over the first second
        else:
            scale = 1.0
        frame = get_frame(t)
        pil_image = Image.fromarray(frame)
        new_size = (int(frame.shape[1] * scale), int(frame.shape[0] * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        # To maintain the original size, paste the resized image onto a black background
        background = Image.new('RGB', (width, height), (0, 0, 0))
        paste_position = (
            (width - new_size[0]) // 2,
            (height - new_size[1]) // 2
        )
        background.paste(pil_image, paste_position)
        return np.array(background)

    # Apply the zoom-out effect to the final video
    final_video = final_video.fl(zoom_out, apply_to=['video'])  # Changed to only apply to 'video'

    # ========== Enhancements End Here ==========
    # Apply the zoom effect at the end
    fused_with_zoom = final_video.fl(zoom_in_out_end, apply_to=['video'])

    # Define the reversed mask for the end of the video
    reversed_mask_func = create_reversed_pentagon_mask(width, height, REVEAL_DURATION, FPS, keyframes_reverse)
    reversed_mask_clip = VideoClip(reversed_mask_func, duration=REVEAL_DURATION).set_fps(FPS)
    reversed_mask_clip = reversed_mask_clip.set_start(VIDEO_DURATION - REVEAL_DURATION)
    reversed_mask_clip.ismask = True  # Indicate that this is a mask

    # Apply the reversed mask to the clip at the end
    final_clip_with_reversed_mask = fused_with_zoom.set_mask(reversed_mask_clip)

    # Create a black background
    black_clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=VIDEO_DURATION)

    # Composite the black background with the fused clip (with reversed mask applied)
    final_video = CompositeVideoClip([black_clip, final_clip_with_reversed_mask])

    # **Step 6: Export the Final Video**
    final_video.write_videofile(
        OUTPUT_VIDEO,
        fps=FPS,  
        codec="libx264",
        audio=False,  # Set to True and provide an audio track if needed
        preset="medium",
        threads=4,
        logger='bar'  
    )

    # Adding audio 

    FINAL_OUTPUT_VIDEO = reels_folder + "/" + clean_string(quote) + "_with_audio.mp4"  # Final video with audio
    AUDIO_FILE = "audio.mp3" 
    add_audio_to_video(OUTPUT_VIDEO, AUDIO_FILE, FINAL_OUTPUT_VIDEO)

    # TODO: Clean up temporary files !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Could be used as thumbnail ? Anyways
    if os.path.exists(BACKGROUND_IMAGE_RESIZED):
        os.remove(BACKGROUND_IMAGE_RESIZED)
    
    return sample_quote, caption, Video_ID, FINAL_OUTPUT_VIDEO

# reel_maker()
