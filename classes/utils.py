import pickle
import os
import logging
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display
from sentence_transformers import SentenceTransformer

_LOGGER = logging.getLogger(__name__)

def display_results(results: List[Tuple[str, float]], width: int = 300, height: int = 200, font_size: int = 24):
    """
    Displays images with their respective scores overlaid as text, sorted by distance (ascending).

    Args:
        results (List[Tuple[str, float]]): List of tuples where each tuple contains
            the image path and the score (distance).
        width (int): The width to resize the images to.
        height (int): The height to resize the images to.
        font_size (int): Font size for the overlaid text.
    """
    results = sorted(results, key=lambda x: x[1])
    
    for res in results:
        img = Image.open(res[0])
        img = img.resize((width, height))
        
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        draw.text((10, 10), f"dist: {res[1]:.2f}", fill=(255, 0, 0), font=font)
        
        display(img)


def load_or_save_model(pickle_path: str, model_name: str):
    """
    Loads a pre-trained SentenceTransformer model from a pickle file if it exists, 
    otherwise downloads the model and saves it to the specified path.

    Args:
        pickle_path (str): The path to the pickle file for saving/loading the model.
        model_name (str): The name of the SentenceTransformer model to download if the pickle file does not exist.

    Returns:
        SentenceTransformer: The loaded or newly downloaded SentenceTransformer model.
    """
    if os.path.exists(pickle_path):
        _LOGGER.info(f"Loading model from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            model = pickle.load(f)
    else:
        _LOGGER.info(f"Downloading and saving model {model_name}...")
        model = SentenceTransformer(model_name)
        with open(pickle_path, 'wb') as f:
            pickle.dump(model, f)
    return model
