from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
from tqdm.autonotebook import tqdm
from torch import Tensor
from typing import List, Union, Optional, Literal

class Embedder:
    """
    A utility class for embedding images and sentences using pre-trained
    SentenceTransformer models.

    Attributes:
        _device (torch.device): The device to run the embeddings on (CPU or CUDA).
        _base_model (SentenceTransformer): The model used for embedding images.
        _ml_model (SentenceTransformer): The model used for embedding sentences.
    """

    def __init__(self, base_model: SentenceTransformer, ml_model: SentenceTransformer, device: Optional[Literal['cpu', 'cuda']] = None):
        """
        Initializes the Embedder with the specified models.

        Args:
            base_model (SentenceTransformer): A pre-trained SentenceTransformer model for images.
            ml_model (SentenceTransformer): A pre-trained SentenceTransformer model for sentences.
        """
        if device is not None:
            self._device = device
        else:
            self._device = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )

        self._base_model = base_model.to(self._device)
        self._ml_model = ml_model.to(self._device)

    def embed_images(
        self, 
        images: Union[List[object], object], 
        batch_size: int = 8, 
        show_progress_bar: bool = False
    ) -> Tensor:
        """
        Embeds images using the base SentenceTransformer model.
    
        Args:
            images (Union[List[object], object]): A single image or a list of images to embed.
            batch_size (int, optional): The batch size for embedding. Defaults to 8.
            show_progress_bar (bool, optional): Whether to display a progress bar. Defaults to False.
    
        Returns:
            Tensor: The normalized embeddings for the images.
        """
        if not isinstance(images, list):
            images = [images]
        embeddings = self._base_model.encode(
            images,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
            device=self._device,
        ).to(self._device)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def embed_sentences(
        self, 
        sentences: Union[List[str], str], 
        batch_size: int = 8, 
        show_progress_bar: bool = False
    ) -> Tensor:
        """
        Embeds sentences using the multilingual SentenceTransformer model.
    
        Args:
            sentences (Union[List[str], str]): A single sentence or a list of sentences to embed.
            batch_size (int, optional): The batch size for embedding. Defaults to 8.
            show_progress_bar (bool, optional): Whether to display a progress bar. Defaults to False.
    
        Returns:
            Tensor: The normalized embeddings for the sentences.
        """
        if not isinstance(sentences, list):
            sentences = [sentences]
        embeddings = self._ml_model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
            device=self._device,
        ).to(self._device)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)