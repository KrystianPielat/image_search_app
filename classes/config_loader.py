import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConfigLoader:
    """
    A dataclass for loading configuration from a `.env` file, exposing specified fields as attributes,
    and ensuring that directories for folder fields are created.

    Attributes must be explicitly defined in the class.
    """
    MODELS_DIR: Optional[str] = field(default=None, metadata={"env": "MODELS_DIR", "is_folder": True})
    IMAGES_DIR: Optional[str] = field(default=None, metadata={"env": "IMAGES_DIR", "is_folder": True})
    IMAGES_COLLECTION_NAME: Optional[str] = field(default=None, metadata={"env": "IMAGES_COLLECTION_NAME"})
    MILVUS_HOST: Optional[str] = field(default='localhost', metadata={"env": "MILVUS_HOST"})
    MILVUS_PORT: Optional[str] = field(default=19530, metadata={"env": "MILVUS_PORT"})
    
    def __post_init__(self):
        """
        Post-initialization to load values from the `.env` file and ensure specified folders exist.
        """
        load_dotenv()
    
        for field_name, field_def in self.__dataclass_fields__.items():
            env_key = field_def.metadata.get("env")
            is_folder = field_def.metadata.get("is_folder", False)
            if env_key:
                value = os.getenv(env_key, getattr(self, field_name))
                if field_def.type == bool and value is not None:
                    value = value.lower() in ["true", "1", "yes"]
                setattr(self, field_name, value)
    
                if is_folder and value:
                    os.makedirs(value, exist_ok=True)


    def __repr__(self):
        """Custom string representation for the configuration."""
        config_values = {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}
        return f"ConfigLoader({config_values})"

config = ConfigLoader()