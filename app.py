import streamlit as st
import time
import logging
from typing import Optional
from PIL import Image
import shutil
import os
from io import BytesIO
from classes.embedder import Embedder
from classes.milvus_connector import MilvusConnector
from classes.utils import display_results, load_or_save_model
from classes.config_loader import config
from pymilvus import FieldSchema, DataType
from pymilvus.exceptions import MilvusException
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if 'last_input_time' not in st.session_state:
    st.session_state.last_input_time = time.time()
    st.session_state.search_triggered = False

@st.cache_resource
def load_embedder():
    clip_ml = load_or_save_model(
        os.path.join(config.MODELS_DIR, 'clip_ml.model'), 'clip-ViT-B-32-multilingual-v1'
    )
    clip = load_or_save_model(
        os.path.join(config.MODELS_DIR, 'clip.model'), 'clip-ViT-B-32'
    )
    return Embedder(base_model=clip, ml_model=clip_ml)

@st.cache_resource
def initialize_database():
    """
    Initialize and sync the database with images from the images folder.
    This function runs only once on startup due to @st.cache_resource.
    """
    check_and_sync_images()

def connect_with_retry(retries=3, delay=5):
    for attempt in range(retries):
        try:
            return MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        except MilvusException as e:
            if attempt < retries - 1:
                st.warning(f"Failed to connect to Milvus. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error("Failed to connect to Milvus after multiple attempts.")
                raise e

def embed_existing_images(connector: Optional[MilvusConnector] = None): 
    logger.info("Adding images from the folder to the collection...")
    images = []
    for image in os.listdir(config.IMAGES_DIR):
        path = os.path.join(config.IMAGES_DIR, image)
        # Skip directories and non-image files
        if os.path.isdir(path) or not image.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        logger.info(f"Processing image: {image}")
        images.append({
            'path': path,
            'image': Image.open(path),
            'embedding': None
        })
    
    logger.info(f"Found {len(images)} images to embed")
    batch = []
    for i, img in enumerate(images):
        logger.info(f"Embedding image {i+1}/{len(images)}: {os.path.basename(img['path'])}")
        img['embedding'] = embedder.embed_images(img['image']).to('cpu').tolist()[0]
        batch.append({'path': img['path'], 'embedding': img['embedding']})

    logger.info(f"Inserting {len(batch)} embeddings into database...")
    if connector:
        connector.insert(batch, collection_name=config.IMAGES_COLLECTION_NAME)
    else:
        with connect_with_retry() as connector:
            connector.insert(batch, collection_name=config.IMAGES_COLLECTION_NAME)
    logger.info("Images added successfully!")


def ensure_collection_exists():
    logger.info("ensure_collection_exists called")
    img_fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='path', dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    with connect_with_retry() as connector:
        if not connector.check_if_collection_exists(config.IMAGES_COLLECTION_NAME):
            logger.info("Image collection not found. Creating the collection...")
            connector.create_collection(config.IMAGES_COLLECTION_NAME, img_fields, remove_if_exists=False)
            logger.info(f"Collection {config.IMAGES_COLLECTION_NAME} created successfully!")
            embed_existing_images(connector)
        else:
            logger.info("Collection already exists, skipping creation")

def clear_collection():
    logger.info("clear_collection called")
    img_fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='path', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512)
    ]

    with connect_with_retry() as connector:
        connector.create_collection(config.IMAGES_COLLECTION_NAME, img_fields, remove_if_exists=True)
        logger.info("Collection cleared and recreated")
        embed_existing_images(connector)

def check_and_sync_images():
    """
    Check if images from the images folder are embedded in the database.
    If not, clean the database and embed all images as startup examples.
    """
    try:
        with connect_with_retry() as connector:
            # Check if collection exists
            if not connector.check_if_collection_exists(config.IMAGES_COLLECTION_NAME):
                logger.info("Database collection not found. Creating collection and embedding startup images...")
                ensure_collection_exists()
                return
            
            # Get all image files from the images folder
            image_files = []
            if os.path.exists(config.IMAGES_DIR):
                for file in os.listdir(config.IMAGES_DIR):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_files.append(file)
            
            if not image_files:
                logger.info("No images found in images folder. Database is ready for new images.")
                return
            
            # Get all paths from the database
            collection = connector.get_collection(config.IMAGES_COLLECTION_NAME)
            collection.load()
            
            # Query all paths from the database
            results = collection.query(
                expr="",
                output_fields=["path"],
                limit=10000  # Adjust if you expect more images
            )
            
            db_paths = set()
            if results:
                for result in results:
                    path = result.get('path', '')
                    if path:
                        # Extract just the filename from the path
                        filename = os.path.basename(path)
                        db_paths.add(filename)
            
            # Check if all image files are in the database
            folder_files = set(image_files)
            missing_files = folder_files - db_paths
            
            if missing_files:
                logger.info(f"Found {len(missing_files)} new images. Cleaning database and re-embedding all images...")
                # Clean the database and re-embed all images
                clear_collection()
            else:
                logger.info(f"Database is synchronized with {len(image_files)} images from the images folder.")
                
    except Exception as e:
        logger.error(f"Error checking database synchronization: {e}")
        # If there's an error, try to recreate the collection
        logger.info("Attempting to recreate database collection...")
        clear_collection()
        ensure_collection_exists()

def delete_images_folder(folder_path="images"):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)  # Delete the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the subfolder and its contents
            except OSError as e:
                logger.error(f"Error deleting {file_path}: {e}")
    else:
        logger.info(f"The folder '{folder_path}' does not exist or is not a directory.")


embedder = load_embedder()
logger.info(f"Config values: IMAGES_DIR={config.IMAGES_DIR}, IMAGES_COLLECTION_NAME={config.IMAGES_COLLECTION_NAME}")
initialize_database()

st.title("Image Search App")

st.sidebar.title("Navigation")
navbar = st.sidebar.radio(
    "Go to:",
    ["Search engine", "Populate Database"])
if st.sidebar.button("Clear Database"):
    delete_images_folder()
    clear_collection()
    st.sidebar.success("Database has been cleared and recreated!")


if navbar == "Search engine":
    st.header("Search Images via inputted text")
    query = st.text_input("Enter a text query to search for images:")
    if time.time() - st.session_state.last_input_time > 1 and not st.session_state.search_triggered:
        if query:
            with connect_with_retry() as connector:
                results = connector.search_threshold(
                    embedder.embed_sentences(query),
                    output_field='path',
                    k=100,
                    threshold=240,
                    collection_name=config.IMAGES_COLLECTION_NAME
                )
            if results:
                st.subheader("Search Results")
                num_columns = 3
                rows = [results[i:i + num_columns] for i in range(0, len(results), num_columns)]

                for row in rows:
                    cols = st.columns(num_columns)
                    for col, (path, dist) in zip(cols, row):
                        with col:
                            result_image = Image.open(path)
                            st.image(result_image, caption=f"Distance: {dist:.2f}", use_container_width=True)
            else:
                st.warning("No results found!")
        else:
            st.error("Please enter a query to search.")
    if query != "" and time.time() - st.session_state.last_input_time < 1:
        st.session_state.search_triggered = False
    st.header("Search Images via uploaded file")

    uploaded_image = st.file_uploader(
        "Upload an image to search for similar images:",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image and st.button("Search by Image"):
        image = Image.open(uploaded_image)
        st.image(image,
            caption="Uploaded Image",
            use_container_width=True
        )

        with connect_with_retry() as connector:
            embeddings = embedder.embed_images(image)
            results = connector.search_threshold(
                embeddings,
                output_field='path', 
                k=20,
                threshold=150,
                collection_name=config.IMAGES_COLLECTION_NAME
            )

        if results:
            st.subheader("Search Results")
            num_columns = 3
            rows = [results[i:i + num_columns] for i in range(0, len(results), num_columns)]

            for row in rows:
                cols = st.columns(num_columns)
                for col, (path, dist) in zip(cols, row):
                    with col:
                        result_image = Image.open(path)
                        st.image(result_image, caption=f"Distance: {dist:.2f}", use_container_width=True)
        else:
            st.warning("No results found!")


elif navbar == "Populate Database":
    st.header("Add New Image")

    with st.form("my-form", clear_on_submit=True):
        uploaded_images = st.file_uploader(
            "Upload an image to add to the database",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        submitted = st.form_submit_button("Add Images")

        if submitted and uploaded_images is not None:
            if uploaded_images:
                paths = []
                embeddings = []
                successes = 0
                failures = 0
                for uploaded_image in uploaded_images:
                    try:
                        img = Image.open(uploaded_image)
                        
                        image_path = os.path.join(config.IMAGES_DIR, uploaded_image.name)
                        img.save(image_path)
                        paths.append(image_path)
                        
                        embedding = embedder.embed_images(img).to('cpu').tolist()[0]
                        embeddings.append(embedding)
                        
                        successes += 1
                    except Exception as e:
                        st.error(f"Failed to process {uploaded_image.name}: {e}")
                        failures += 1
                
                try:
                    with connect_with_retry() as connector:
                        batch_data = [{'path': path, 'embedding': embedding} for path, embedding in zip(paths, embeddings)]
                        connector.insert(batch_data, collection_name=config.IMAGES_COLLECTION_NAME)
                    
                    st.success(f"Successfully added {successes} images to the database.")
                except Exception as e:
                    st.error(f"Failed to insert images into the database: {e}")

                if failures > 0:
                    st.error(f"Failed to add {failures} images. Check the error messages above.")
              
            else:
                st.error("Please upload at least one image to add.")

st.sidebar.write("---")                                              
st.sidebar.write("Developed for Natural Language Processing Class.")