import streamlit as st
import time
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


def embed_existing_images(connector: Optional[MilvusConnector] = None): 
    st.warning("Adding images from the folder to the collection...")
    images = []
    for image in os.listdir(config.IMAGES_DIR):
        path = os.path.join(config.IMAGES_DIR, image)
        images.append({
            'path': path,
            'image': Image.open(path),
            'embedding': None
        })
        
    batch = []
    for img in images:
        img['embedding'] = embedder.embed_images(img['image']).to('cpu').tolist()[0]
        batch.append({'path': img['path'], 'embedding': img['embedding']})

    if connector:
        connector.insert(batch, collection_name='images')
    else:
        with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
            connector.insert(batch, collection_name='images')
    st.success("Images added successfully!")


def ensure_collection_exists():
    img_fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='path', dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512)
    ]
    with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
        if not connector.check_if_collection_exists(config.IMAGES_COLLECTION_NAME):
            st.warning("Image collection not found. Creating the collection...")
            connector.create_collection(config.IMAGES_COLLECTION_NAME, img_fields, remove_if_exists=False)
            st.success(f"Collection {config.IMAGES_COLLECTION_NAME} created successfully!")
            embed_existing_images(connector)

def clear_collection():
    img_fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='path', dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512)
    ]

    with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
        connector.create_collection(config.IMAGES_COLLECTION_NAME, img_fields, remove_if_exists=True)

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
                print(f"Error deleting {file_path}: {e}")
    else:
        print(f"The folder '{folder_path}' does not exist or is not a directory.")


embedder = load_embedder()
ensure_collection_exists()

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
            with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
                results = connector.search_threshold(
                    embedder.embed_sentences(query),
                    output_field='path',
                    k=100,
                    threshold=240
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

        with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
            embeddings = embedder.embed_images(image)
            results = connector.search_threshold(
                embeddings,
                output_field='path', 
                k=20,
                threshold=150
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
                    with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
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