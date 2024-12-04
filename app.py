import streamlit as st
from typing import Optional
from PIL import Image
import os
from io import BytesIO
from classes.embedder import Embedder
from classes.milvus_connector import MilvusConnector
from classes.utils import display_results, load_or_save_model
from classes.config_loader import config
from pymilvus import FieldSchema, DataType

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
            



embedder = load_embedder()
ensure_collection_exists()

st.title("Image Search App")
st.sidebar.header("Options")

st.header("Search Images")
query = st.text_input("Enter a text query to search for images:")
if st.button("Search"):
    if query:
        with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
            results = connector.search_topk(
                embedder.embed_sentences(query), output_field='path', k=4
            )
        if results:
            st.subheader("Search Results")
            for path, dist in results:
                img = Image.open(path)
                st.image(img, caption=f"Distance: {dist:.2f}", use_container_width=True)
        else:
            st.warning("No results found!")
    else:
        st.error("Please enter a query to search.")

st.header("Add New Image")
uploaded_image = st.file_uploader("Upload an image to add to the database", type=["jpg", "jpeg", "png"])
if st.button("Add Image"):
    if uploaded_image:
        img = Image.open(uploaded_image)
        image_path = os.path.join(config.IMAGES_DIR, uploaded_image.name)
        img.save(image_path)
        embedding = embedder.embed_images(img).to('cpu').tolist()[0]
        with MilvusConnector(host=config.MILVUS_HOST, port=config.MILVUS_PORT) as connector:
            connector.insert(
                [{'path': image_path, 'embedding': embedding}], collection_name=config.IMAGES_COLLECTION_NAME
            )
        st.success(f"Image {uploaded_image.name} added to the database.")
    else:
        st.error("Please upload an image to add.")

st.sidebar.write("---")
st.sidebar.write("Developed for Natural Language Processing Class.")
