"""
Simple 100K Product Ingestion - No PostgreSQL, just Qdrant
"""
import json
import time
import os
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Configuration
JSONL_PATH = Path(__file__).parent.parent / "data" / "meta_Electronics.jsonl"
LIMIT = 20000
BATCH_SIZE = 2000
COLLECTION_NAME = "products_100k"

def create_text(product):
    name = product.get('title', '')
    cat = product.get('main_category', 'electronics')
    brand = product.get('store', 'Generic')
    desc = product.get('description', [])
    if isinstance(desc, list) and desc:
        desc_text = str(desc[0])[:200]
    else:
        desc_text = str(desc)[:200] if desc else ''
    return f"{name} {cat} {brand} {desc_text}"

def parse_product(data):
    try:
        price = data.get('price')
        if not price or price == 'None':
            return None
        price = float(price)
        if price < 1 or price > 50000:
            return None
        
        title = data.get('title', '')
        if not title or len(title) < 5:
            return None
        
        # Extract image URL
        images = data.get('images', [])
        image_url = ''
        if images and isinstance(images, list):
            first_img = images[0]
            if isinstance(first_img, dict):
                image_url = first_img.get('large', first_img.get('thumb', ''))
            elif isinstance(first_img, str):
                image_url = first_img
        
        return {
            'id': data.get('parent_asin', ''),
            'name': title,
            'category': data.get('main_category', 'electronics'),
            'brand': data.get('store', 'Generic'),
            'price': price,
            'rating': float(data.get('average_rating', 0) or 0),
            'rating_count': int(data.get('rating_number', 0) or 0),
            'image_url': image_url,
        }
    except:
        return None

def main():
    print(f"Loading {LIMIT} products from {JSONL_PATH}")
    
    # Load products
    products = []
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= LIMIT:
                break
            try:
                data = json.loads(line)
                p = parse_product(data)
                if p:
                    products.append(p)
            except:
                continue
    
    print(f"Loaded {len(products)} valid products")
    
    # Generate embeddings
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating embeddings...")
    texts = [create_text(p) for p in products]
    embeddings = model.encode(texts, batch_size=128, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Connect to Qdrant
    print("Connecting to Qdrant...")
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'),
        api_key=os.getenv('QDRANT_API_KEY'),
        timeout=180
    )
    
    # Create collection
    print(f"Creating collection: {COLLECTION_NAME}")
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    # Upload in batches
    print(f"Uploading {len(products)} products...")
    start_time = time.time()
    
    for i in range(0, len(products), BATCH_SIZE):
        batch = products[i:i+BATCH_SIZE]
        batch_emb = embeddings[i:i+BATCH_SIZE]
        
        points = [
            PointStruct(
                id=i+j,
                vector=batch_emb[j].tolist(),
                payload=batch[j]
            )
            for j in range(len(batch))
        ]
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        elapsed = time.time() - start_time
        rate = (i + len(batch)) / elapsed
        print(f"  Uploaded {i + len(batch)}/{len(products)} ({rate:.1f} prod/s)")
    
    # Create indexes
    from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType
    
    print("Creating indexes...")
    for field in ['category', 'brand']:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD
            )
        except:
            pass
    
    for field in ['price', 'rating']:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=PayloadSchemaType.FLOAT
            )
        except:
            pass
    
    # Verify
    info = client.get_collection(COLLECTION_NAME)
    elapsed = time.time() - start_time
    print(f"\nDone! {info.points_count} points in {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()