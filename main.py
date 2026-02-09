from fastapi import FastAPI, UploadFile, File
import pandas as pd
import os
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np

app = FastAPI()

# ---------- Load CSV ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "products.csv")
products_df = pd.read_csv(CSV_PATH)

# ---------- Load ResNet (light & stable) ----------
resnet = models.resnet50(pretrained=True)
resnet.eval()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

# ---------- Image preprocessing ----------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- Image → embedding ----------
def get_image_embedding(image_url: str):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(image_url, headers=headers, timeout=5)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = resnet(image_tensor)

    features = features.squeeze().numpy()
    features = features / np.linalg.norm(features)
    return features
def get_image_embedding_from_file(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        features = resnet(image_tensor)

    features = features.squeeze().numpy()
    features = features / np.linalg.norm(features)
    return features


# ---------- Precompute embeddings ----------
product_embeddings = []

for idx, row in products_df.iterrows():
    try:
        emb = get_image_embedding(row["image_url"])
        product_embeddings.append(emb)
    except Exception:
        product_embeddings.append(None)

# ---------- Cosine similarity ----------
def cosine_similarity(v1, v2):
    return float(np.dot(v1, v2))

# ---------- Routes ----------
@app.get("/")
def home():
    return {
        "message": "Visual Product Matcher API running",
        "total_products": int(len(products_df))
    }

@app.get("/test-embedding")
def test_embedding():
    url = products_df.iloc[0]["image_url"]
    emb = get_image_embedding(url)
    return {
        "embedding_length": int(len(emb))
    }

@app.post("/search-upload")
def search_upload(file: UploadFile = File(...), top_k: int = 5):
    query_emb = get_image_embedding_from_file(file)

    scores = []
    for idx, prod_emb in enumerate(product_embeddings):
        if prod_emb is None:
            continue

        score = cosine_similarity(query_emb, prod_emb)
        scores.append((idx, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in scores:
        product = products_df.iloc[idx]
        results.append({
            "id": int(product["id"]),
            "name": product["name"],
            "category": product["category"],
            "image_url": product["image_url"],
            "similarity": round(score, 3)
        })

    return {"results": results}


@app.get("/search")
def search(image_url: str, top_k: int = 5):
    query_emb = get_image_embedding(image_url)

    scores = []
    for idx, prod_emb in enumerate(product_embeddings):
        if prod_emb is None:
            continue
        score = cosine_similarity(query_emb, prod_emb)
        scores.append((idx, score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    results = []
    for idx, score in scores:
        product = products_df.iloc[idx]
        results.append({
            "id": int(product["id"]),
            "name": product["name"],
            "category": product["category"],
            "image_url": product["image_url"],
            "similarity": round(score, 3)
        })

    return {"results": results}

