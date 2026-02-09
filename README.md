# Visual Product Matcher

A web application that helps users find visually similar products based on an uploaded image or image URL.  
This project was built as part of a technical assessment to demonstrate full-stack development and applied computer vision.

---

## 🚀 Features

- Upload an image **or** provide an image URL
- View visually similar products ranked by similarity score
- Filter results using a minimum similarity threshold
- Responsive and clean user interface
- REST API with interactive Swagger documentation

---

## 🧠 Technical Approach

The backend is built using **FastAPI** and leverages a pretrained **ResNet-50** convolutional neural network for image feature extraction.  
Each product image is converted into a fixed-length embedding vector. When a user submits a query image, its embedding is computed and compared with stored product embeddings using **cosine similarity** to find visually similar items.

The frontend is implemented using **HTML, CSS, and vanilla JavaScript**, providing a lightweight and responsive interface that communicates with the backend via HTTP requests.

---

## 🗂️ Project Structure

visual-product-matcher/
├── main.py # FastAPI backend
├── products.csv # Product dataset (60+ products)
├── requirements.txt # Python dependencies
├── frontend/
│ └── index.html # Frontend UI
└── README.md


---

## 🛠️ Running Locally

### Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload

Visit:

http://127.0.0.1:8000/docs

Frontend

Open frontend/index.html directly in your browser.

🌐 Deployment

Backend: Deployed on Render

Frontend: Deployed on Vercel

(Links provided in the project submission.)


📦 Dataset

60+ products

Public image URLs

Includes basic metadata such as name and category




Project Explanation 

I built a Visual Product Matcher web application that enables users to find visually similar products using an uploaded image or an image URL. The objective of the project was to design a practical, end-to-end system that combines computer vision with a clean, user-friendly web interface.

The backend is implemented using FastAPI and leverages a pretrained convolutional neural network (ResNet-18) for visual feature extraction. The model’s classification head is removed so that it outputs high-level image embeddings rather than class predictions. All product images in the dataset are processed to generate embeddings, and when a user submits a query image, its embedding is compared against stored product embeddings using cosine similarity to identify visually similar products.

The product database contains over 60 products, each with a public image URL and basic metadata such as product name and category. Basic error handling is included to manage invalid image URLs, failed image downloads, and processing errors.

The frontend is developed using HTML, CSS, and vanilla JavaScript, providing a lightweight and responsive user experience. Users can upload images, preview inputs, filter results by similarity score, and view ranked results in real time. The backend is deployed on Render and the frontend on Vercel, making the application fully accessible online.