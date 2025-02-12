# ViSEO - SEO-Optimized Product Description Generation

## 📌 Overview
ViSEO is an AI-powered system designed to automate the generation of SEO-optimized product descriptions for Vietnamese e-commerce platforms. The system leverages **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**, integrating multiple data sources, keyword research, and vector search techniques to create high-quality, optimized content.

## 🔥 Features
- **SEO Keyword Research**: Automatically extracts and categorizes SEO-friendly keywords from **SerpAPI**, **Wordtracker**, and AI models.
- **Product Research**: Uses **Tavily Search API** to fetch the latest product-related information.
- **Content Generation**: Generates structured and SEO-optimized product descriptions using **GPT-4o**, **Gemini**, and **LangChain**.
- **Vector Search with MongoDB Atlas**: Enhances retrieval efficiency for relevant product information.
- **Data Indexing & Embedding**: Stores and indexes product descriptions and topics using **FAISS-like vector indexing** in MongoDB.
- **Multi-Agent Orchestration**: Implements **LangGraph** for efficient multi-agent task handling.

## 🛠 Tech Stack
- **Backend**: Python, Flask
- **AI & NLP**: OpenAI GPT-4o, Gemini, Sentence Transformers
- **Database**: MongoDB Atlas (Vector Search)
- **APIs**: SerpAPI, Wordtracker API, Tavily Search API
- **Frameworks**: LangChain, LangGraph

## 🚀 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ViSEO.git
cd ViSEO
```

### 2️⃣ Set Up Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables
Create a `.env` file in the root directory and add:
```ini
OPENAI_API_KEY=your_openai_key
MONGODB_URI=your_mongo_uri
SERPAPI_KEY=your_serpapi_key
WORDTRACKER_APP_ID=your_wordtracker_id
WORDTRACKER_APP_KEY=your_wordtracker_key
TAVILY_API_KEY=your_tavily_key
```

## 🏃‍♂️ Usage
### Running the API Server
```bash
python app.py
```
The server will start at `http://localhost:5000`.

### Uploading Product Data (via API)
Send a **POST** request with an Excel file containing product names:
```bash
curl -X POST -F "file=@products.xlsx" http://localhost:5000/upload
```

### Performing a Search Query
```bash
curl -X GET "http://localhost:5000/search?query=Samsung"
```
