import os
import json
from dotenv import load_dotenv
import logging
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

load_dotenv()

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

def embedding_model(text, model="text-embedding-3-small"):
    """
    Get embedding for a given text using OpenAI API.
    """
    try:
        response = client.embeddings.create(input=text, model=model)
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Lỗi khi tạo embedding: {e}")
        return None

def process_and_index_data(data):
    """
    Process, chunk, embed, and prepare data for MongoDB indexing.
    - Stores topics separately in `seo_data` with `doc_type: "topic"`
    - Stores chunked product details separately in `seo_data` with `doc_type: "chunk"`
    """
    if not data:
        logging.warning("Dữ liệu đầu vào rỗng!")
        return []

    indexed_data = []  # Danh sách dữ liệu để trả về

    try:
        logging.info(f"Đang xử lý {len(data)} sản phẩm cho indexing...")

        for item in data:
            product_name = item.get("product_name", "")
            content_topics = item.get("content_topics", [])
            product_report = item.get("product_report", "")

            if not product_name:
                logging.warning("Sản phẩm không có tên, bỏ qua mục này.")
                continue

            # Generate product-level embedding
            product_embedding = embedding_model(product_name)
            if not product_embedding:
                logging.warning(f"Không thể tạo embedding cho sản phẩm: {product_name}.")
                continue

            # Process each topic separately
            for topic in content_topics:
                topic_title = topic.get("title", "No Title")
                primary_keywords = topic.get("primary_keywords", [])
                secondary_keywords = topic.get("secondary_keywords", [])

                # Generate topic embedding
                topic_embedding = embedding_model(topic_title)

                if not topic_embedding:
                    logging.warning(f"Không thể tạo embedding cho topic '{topic_title}'.")
                    continue

                # Prepare topic document
                topic_data = {
                    "product_name": product_name,
                    "product_embedding": product_embedding,
                    "topic_title": topic_title,
                    "topic_embedding": topic_embedding,
                    "primary_keywords": primary_keywords,
                    "secondary_keywords": secondary_keywords
                }
                indexed_data.append(topic_data)

            # Chunk `product_report` and store in `seo_data`
            # if product_report:
            #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            #     chunks = text_splitter.split_text(product_report)

            #     for chunk in chunks:
            #         chunk_embedding = embedding_model(chunk)
            #         if not chunk_embedding:
            #             continue

            #         chunk_data = {
            #             "product_name": product_name,
            #             "product_embedding": product_embedding,
            #             "chunk": chunk,
            #             "chunk_embedding": chunk_embedding
            #         }
            #         indexed_data.append(chunk_data)
            # Kết hợp thông tin Giá và Chính sách bảo hành vào Product Report
            full_product_report = (
                f"Giá: {item.get('Giá', 'Không có thông tin')}\n"
                f"Chính sách bảo hành: {item.get('Chính sách bảo hành', 'Không có thông tin')}\n"
                f"Product Report:\n{product_report}"
            )
            # Tạo Embedding cho Product Report
            if full_product_report:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_text(full_product_report)

                for chunk in chunks:
                    chunk_embedding = embedding_model(chunk)
                    if not chunk_embedding:
                        continue

                    chunk_data = {
                        "product_name": product_name,
                        "product_embedding": product_embedding,
                        "chunk": chunk,
                        "chunk_embedding": chunk_embedding
                    }
                    indexed_data.append(chunk_data)

        logging.info(f"Hoàn tất indexing, tổng cộng {len(indexed_data)} mục dữ liệu.")
        return indexed_data  # Trả về dữ liệu đã xử lý
    
    except Exception as e:
        logging.error(f"Lỗi trong quá trình xử lý dữ liệu: {e}")
        return []

