import os
import pandas as pd
import logging
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_PATH = os.path.join(ROOT_DIR, "data/dataset/")
os.makedirs(STORAGE_PATH, exist_ok=True)

# MongoDB setup
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["test_database"]
indexed_collection = db["indexed_data"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize LLM
OPENAI_API = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=3000,
    openai_api_key=OPENAI_API,
)

# def vector_search(query):
#     """
#     Retrieve relevant context from the database using both product-level and chunk-level embeddings.
#     """
#     try:
#         query_vector = embedding_model.encode(query)
#         product_pipeline = [
#             {
#                 "$vectorSearch": {
#                     "index": "product_vector_index",
#                     "path": "product_embedding",
#                     "queryVector": query_vector.tolist(),
#                     "numCandidates": 20,
#                     "limit": 1
#                 }
#             },
#             {"$project": {"_id": 0, "product_name": 1, "score": {"$meta": "vectorSearchScore"}}}
#         ]

#         product_results = list(indexed_collection.aggregate(product_pipeline))
#         if not product_results:
#             logging.warning("No matching products found.")
#             return ""

#         matching_products = list(set(result["product_name"] for result in product_results))

#         chunk_pipeline = [
#             {"$match": {"product_name": {"$in": matching_products}}},
#             {"$project": {"_id": 0, "chunk": 1}}
#         ]
#         chunk_results = list(indexed_collection.aggregate(chunk_pipeline))

#         context = " ".join(result["chunk"] for result in chunk_results)
#         logging.info(f"Retrieved Chunks: {context}")
#         return context
#     except Exception as e:
#         logging.error(f"Error in vector search: {e}")
#         return ""

def vector_search(query):
    """
    Retrieve relevant context from the database using both product-level and chunk-level embeddings.
    Return a list of contexts as required by RAGAS.
    """
    try:
        query_vector = embedding_model.encode(query)
        product_pipeline = [
            {
                "$vectorSearch": {
                    "index": "product_vector_index",
                    "path": "product_embedding",
                    "queryVector": query_vector.tolist(),
                    "numCandidates": 20,
                    "limit": 1
                }
            },
            {"$project": {"_id": 0, "product_name": 1, "score": {"$meta": "vectorSearchScore"}}}
        ]

        product_results = list(indexed_collection.aggregate(product_pipeline))
        if not product_results:
            logging.warning("No matching products found.")
            return []  # Return an empty list for no matches

        matching_products = list(set(result["product_name"] for result in product_results))

        chunk_pipeline = [
            {"$match": {"product_name": {"$in": matching_products}}},
            {"$project": {"_id": 0, "chunk": 1}}
        ]
        chunk_results = list(indexed_collection.aggregate(chunk_pipeline))

        # Ensure each chunk is a string and construct a list
        contexts = [result["chunk"] for result in chunk_results if isinstance(result["chunk"], str)]
        logging.info(f"Retrieved Chunks: {contexts}")
        return contexts  # Return list of contexts
    except Exception as e:
        logging.error(f"Error in vector search: {e}")
        return []  # Return an empty list in case of error


# def generate_seo_tool(query, combined_chunks):
#     """
#     Generate SEO content using LLM based on query and chunks.
#     """
#     prompt = f"""
#     Viết mô tả cho sản phẩm sau, đảm bảo tối ưu SEO dài khoảng 1000 đến 1200 từ, dựa trên các thông tin sau:
#     **Phân đoạn thông tin sản phẩm**: {combined_chunks}
#     **Query**: {query}
#     **Yêu cầu cụ thể**:
#             1. Tự phân tích và xác định:
#                 - **Từ khóa chính** (Primary Keyword)
#                 - **Từ khóa phụ** (Secondary Keywords)
#                 - **Từ khóa đuôi dài** (Long-tail Keywords)
#                 - **Câu hỏi thường gặp** (Question-based Keywords)

#             2. Viết bài chuẩn SEO với cấu trúc như sau:
#                 - **Page Title**: Ngắn gọn, hấp dẫn, tối đa 60 ký tự, chứa từ khóa chính và đặt ở đầu tiêu đề.  
#                 - **Meta Description**: Viết 1-2 câu mô tả ngắn gọn, súc tích (tối đa 160 ký tự), tập trung vào lợi ích của sản phẩm và chứa từ khóa chính.
#                 - **Tiêu đề (H1)**: Bao gồm từ khóa chính.
#                 - **Phần mở bài**: Hấp dẫn và thu hút người đọc.
#                 - **Thân bài**: Miêu tả chi tiết công năng, lợi ích, đặc điểm nổi bật của sản phẩm.
#                 - **Phần FAQ**: Bao gồm câu hỏi thường gặp và câu trả lời.
#                 - **Kết bài**: Kêu gọi hành động mạnh mẽ.

#         ### Hướng dẫn viết bài theo chuẩn SEO:
#         - Chỉ xuất ra **nội dung bài viết**, không thêm phần giải thích hay nhận xét.
#         - Nhấn mạnh các từ khóa quan trọng bằng cách sử dụng **in đậm** một cách tự nhiên.
#         - Sử dụng **Markdown** để định dạng bài viết: tiêu đề, danh sách gạch đầu dòng (UL/OL), và trích dẫn (block quotes).
#         - Nội dung bài viết phải đáp ứng các tiêu chuẩn sau:

#         ### Tối ưu chuẩn E-E-A-T:
#         - **Trải nghiệm (Experience)**: 
#             - Lồng ghép các ví dụ thực tế hoặc kinh nghiệm sử dụng sản phẩm để tăng tính xác thực và hấp dẫn.
#         - **Chuyên môn (Expertise)**: 
#             - Nội dung cần thể hiện kiến thức sâu rộng và chính xác về sản phẩm, nhấn mạnh các khía cạnh kỹ thuật hoặc đặc trưng nổi bật của sản phẩm.
#         - **Tính thẩm quyền (Authoritativeness)**: 
#             - Đảm bảo thông tin được dẫn dắt bởi nguồn đáng tin cậy, ví dụ: trích dẫn nghiên cứu, chuyên gia, hoặc khách hàng có uy tín.
#         - **Độ tin cậy (Trustworthiness)**: 
#             - Trình bày minh bạch, đảm bảo nội dung chính xác, rõ ràng, và tránh gây hiểu lầm. Đề xuất cách kiểm chứng thông tin (liên kết nguồn uy tín nếu cần).

#         ### Tối ưu SEO:
#         - Từ khóa chính phải xuất hiện trong:
#             - - **Page Title**, **Meta Description**, **Tiêu đề**, **mô tả meta SEO**, **URL**, và trong **10% đầu tiên** của nội dung.
#             - Các thẻ **H2, H3, H4**, thuộc tính **alt của hình ảnh**, và phân bổ xuyên suốt nội dung (mật độ từ khóa từ **1% - 1.5%**).
#         - URL không dài quá **75 ký tự**.
#         - Sử dụng định dạng **in đậm** cho từ khóa chính để làm nổi bật nội dung quan trọng.
#         - Đa dạng hóa cách trình bày để cải thiện trải nghiệm đọc:
#             - Dùng `gạch đầu dòng`, `danh sách số thứ tự`, và `trích dẫn` khi cần.
#             - Tiêu đề nên có con số hoặc cụm từ mang tính hứa hẹn để tăng mức độ tương tác.

#         ### Nội dung bài viết:
#         - **Page Title**: Bắt đầu với từ khóa chính, không vượt quá 60 ký tự.
#         - **Meta Description**: Tập trung làm nổi bật lợi ích của sản phẩm, không vượt quá 160 ký tự.
#         - Bắt đầu với một phần giới thiệu cuốn hút về sản phẩm.
#         - Giữ văn phong thân thiện, gần gũi, thuyết phục, như được viết bởi con người.
#         - Tuân thủ cấu trúc chuẩn SEO, tập trung cung cấp thông tin chi tiết, hữu ích.
#         - Sử dụng Markdown đúng cách:
#             - `#` cho tiêu đề chính (Heading 1), `##` cho tiêu đề phụ (Heading 2),...
#         - Mô tả **giá** của sản phẩm và **chính sách bảo hành**

#         **Lưu ý đặc biệt**: 
#         - Nội dung phải tuân thủ chuẩn E-E-A-T, nhấn mạnh trải nghiệm, chuyên môn, tính thẩm quyền và độ tin cậy.
#         - Đặc biệt chú trọng tính chính xác và độ tin cậy nếu sản phẩm thuộc các lĩnh vực YMYL (Your Money, Your Life) như sức khỏe, tài chính, hoặc pháp luật.
#         - Xuất ra nội dung định dạng Markdown hoàn chỉnh.
#     """
#     try:
#         human_message = HumanMessage(content=prompt)
#         response = llm.invoke([human_message])
#         generated_output = response.content.strip()
#         logging.info(f"Generated Output: {generated_output}")
#         return generated_output
#     except Exception as e:
#         logging.error(f"Error in generating SEO tool output: {e}")
#         return ""

# def process_queries(input_file, output_file):
#     """
#     Process queries from input file and save results to output file.
#     """
#     # Read input queries
#     input_df = pd.read_excel(input_file, engine='openpyxl')
#     queries = input_df['query'].tolist()

#     results = []
#     for query in queries:
#         logging.info(f"Processing query: {query}")
#         retrieved_chunks = vector_search(query)
#         # generated_output = generate_seo_tool(query, retrieved_chunks)
#         results.append({
#             "Query": query,
#             "Retrieved Chunks": retrieved_chunks,
#             # "Generated Output": generated_output
#         })

#     # Save results to output file
#     output_df = pd.DataFrame(results)
#     output_df.to_excel(output_file, index=False, engine='openpyxl')
#     logging.info(f"Results saved to {output_file}")

def process_queries(input_file, output_file):
    """
    Process queries from input file and save results to output file.
    Ensure retrieved contexts are in correct format for RAGAS.
    """
    # Read input queries
    input_df = pd.read_excel(input_file, engine='openpyxl')
    queries = input_df['query'].tolist()

    results = []
    for query in queries:
        logging.info(f"Processing query: {query}")
        retrieved_chunks = vector_search(query)
        # Ensure retrieved_chunks is a list of strings
        if not isinstance(retrieved_chunks, list):
            logging.warning(f"Retrieved chunks for query '{query}' is not a list.")
            retrieved_chunks = []
        retrieved_chunks = [str(chunk) for chunk in retrieved_chunks if isinstance(chunk, str)]
        # Append results
        results.append({
            "Query": query,
            "Retrieved Chunks": "|||".join(retrieved_chunks),  # Join for saving to Excel
        })

    # Save results to output file
    output_df = pd.DataFrame(results)
    output_df.to_excel(output_file, index=False, engine='openpyxl')
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_file = os.path.join(STORAGE_PATH, "queries_product_description.xlsx")
    output_file = os.path.join(STORAGE_PATH, "evaluation_results.xlsx")

    process_queries(input_file, output_file)
