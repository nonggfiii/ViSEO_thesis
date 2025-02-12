# from flask import request, render_template, jsonify
# import pandas as pd
# import logging
# from pymongo import MongoClient
# from src.config.global_settings import MONGO_URI
# from src.data_collection_new import run_pipeline
# from src.data_indexing_new import process_and_index_data

# # Logging configuration
# logging.basicConfig(level=logging.INFO)

# # Kết nối MongoDB
# try:
#     client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
#     db = client["eval_database"]
#     collection = db["test_collection"]
#     indexed_collection = db["indexed_data_openai_text3"]
#     print("Kết nối MongoDB thành công!")
# except Exception as e:
#     print(f"Kết nối MongoDB thất bại: {e}")

# async def upload_file(request):
#     if request.method == "POST":
#         if "file" not in request.files:
#             logging.error("No file part")
#             return jsonify({"error": "No file part"}), 400
        
#         file = request.files["file"]
#         if file.filename == "":
#             logging.error("No selected file")
#             return jsonify({"error": "No selected file"}), 400

#         if file:
#             try:
#                 # Đọc file Excel và chuyển thành JSON
#                 logging.info("Đang đọc file Excel...")
#                 df = pd.read_excel(file)
#                 data = df.to_dict(orient="records")
#                 logging.info(f"Đọc thành công {len(data)} bản ghi từ file Excel.")

#                 # Chuyển đổi key "Tên sản phẩm" thành "product_name"
#                 for item in data:
#                     product_name = item.get("Tên sản phẩm")
#                     if not product_name:
#                         logging.warning("Không có tên sản phẩm, bỏ qua.")
#                         continue
#                     item["product_name"] = product_name  # Chuẩn hóa key

#                     # Lưu vào MongoDB nếu chưa tồn tại
#                     if not collection.find_one({"product_name": product_name}):
#                         collection.insert_one(item)
#                         logging.info(f"Đã thêm sản phẩm mới: {product_name}")
#                     else:
#                         logging.info(f"Sản phẩm đã tồn tại: {product_name}")

#                 logging.info("Đã lưu dữ liệu gốc vào MongoDB.")

#                 # Gọi pipeline để xử lý dữ liệu
#                 enriched_data = []
#                 for item in data:
#                     product_name = item.get("product_name")  # Dùng key mới
#                     if not product_name:
#                         continue

#                     # **Gọi `run_pipeline()` lấy content_topics và product_report**
#                     result = await run_pipeline(product_name)

#                     # **Hợp nhất kết quả vào item**
#                     item["content_topics"] = result.get("content_topics", [])
#                     item["product_report"] = result.get("product_report", "")

#                     enriched_data.append(item)

#                 # **Gọi `process_and_index_data()` với enriched_data**
#                 logging.info("Bắt đầu xử lý chunking, embedding và indexing...")
#                 indexed_data = process_and_index_data(enriched_data)

#                 # **Lưu dữ liệu đã index vào MongoDB**
#                 if indexed_data:
#                     logging.info(f"Lưu {len(indexed_data)} mục dữ liệu vào indexed_collection.")
#                     indexed_collection.insert_many(indexed_data)
#                     logging.info("Dữ liệu đã được lưu vào indexed_collection.")
#                 else:
#                     logging.warning("Không có dữ liệu nào để index.")

#                 return jsonify({"message": "Dữ liệu đã được xử lý và lưu vào MongoDB."}), 200

#             except Exception as e:
#                 logging.error(f"Lỗi khi xử lý file Excel: {e}")
#                 return jsonify({"error": f"Lỗi khi xử lý file Excel: {e}"}), 500

#     return render_template("index.html")

from flask import request, render_template, jsonify
import pandas as pd
import logging
from pymongo import MongoClient
from src.config.global_settings import MONGO_URI
from src.data_collection_new import run_pipeline
from src.data_indexing_new import process_and_index_data

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Kết nối MongoDB
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client["eval_database"]
    collection = db["test_collection"]
    indexed_collection = db["indexed_data_openai_text3"]
    print("Kết nối MongoDB thành công!")
except Exception as e:
    print(f"Kết nối MongoDB thất bại: {e}")

async def upload_file(request):
    if request.method == "POST":
        if "file" not in request.files:
            logging.error("No file part")
            return jsonify({"error": "No file part"}), 400
        
        file = request.files["file"]
        if file.filename == "":
            logging.error("No selected file")
            return jsonify({"error": "No selected file"}), 400

        if file:
            try:
                # Đọc file Excel và chuyển thành JSON
                logging.info("📌 Đang đọc file Excel...")
                df = pd.read_excel(file)
                data = df.to_dict(orient="records")
                logging.info(f"✅ Đọc thành công {len(data)} bản ghi từ file Excel.")

                # Chuyển đổi key "Tên sản phẩm" thành "product_name"
                for item in data:
                    product_name = item.get("Tên sản phẩm")
                    if not product_name:
                        logging.warning("⚠️ Không có tên sản phẩm, bỏ qua.")
                        continue
                    item["product_name"] = product_name  # Chuẩn hóa key

                    # Lưu vào MongoDB nếu chưa tồn tại
                    if not collection.find_one({"product_name": product_name}):
                        collection.insert_one(item)
                        logging.info(f"🟢 Đã thêm sản phẩm mới vào MongoDB: {product_name}")
                    else:
                        logging.info(f"🔹 Sản phẩm đã tồn tại: {product_name}")

                logging.info("✅ Đã lưu dữ liệu gốc vào MongoDB.")

                # Gọi pipeline để xử lý dữ liệu
                enriched_data = []
                for item in data:
                    product_name = item.get("product_name")  # Dùng key mới
                    if not product_name:
                        continue

                    # **Gọi `run_pipeline()` lấy content_topics và product_report**
                    logging.info(f"\n🔎 **Đang chạy pipeline cho sản phẩm:** {product_name}")
                    result = await run_pipeline(product_name)

                    # **In kết quả từng bước**
                    print("\n📌 **Kết quả Pipeline:**")
                    print(f"🔹 **Tên sản phẩm:** {product_name}")
                    print(f"🔹 **Content Topics:** {result.get('content_topics', [])}")
                    print(f"🔹 **Product Report:** {result.get('product_report', 'N/A')}")
                    print(f"🔹 **Categorized Keywords:** {result.get('categorized_keywords', {})}\n")

                    logging.info(f"✅ Pipeline hoàn thành cho sản phẩm: {product_name}")

                    # **Hợp nhất kết quả vào item**
                    item["content_topics"] = result.get("content_topics", [])
                    item["product_report"] = result.get("product_report", "")

                    enriched_data.append(item)

                # **Gọi `process_and_index_data()` với enriched_data**
                logging.info("\n🚀 **Bắt đầu xử lý chunking, embedding và indexing...**")
                indexed_data = process_and_index_data(enriched_data)

                # **Lưu dữ liệu đã index vào MongoDB**
                if indexed_data:
                    logging.info(f"✅ Đã index {len(indexed_data)} mục dữ liệu vào MongoDB.")
                    indexed_collection.insert_many(indexed_data)
                else:
                    logging.warning("⚠️ Không có dữ liệu nào để index.")

                return jsonify({"message": "Dữ liệu đã được xử lý và lưu vào MongoDB."}), 200

            except Exception as e:
                logging.error(f"❌ Lỗi khi xử lý file Excel: {e}")
                return jsonify({"error": f"Lỗi khi xử lý file Excel: {e}"}), 500

    return render_template("index.html")
