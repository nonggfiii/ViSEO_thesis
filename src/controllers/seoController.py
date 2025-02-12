import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI 
from langchain.tools import tool
from langgraph.graph import StateGraph
import logging
from openai import OpenAI
import facebook
from flask import request,jsonify
import json
# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
print(os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN"))
print(os.getenv("FACEBOOK_PAGE_ID"))
FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_PAGE_ACCESS_TOKEN")
FACEBOOK_PAGE_ID = os.getenv("FACEBOOK_PAGE_ID")
logging.info(f"📢 DEBUG - FACEBOOK_ACCESS_TOKEN: {FACEBOOK_ACCESS_TOKEN}")
graph = facebook.GraphAPI(access_token=FACEBOOK_ACCESS_TOKEN)

# Connect to MongoDB
# MONGO_URI = os.getenv("MONGODB_URI")
# client = MongoClient(MONGO_URI)
# db = client["test_database"]
# indexed_collection = db["indexed_data"]

MONGO_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["eval_database"]
indexed_collection = db["indexed_data_openai_text3"]

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Load embedding model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# Initialize LLM for Supervisor
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=8000,
    openai_api_key=OPENAI_API,
    verbose=True
)
global_state = None

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


from langchain.schema import SystemMessage, HumanMessage

from pydantic import BaseModel
from typing import List

class GenerateSEOToolInput(BaseModel):
    product_name: str
    combined_chunks: str
    topic_title: str
    primary_keywords: List[str]
    secondary_keywords: List[str]
    platform: str = "website"

from langchain.schema import SystemMessage, HumanMessage


class Supervisor:
    def __init__(self):
        self.state = None

    def allocate_task(self):
        try:
            # 🔥 Tạo thông điệp cho LLM
            system_message = SystemMessage(
                content="""
                You are a supervisor tasked with managing tasks for multiple agents.
                Your job is to allocate the correct tasks to the appropriate agents based on the user query.

                ### Agents Available:
                1. **Website SEO Writer Agent**: Writes SEO content tailored for websites.
                2. **Facebook SEO Writer Agent**: Writes SEO content optimized for Facebook posts.
                3. **Publisher Agent**: Handles publishing content to specific platforms.

                ### Output Format:
                - Thought: Describe the reasoning.
                - Action: Specify the agent to invoke.
                - Action Input: JSON format for agent inputs.

                ### Valid Actions:
                - write_seo_website
                - write_seo_facebook
                - publish_content

                If you cannot determine the next step, respond:
                - Thought: "I am unable to determine the next step with the given inputs."
                - Action: "None"
                """
            )

            user_message = HumanMessage(
                content=f"""
                Given the query: "{self.state.query}", determine the next action based on the agents available.
                Current State: {self.state.__dict__}.
                """
            )

            # 🔥 Gọi LLM để quyết định Agent nào cần xử lý
            response = llm.generate([[system_message, user_message]])
            decision = response.generations[0][0].message.content.strip()
            logging.info(f"LLM Decision: {decision}")

            # 🛠 Kiểm tra quyết định và gọi Agent phù hợp
            if "write_seo_website" in decision:
                self.state.result = website_seo_writer_agent.write_seo(self.state)
                self.state.next_task = "end"

            elif "write_seo_facebook" in decision:
                self.state.result = facebook_seo_writer_agent.write_seo(self.state)

                # ✅ Đảm bảo `state.context` không bị rỗng
                if self.state.context is None:
                    self.state.context = {}

                # ✅ Lưu nội dung vào `state.context`
                self.state.context["last_facebook_post"] = {
                    "content": self.state.result,
                    "product_name": self.state.query
                }
                logging.info(f"✅ DEBUG: Đã lưu bài viết Facebook vào state.context: {self.state.context}")

                self.state.next_task = "end"

            elif "publish_content" in decision:
                # ✅ Kiểm tra `state.context["last_facebook_post"]`
                if self.state.context is None:
                    logging.error("❌ DEBUG: `state.context` bị mất hoặc không tồn tại.")
                    self.state.result = "Không có nội dung nào để đăng lên Facebook. Hãy yêu cầu tạo nội dung trước."
                    self.state.next_task = "end"
                    return

                last_post = self.state.context.get("last_facebook_post")

                if last_post:
                    logging.info("📢 DEBUG: Đang đăng bài lên Facebook...")

                    # ✅ Gọi `PublisherAgent` với đầy đủ `state.context`
                    self.state.result = publisher_agent.publish_content(self.state)
                else:
                    logging.error("❌ DEBUG: Không có nội dung nào để đăng lên Facebook.")
                    self.state.result = "Không có nội dung nào để đăng lên Facebook. Hãy yêu cầu tạo nội dung trước."

                self.state.next_task = "end"

            else:
                logging.error(f"Invalid task returned by LLM: {decision}")
                self.state.result = "Invalid task."
                self.state.next_task = "end"

        except Exception as e:
            logging.error(f"❌ Error in allocate_task: {e}")
            self.state.result = "An error occurred."
            self.state.next_task = "end"

    
    def run(self, query):
        """Giữ nguyên context khi chạy Supervisor nhiều lần."""
        if self.state is None:
            self.state = AgentState(query=query, next_task="start")
        else:
            self.state.query = query
            self.state.next_task = "start"

        logging.info(f"🔄 DEBUG - State khi bắt đầu run: {self.state.__dict__}")

        while self.state.next_task != "end":
            self.allocate_task()

            if not self.state.result:
                logging.error("❌ No result returned by the agent.")
                self.state.next_task = "end"



# Define AgentState
class AgentState:
    def __init__(self, query, next_task, context=None):
        self.query = query
        self.next_task = next_task
        self.result = None
        self.context = context if context else {}  


@tool("vector_search")
def vector_search(query):
    """
    Truy vấn MongoDB Atlas bằng vector search để lấy `topic_title`, `primary_keywords`, `secondary_keywords`, 
    và `chunk_texts` liên quan đến sản phẩm.
    """
    if not query:
        logging.error("❌ Không có query đầu vào cho vector search.")
        return {}

    try:
        # 🔹 Encode `query` thành vector
        query_vector = embedding_model(query)

        # 🔹 1️⃣ Tìm topic bằng `topic_embedding`
        topic_pipeline = [
            {
                "$vectorSearch": {
                    "index": "product_index_openai_text3",
                    "path": "topic_embedding",
                    "queryVector": query_vector,
                    "numCandidates": 5,
                    "limit": 1
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "product_name": 1,
                    "topic_title": 1,
                    "primary_keywords": 1,
                    "secondary_keywords": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]

        topic_results = list(indexed_collection.aggregate(topic_pipeline))

        # ❌ Nếu không tìm thấy topic, trả về lỗi
        if not topic_results:
            logging.warning("⚠️ Không tìm thấy topic phù hợp.")
            return {}

        # ✅ Chọn topic có điểm cao nhất
        selected_topic = topic_results[0]
        matching_product = selected_topic.get("product_name", "Không xác định")
        topic_title = selected_topic.get("topic_title", f"Hướng dẫn về {matching_product}")
        primary_keywords = selected_topic.get("primary_keywords", []) or [matching_product]
        secondary_keywords = selected_topic.get("secondary_keywords", []) or []

        logging.info(f"🟢 Tìm thấy topic: {topic_title} cho sản phẩm {matching_product}")

        # 🔹 2️⃣ Tìm `chunk_texts` trong `chunk_embedding` theo `product_name`
        chunk_pipeline = [
            {
                "$match": {
                    "product_name": matching_product
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "chunk": 1
                }
            }
        ]

        chunk_results = list(indexed_collection.aggregate(chunk_pipeline))

        # ✅ Nếu không tìm thấy chunk, gán giá trị mặc định
        if not chunk_results:
            logging.warning(f"⚠️ Không tìm thấy chunk nào cho sản phẩm {matching_product}.")
            chunk_texts = "Không có dữ liệu sản phẩm chi tiết."
        else:
            # Kiểm tra từng document xem có `chunk` không
            valid_chunks = [result.get("chunk", "") for result in chunk_results if "chunk" in result]

            if not valid_chunks:
                logging.warning(f"⚠️ Không có document nào chứa `chunk` cho sản phẩm {matching_product}.")
                chunk_texts = "Không có dữ liệu sản phẩm chi tiết."
            else:
                chunk_texts = " ".join(valid_chunks)

        logging.info(f"✅ Tìm thấy {len(chunk_results)} chunks.")

        # 🔹 3️⃣ Trả về context đầy đủ
        final_context = {
            "product_name": matching_product,
            "topic_title": topic_title,
            "primary_keywords": primary_keywords,
            "secondary_keywords": secondary_keywords,
            "chunk_texts": chunk_texts
        }

        logging.info(f"✅ Final context retrieved for {matching_product}")
        return final_context

    except Exception as e:
        logging.error(f"❌ Lỗi trong vector search: {e}")
        return {}

def search_topics_by_product_name(query):
    try:
        pipeline = [
            {
                "$search": {
                    "index": "topic_title_index",
                    "autocomplete": {
                        "query": query,
                        "path": "product_name"
                    }
                }
            },
            {
                "$match": { "topic_title": { "$exists": True } }
            },
            {
                "$project": {
                    "_id": 0,
                    "product_name": 1,
                    "topic_title": 1
                }
            }
        ]

        results = list(indexed_collection.aggregate(pipeline))

        if not results:
            return {"message": f"Không tìm thấy topic nào cho sản phẩm: {query}"}, 404

        return {"topics": results}, 200

    except Exception as e:
        return {"error": str(e)}, 500

import requests

SERPAPI_KEY = os.getenv("SERPAPI_KEY")

def fetch_product_images(product_name, num_images=4):
    """
    Fetch product images from SerpApi Google Images API.
    Returns a list of image URLs (up to `num_images`).
    """
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google_images",
            "q": product_name,
            "api_key": SERPAPI_KEY,
            "num": num_images  # Lấy số lượng ảnh cần
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Lọc ra tối đa `num_images` ảnh
        image_urls = [
            img["original"] for img in data.get("images_results", [])[:num_images]
        ]

        if image_urls:
            logging.info(f"Found {len(image_urls)} images for {product_name}.")
            return image_urls
        else:
            logging.warning(f"No images found for {product_name}.")
            return []

    except Exception as e:
        logging.error(f"Error fetching images for {product_name}: {e}")
        return []


@tool("generate_seo_tool", args_schema=GenerateSEOToolInput)
def generate_seo_tool(
    product_name: str,
    combined_chunks: str,
    topic_title: str,
    primary_keywords: List[str],
    secondary_keywords: List[str],
    platform: str = "website",
):
    """Generate SEO-optimized content for Website or Facebook based on product data and topics."""
        # **🔎 Nếu không tìm thấy Topic, AI sẽ tự động sinh Topic & Keywords**
    if not topic_title:
        logging.warning(f"Không tìm thấy topic phù hợp cho sản phẩm {product_name}. AI sẽ tự sinh topic...")

        generate_topic_prompt = f"""
        Bạn là một chuyên gia SEO. Hãy tạo một **Topic tiêu đề và bộ từ khóa** phù hợp để tối ưu SEO cho sản phẩm **{product_name}**.

        - **Topic Title**: Phải hấp dẫn, chứa thông tin sản phẩm, chuẩn SEO.
        - **Primary Keywords**: 3-5 từ khóa chính, có lượt tìm kiếm cao.
        - **Secondary Keywords**: 3-5 từ khóa bổ trợ.

        Xuất kết quả dưới JSON:
        ```json
        {{
            "topic_title": "Tên topic...",
            "primary_keywords": ["keyword1", "keyword2", "keyword3"],
            "secondary_keywords": ["related1", "related2", "related3"]
        }}
        ```
        """
        human_message = HumanMessage(content=generate_topic_prompt)
        response = llm.invoke([human_message])

        try:
            generated_data = json.loads(response.content.strip("```json").strip())
            topic_title = generated_data["topic_title"]
            primary_keywords = generated_data["primary_keywords"]
            secondary_keywords = generated_data["secondary_keywords"]
            logging.info(f"📌 AI sinh topic mới: {topic_title}")
        except Exception as e:
            logging.error(f"❌ Lỗi khi sinh topic bằng AI: {e}")
            topic_title = f"Tổng Quan về {product_name}"  # Fallback
            primary_keywords = [product_name]
            secondary_keywords = []

    # **🔎 Fetch product images**
    product_images = fetch_product_images(product_name)
    image_section = "\n\n".join([
        f"![{product_name} Image {i+1}]({img})"
        for i, img in enumerate(product_images)
    ]) if product_images else "Không có ảnh sản phẩm."

    try:
        # ✅ **Generate Website SEO Content**
        if platform == "website":
            prompt = f"""
            Bạn là một chuyên gia viết nội dung SEO. Hãy tạo bài viết tối ưu hóa SEO về sản phẩm **{product_name}**, có độ dài khoảng **2500-3000 từ**, dựa trên thông tin sau:

            **Topic**: {topic_title}
            **Primary Keywords**: {', '.join(primary_keywords)}
            **Secondary Keywords**: {', '.join(secondary_keywords)}

            **Thông tin sản phẩm**: {combined_chunks}
            **Hình ảnh sản phẩm**: {image_section}
            
            ## **Yêu cầu tối ưu SEO:**
            1. **Từ khóa quan trọng cần xuất hiện**:
            - **Primary Keywords**: Xuất hiện **5-7 lần**.
            - **Secondary Keywords**: Xuất hiện **3-5 lần**.
            - Từ khóa phải được sử dụng **tự nhiên**, không nhồi nhét.

            2. **Hướng dẫn chèn từ khóa**:
            - Từ khóa chính **phải có trong**:
                - **Tiêu đề chính (H1)**, **Meta Description**, **URL**, **10% đầu bài viết**.
                - **Tiêu đề phụ (H2, H3, H4)**, **alt của hình ảnh**, và phân bổ tự nhiên trong nội dung (**1% - 1.5% mật độ từ khóa**).
            - **Không lặp lại từ khóa y nguyên quá nhiều lần**, hãy dùng từ đồng nghĩa, từ khóa phụ.

            ---

            ## **Cấu trúc bài viết chuẩn SEO:**
            1. **Mở bài**: 
            - Thu hút người đọc, giới thiệu tổng quan sản phẩm.
            - Chứa **Primary Keyword** trong câu đầu tiên.

            2. **Thân bài**:
            - Mô tả **đặc điểm, công năng, lợi ích**.
            - **FAQ (Câu hỏi thường gặp)**: Sử dụng câu hỏi có từ khóa phụ.
            - **Hình ảnh** (sử dụng Markdown):
                - 🖼 **Sau mở bài**  
                - 🖼 **Giữa bài viết**  
                - 🖼 **Ở phần đặc điểm sản phẩm**  
                - 🖼 **Trước phần kết luận**  

            3. **Kết bài**:
            - Kêu gọi hành động (CTA) rõ ràng.

            ---

            ## **Chuẩn E-E-A-T trong SEO Content:**
            - **Trải nghiệm (Experience)**: Thêm ví dụ thực tế về sản phẩm.  
            - **Chuyên môn (Expertise)**: Nội dung chính xác, nhấn mạnh đặc điểm kỹ thuật.  
            - **Tính thẩm quyền (Authoritativeness)**: Dẫn chứng từ nguồn tin cậy.  
            - **Độ tin cậy (Trustworthiness)**: Trình bày rõ ràng, không gây hiểu lầm.

            ---

            ## **Định dạng & Kỹ thuật SEO:**
            - **Page Title**: Bắt buộc chứa **Primary Keyword**, không quá **60 ký tự**.
            - **Meta Description**: Tóm tắt lợi ích sản phẩm, chứa từ khóa, **tối đa 160 ký tự**.
            - **URL**: Ngắn gọn, không dài quá **75 ký tự**.
            - **Markdown**:
            - `#` cho **H1**, `##` cho **H2**, `###` cho **H3**.
            - **In đậm** (`**bold**`) cho từ khóa quan trọng.
            - Dùng `-` cho danh sách gạch đầu dòng, `1.` cho danh sách số thứ tự.
            - Dùng blockquote (`>`) cho các trích dẫn quan trọng.

            ---

            ## **Lưu ý quan trọng**:
            - **Xuất ra bài viết hoàn chỉnh dưới định dạng Markdown**.
            - Bài viết phải bao gồm thông tin Giá bán sản phẩm, Chính sách bảo hành đã được cung cấp, và Đảm bảo hàng chính hãng.
            - Chèn thêm mấy câu kiểu như "Hãy đến với [Chỗ này cho người dùng tự nhập tên cửa hàng] để mua sản phẩm" ở cuối bài (bạn có thể viết câu khác cùng ý nghĩa nhưng hay hơn).
            - **Không thêm phần giải thích, chỉ xuất nội dung bài viết**.
            - **Tuân thủ tiêu chuẩn SEO và E-E-A-T**, đặc biệt nếu sản phẩm thuộc lĩnh vực **YMYL (Your Money, Your Life)** như sức khỏe, tài chính.
            """

        
        # ✅ **Generate Facebook Ad Content**
        elif platform == "facebook":
            prompt = f"""
            Viết nội dung **quảng cáo Facebook** thu hút, tối ưu SEO cho sản phẩm **{product_name}**, dựa trên các thông tin sau:

            **Topic**: {topic_title}
            **Primary Keywords**: {', '.join(primary_keywords)}
            **Secondary Keywords**: {', '.join(secondary_keywords)}

            **Thông tin sản phẩm**: {combined_chunks}
            **Hình ảnh sản phẩm**: {image_section}

                    ### 🎯 **Yêu cầu quan trọng:**
            - **Giọng văn:**  
            - Ngắn gọn, hấp dẫn, thân thiện.  
            - Tập trung vào việc khơi gợi cảm xúc, kích thích hành động.  
            - **Từ khóa SEO:**  
            - Từ khóa chính: **{', '.join(primary_keywords)}**.
            - Xuất hiện ít nhất **3 lần** trong bài.  
            - Trình bày tự nhiên, tránh nhồi nhét từ khóa.  
            - **Độ dài:**  
            - Dưới **300 từ**, tránh diễn đạt lan man.  
            - Mỗi đoạn không quá **2-3 câu** để đảm bảo dễ đọc.  

            ---

            ### 🎯 **Cấu trúc nội dung:**
            1. **Tiêu đề (Headline) ngắn gọn & nổi bật:**  
            - Tối đa **60 ký tự**.  
            - Bắt đầu với **từ khóa chính** và tập trung vào **lợi ích chính** của sản phẩm.  
            - Ví dụ: `"iPhone 16 Pro – Đỉnh Cao Công Nghệ Hiện Đại!"`  

            2. **Mở bài hấp dẫn:**  
            - Gợi mở sự tò mò hoặc nêu vấn đề cần giải quyết.  
            - Chèn từ khóa chính **ngay đầu đoạn**.  
            - Ví dụ: `"Bạn đang tìm kiếm một chiếc smartphone cao cấp với camera đỉnh cao và hiệu suất mạnh mẽ? Hãy khám phá ngay iPhone 16 Pro!"`  

            3. **Thân bài – Nhấn mạnh lợi ích chính:**  
            - Nêu **3 lợi ích chính** nổi bật nhất.  
            - Dùng **dấu gạch đầu dòng** hoặc biểu tượng cảm xúc để tăng điểm nhấn.  
            - Ví dụ:  
                - ✅ **Camera 48MP sắc nét, chụp ảnh chuyên nghiệp.**  
                - ✅ **Chip A18 Pro mạnh mẽ, xử lý mượt mà mọi tác vụ.**  
                - ✅ **Màn hình Super Retina XDR sống động, hiển thị hoàn hảo.**  

            4. **Kêu gọi hành động mạnh mẽ (CTA):**  
            - Đưa ra **hành động rõ ràng** & **hấp dẫn**.  
            - Ví dụ:  
                - 👉 **Đặt hàng ngay hôm nay và nhận ưu đãi độc quyền!**  
                - 🎯 **Mua ngay tại [Link bán hàng]** để trải nghiệm công nghệ đỉnh cao.  

            ---

            ### 🎯 **Tối ưu E-E-A-T trên Facebook:**
            - **Experience:** Chèn ví dụ thực tế hoặc đánh giá từ khách hàng.  
            - Ví dụ: `"Hơn 1000 khách hàng đã trải nghiệm và hài lòng!"`  
            - **Expertise:** Đưa ra **thông số kỹ thuật ngắn gọn** giúp tăng độ tin cậy.  
            - **Authoritativeness:** Đảm bảo thông tin từ **nguồn chính hãng** (Apple hoặc cửa hàng uy tín).  
            - **Trustworthiness:** Đề cập **giá cả**, **chính sách bảo hành**, và **đảm bảo hàng chính hãng**.  

            ---

            ### 🎯 **Lưu ý kỹ thuật SEO:**
            - Từ khóa chính **phải xuất hiện trong**:  
            - **Tiêu đề, Mở bài, Thân bài và CTA.**  
            - **Tránh spam từ khóa** và đảm bảo giọng văn **tự nhiên**.  
            - Nội dung nên sử dụng **Markdown** hoặc biểu tượng cảm xúc để tăng điểm nhấn.  
            - Chèn hình ảnh sản phẩm để tăng sự hấp dẫn.
            ---

            ### **Kết quả yêu cầu:**
            - Xuất nội dung dưới **300 từ**.  
            - Đảm bảo cấu trúc tiêu đề, mở bài, thân bài, CTA rõ ràng. 
            - Thêm các hashtag hiệu quả 
            - **Không giải thích, chỉ xuất nội dung hoàn chỉnh.**  
            ### **Xuất nội dung hoàn chỉnh dưới 300 từ.**
            """

        else:
            raise ValueError("Invalid platform")

        # **🧠 Gọi LLM để sinh nội dung**
        human_message = HumanMessage(content=prompt)
        response = llm.invoke([human_message])
        return response.content.strip()

    except Exception as e:
        logging.error(f"Error in generate_seo_tool: {e}")
        return f"Error: {e}"

class WebsiteSEOWriterAgent:
    def write_seo(self, state):
        try:
            # 🔎 Gọi `vector_search()` để lấy dữ liệu
            context_data = vector_search(state.query)

            # ❌ Nếu không có kết quả từ vector_search, trả về lỗi
            if not context_data:
                logging.error("❌ Không tìm thấy kết quả nào từ vector_search.")
                state.result = "Không tìm thấy kết quả nào phù hợp với từ khóa."
                state.next_task = "end"
                return state.result

            # 🏷️ Lấy thông tin từ context_data
            product_name = context_data.get("product_name", state.query) or "Không xác định"
            topic_title = context_data.get("topic_title", f"Hướng dẫn về {product_name}")
            primary_keywords = context_data.get("primary_keywords", []) or [product_name]
            secondary_keywords = context_data.get("secondary_keywords", []) or []
            combined_chunks = context_data.get("chunk_texts", "Không có dữ liệu chi tiết.")

            # 🔎 Kiểm tra dữ liệu trước khi gọi API để tránh lỗi thiếu tham số
            if not topic_title:
                logging.error("❌ `topic_title` bị thiếu!")
                topic_title = f"Hướng dẫn về {product_name}"

            if not primary_keywords:
                logging.error("❌ `primary_keywords` bị thiếu!")
                primary_keywords = [product_name]

            if not secondary_keywords:
                logging.error("❌ `secondary_keywords` bị thiếu!")
                secondary_keywords = []

            if not combined_chunks:
                logging.warning("⚠️ `combined_chunks` trống, nội dung có thể bị thiếu.")
                combined_chunks = "Không có dữ liệu sản phẩm chi tiết."

            # 🛠 Debug log để kiểm tra dữ liệu truyền vào API
            logging.info(f"✅ Gọi generate_seo_tool với:")
            logging.info(f"🔹 Topic Title: {topic_title}")
            logging.info(f"🔹 Primary Keywords: {primary_keywords}")
            logging.info(f"🔹 Secondary Keywords: {secondary_keywords}")
            logging.info(f"🔹 Combined Chunks (Preview): {combined_chunks[:100]}...")  # Giới hạn log

            # 🔹 **Fix cách gọi `generate_seo_tool()`**
            response = generate_seo_tool.invoke({
                "product_name": product_name,
                "combined_chunks": combined_chunks,
                "topic_title": topic_title,
                "primary_keywords": primary_keywords,
                "secondary_keywords": secondary_keywords,
                "platform": "website"
            })

            return response

        except ValueError as e:
            logging.error(f"❌ Lỗi ValueError: {e}")
            state.result = str(e)
            state.next_task = "end"
            return state.result

        except Exception as e:
            logging.error(f"❌ Unexpected error in WebsiteSEOWriterAgent: {e}")
            state.result = "An error occurred during content generation."
            state.next_task = "end"
            return state.result

website_seo_writer_agent = WebsiteSEOWriterAgent()


class FacebookSEOWriterAgent:
    def write_seo(self, state):
        try:
            # 🔎 Gọi `vector_search()` để lấy dữ liệu
            context_data = vector_search(state.query)

            # ❌ Nếu không có kết quả, trả về lỗi
            if not context_data:
                logging.error("❌ Không tìm thấy kết quả nào từ vector_search.")
                state.result = "Không tìm thấy kết quả nào phù hợp với từ khóa."
                state.next_task = "end"
                return state.result

            # 🏷️ Lấy thông tin từ context_data
            product_name = context_data.get("product_name", state.query) or "Không xác định"
            topic_title = context_data.get("topic_title", f"Quảng cáo cho {product_name}")
            primary_keywords = context_data.get("primary_keywords", []) or [product_name]
            secondary_keywords = context_data.get("secondary_keywords", []) or []
            combined_chunks = context_data.get("chunk_texts", "Không có dữ liệu chi tiết.")

            # 🔎 Kiểm tra dữ liệu trước khi gọi API
            if not topic_title:
                logging.warning("⚠️ `topic_title` bị thiếu! Dùng mặc định.")
                topic_title = f"Quảng cáo cho {product_name}"

            if not primary_keywords:
                logging.warning("⚠️ `primary_keywords` bị thiếu! Dùng mặc định.")
                primary_keywords = [product_name]

            if not secondary_keywords:
                logging.warning("⚠️ `secondary_keywords` bị thiếu! Dùng mặc định.")
                secondary_keywords = []

            if not combined_chunks:
                logging.warning("⚠️ `combined_chunks` trống, nội dung có thể bị thiếu.")
                combined_chunks = "Không có dữ liệu sản phẩm chi tiết."

            # 🔹 Gọi `generate_seo_tool` để tạo nội dung bài viết Facebook
            response = generate_seo_tool.invoke({
                "product_name": product_name,
                "combined_chunks": combined_chunks,
                "topic_title": topic_title,
                "primary_keywords": primary_keywords,
                "secondary_keywords": secondary_keywords,
                "platform": "facebook"
            })

            # ✅ Đảm bảo `state.context` không phải `None`
            if state.context is None:
                state.context = {}

            # ✅ Lưu bài viết vào `state.context["last_facebook_post"]`
            state.context["last_facebook_post"] = {
                "product_name": product_name,
                "content": response
            }
            state.result = response
            state.next_task = "end"

            logging.info(f"✅ Bài viết Facebook đã được lưu vào state.context: {product_name}")

            return response

        except Exception as e:
            logging.error(f"❌ Unexpected error in FacebookSEOWriterAgent: {e}")
            state.result = "An error occurred during content generation."
            state.next_task = "end"
            return state.result

facebook_seo_writer_agent = FacebookSEOWriterAgent()

# Define Publisher Agent
import requests
import logging

class PublisherAgent:
    def publish_content(self, state):
        try:
            logging.info(f"📢 DEBUG - state.context trước khi đăng: {state.context}")

            # ✅ Kiểm tra nếu `state.context` không có dữ liệu
            if not state.context or "last_facebook_post" not in state.context:
                logging.error("❌ Không có nội dung nào trong state.context để đăng lên Facebook.")
                state.result = "Không có nội dung nào để đăng lên Facebook. Hãy yêu cầu tạo nội dung trước."
                state.next_task = "end"
                return state.result

            # ✅ Lấy bài viết cuối cùng từ `FacebookSEOWriterAgent`
            last_facebook_post = state.context["last_facebook_post"]

            if not isinstance(last_facebook_post, dict) or "content" not in last_facebook_post:
                logging.error("❌ Dữ liệu bài viết không hợp lệ!")
                state.result = "Không có nội dung nào để đăng lên Facebook. Hãy yêu cầu tạo nội dung trước."
                state.next_task = "end"
                return state.result

            # 🔎 Kiểm tra nội dung bài viết
            post_content = last_facebook_post.get("content", "").strip()
            product_name = last_facebook_post.get("product_name", "Không xác định")

            if not post_content:
                logging.error("❌ Nội dung bài viết rỗng, không thể đăng lên Facebook.")
                state.result = "Nội dung bài viết rỗng, hãy kiểm tra lại."
                state.next_task = "end"
                return state.result

            # ✅ Gửi bài đăng lên Facebook Page sử dụng Graph API
            url = f"https://graph.facebook.com/{FACEBOOK_PAGE_ID}/feed"
            payload = {
                "message": post_content,
                "access_token": FACEBOOK_ACCESS_TOKEN
            }
            response = requests.post(url, data=payload)
            response_data = response.json()

            # 🔎 Kiểm tra phản hồi từ Facebook API
            if "id" in response_data:
                post_id = response_data["id"]
                logging.info(f"✅ Bài viết '{product_name}' đã được đăng thành công! Post ID: {post_id}")
                state.result = f"✅ Bài viết '{product_name}' đã đăng thành công! Xem tại: https://www.facebook.com/{post_id}"
            else:
                error_message = response_data.get("error", {}).get("message", "Lỗi không xác định")
                logging.error(f"❌ Lỗi khi đăng bài: {error_message}")
                state.result = f"❌ Lỗi khi đăng bài: {error_message}"

            state.next_task = "end"
            return state.result

        except Exception as e:
            logging.error(f"❌ Unexpected error in PublisherAgent: {e}")
            state.result = "An error occurred during publishing."
            state.next_task = "end"
            return state.result
publisher_agent = PublisherAgent()

# Lưu trạng thái global
global_state = None

def generate_seo():
    global global_state  # Dùng biến toàn cục để giữ trạng thái Supervisor giữa các lần gọi

    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Nếu Supervisor đã tồn tại, giữ nguyên state và chỉ cập nhật query
        if global_state is None:
            global_state = Supervisor()
            global_state.state = AgentState(query=query, next_task="start")
        else:
            global_state.state.query = query
            global_state.state.next_task = "start"

        # Chạy Supervisor
        global_state.run(query=query)
        
        if not global_state.state.result:
            return jsonify({"error": "Không tìm thấy sản phẩm phù hợp."}), 404

        # Kiểm tra dữ liệu trước khi trả về
        supervisor_data = global_state.state.result
        print("Final Result Here", global_state.state.result)
        
        # Xóa chuỗi "```markdown" nếu có
        if supervisor_data and isinstance(supervisor_data, str):
            supervisor_data = supervisor_data.replace("```markdown", "").replace("```", "")

        return jsonify({"result": supervisor_data, "format": "markdown"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

