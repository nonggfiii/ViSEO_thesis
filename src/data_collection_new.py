import logging
import os
from dotenv import load_dotenv
import json
from SeoKeywordResearch import SeoKeywordResearch
from pymongo import MongoClient
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from tavily import AsyncTavilyClient
from datetime import datetime
import asyncio
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import START
from langchain_core.tools import tool

load_dotenv()

# Lấy các biến môi trường
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MONGO_URI = os.getenv("MONGODB_URI")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
app_id = os.getenv("WORDTRACKER_APP_ID")
app_key = os.getenv("WORDTRACKER_APP_KEY")

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize MongoDB
client = MongoClient(MONGO_URI)
db = client['seo_database']
collection = db['seo_results']

# Initialize LLM
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    max_output_tokens=8192,
    api_key=GOOGLE_API_KEY
)

# Define Input Models
class TavilyQuery(BaseModel):
    query: str = Field(description="Search query for Tavily API")
    topic: str = Field(default="general", description="Search type (general or news)")
    days: int = Field(default=7, description="Number of days back to search")
    domains: Optional[List[str]] = Field(default=None, description="Filter domains")

class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="List of sub-queries for Tavily search")

async def tavily_search(input_data: TavilySearchInput) -> List[Dict[str, Any]]:
    """Search product-related information using Tavily."""
    async def perform_search(query_item: TavilyQuery):
        try:
            response = await tavily_client.search(
                query=f"{query_item.query} {datetime.now().strftime('%m-%Y')}",
                topic=query_item.topic,
                days=query_item.days,
                max_results=10,
            )
            return response.get("results", [])
        except Exception as e:
            logging.error(f"Error during Tavily search: {e}")
            return []

    tasks = [perform_search(query) for query in input_data.sub_queries]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]

def fetch_keywords_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch and filter SEO keywords using SerpApi and AI filtering."""
    if "product_name" not in state:
        logging.error("Invalid state: 'product_name' is missing in state")
        state["keywords"] = []
        return state

    product_name = state["product_name"]
    try:
        # Fetch keywords from SerpApi
        keyword_research = SeoKeywordResearch(
            query=product_name, api_key=SERPAPI_KEY, lang="vi", country="vn", domain="google.com"
        )
        auto_complete = keyword_research.get_auto_complete()
        related_searches = keyword_research.get_related_searches()
        related_questions = keyword_research.get_related_questions()

        # Combine and deduplicate keywords, removing None values
        raw_keywords = list(set(filter(None, auto_complete + related_searches + related_questions)))
        logging.info(f"Fetched raw keywords: {raw_keywords}")

        if not raw_keywords:
            logging.error("No valid keywords retrieved.")
            state["keywords"] = []
            return state

        # Filter keywords using AI
        prompt = f"""
        Filter the following Vietnamese keywords to keep only those directly relevant to the product '{product_name}'.
        Keywords: {', '.join(raw_keywords)}
        Return the filtered list as plain text, one keyword per line.
        """
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            filtered_keywords = [kw.strip() for kw in response.content.split("\n") if kw.strip()]
            logging.info(f"Filtered keywords: {filtered_keywords}")
        except Exception as ai_error:
            logging.error(f"AI filtering failed: {ai_error}")
            filtered_keywords = raw_keywords  # Fallback: Use raw keywords if AI fails

        # Update state
        state["keywords"] = filtered_keywords
    except Exception as e:
        logging.error(f"Error fetching keywords: {e}")
        state["keywords"] = []
    return state


def generate_keywords_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate additional SEO keywords using AI from existing keywords, starting with a new seed keyword."""
    if "keywords" not in state or not state["keywords"]:
        logging.error("Invalid state: 'keywords' is missing or empty in state")
        state["generated_keywords"] = []
        state["seed_keyword"] = ""
        return state

    keywords = state["keywords"]

    # Step 1: Generate a new seed keyword
    try:
        seed_prompt = f"""
        From the following Vietnamese keywords, identify one primary seed keyword that is central to the product context:
        Keywords: {', '.join(keywords)}
        Return only the seed keyword as plain text.
        """
        seed_response = llm.invoke([HumanMessage(content=seed_prompt)])
        seed_keyword = seed_response.content.strip()
        state["seed_keyword"] = seed_keyword
        logging.info(f"Generated seed keyword: {seed_keyword}")
    except Exception as e:
        logging.error(f"Error generating seed keyword: {e}")
        state["seed_keyword"] = keywords[0]  # Fallback: Use the first keyword in the list
        seed_keyword = state["seed_keyword"]

    # Step 2: Generate additional keywords based on the new seed keyword
    try:
        generate_prompt = f"""
        Generate a list of additional SEO keywords in Vietnamese related to this seed keyword: {seed_keyword}.
        Ensure the keywords are relevant, diverse, and include long-tail keywords where possible.
        """
        response = llm.invoke([HumanMessage(content=generate_prompt)])
        state["generated_keywords"] = [kw.strip() for kw in response.content.split("\n") if kw.strip()]
        logging.info(f"Generated keywords: {state['generated_keywords']}")
    except Exception as e:
        logging.error(f"Error generating additional keywords: {e}")
        state["generated_keywords"] = []

    return state

import urllib.parse
import requests

app_id = os.getenv("WORDTRACKER_APP_ID")
app_key = os.getenv("WORDTRACKER_APP_KEY")

def analyze_keywords_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze keywords using Wordtracker API and pick top 10 keywords by search volume."""
    if "seed_keyword" not in state or not state["seed_keyword"]:
        logging.error("Invalid state: 'seed_keyword' is missing or empty")
        state["analyzed_keywords"] = []
        return state

    seed_keyword = state["seed_keyword"]

    # Fetch app_id and app_key from environment variables
    app_id = os.getenv("WORDTRACKER_APP_ID")
    app_key = os.getenv("WORDTRACKER_APP_KEY")

    if not app_id or not app_key:
        logging.error("Wordtracker app_id or app_key is not set in the environment variables.")
        state["analyzed_keywords"] = []
        return state

    try:
        # Define Wordtracker API endpoint and parameters
        url = "https://api.lc.wordtracker.com/v3/keywords/search"
        params = {
            "seeds": seed_keyword,  # Từ khóa chính
            "country_code": "VN",  # Mã quốc gia
            "language_code": "vi",  # Ngôn ngữ tiếng Việt
            "sort_by": "search_volume",  # Sắp xếp theo lượng tìm kiếm
            "sort_order": "descending",  # Thứ tự giảm dần
            "limit": 10,  # Lấy tối đa 10 kết quả
            "app_id": app_id,
            "app_key": app_key,
        }

        # Gửi yêu cầu đến API
        logging.info(f"Requesting Wordtracker API: {url} with params: {params}")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Xử lý kết quả trả về
        data = response.json()
        logging.info(f"Wordtracker API response: {data}")

        # Lấy danh sách từ khóa
        results = data.get("results", [])
        if not results:
            logging.warning("No results found in Wordtracker API response.")
            state["analyzed_keywords"] = []
            return state

        # Chọn top 10 từ khóa dựa trên lượng tìm kiếm
        top_keywords = [
            {
                "keyword": keyword.get("keyword", ""),
                "total_volume": keyword.get("search_volume", 0),
            }
            for keyword in results
        ]

        state["analyzed_keywords"] = top_keywords
        logging.info(f"Top keywords: {top_keywords}")

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP error while calling Wordtracker API: {e}")
        state["analyzed_keywords"] = []
    except Exception as e:
        logging.error(f"Unexpected error analyzing keywords: {e}")
        state["analyzed_keywords"] = []

    return state


def categorize_keywords_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Categorize keywords into SEO categories and group them into topics with primary and secondary keywords."""
    if "analyzed_keywords" not in state or not state["analyzed_keywords"]:
        logging.error("Invalid state: 'analyzed_keywords' is missing or empty in state")
        state["categorized_keywords"] = {"Uncategorized": []}
        state["content_topics"] = []
        return state

    analyzed_keywords = [kw["keyword"] for kw in state["analyzed_keywords"]]
    prompt = f"""
    Categorize the following Vietnamese keywords into these SEO categories:
    - Primary Keywords
    - Secondary Keywords
    - Long-tail Keywords
    - Transactional Keywords
    - Question-based Keywords
    - Branded Keywords
    - USP Keywords
    - Related Keywords
    - Semantic Keywords

    Additionally, group these keywords into content topics for SEO articles in Vietnamese. For each topic:
    - Provide a clear topic title.
    - Divide the keywords into:
        - Primary Keywords: Keywords most important and central to the topic.
        - Secondary Keywords: Supporting keywords that enhance the topic.
    Keywords: {', '.join(analyzed_keywords)}

    Return the result as a JSON object with the structure:
    {{
        "categories": {{
            "Primary Keywords": [...],
            "Secondary Keywords": [...],
            ...
        }},
        "topics": [
            {{
                "title": "Topic title 1",
                "primary_keywords": ["keyword1", "keyword2"],
                "secondary_keywords": ["keyword3", "keyword4"]
            }},
            {{
                "title": "Topic title 2",
                "primary_keywords": ["keyword5"],
                "secondary_keywords": ["keyword6", "keyword7"]
            }}
            {{
                "title": "Topic title 3",
                "primary_keywords": ["keyword1", "keyword2"],
                "secondary_keywords": ["keyword3", "keyword4"]
            }},
            {{
                "title": "Topic title 4",
                "primary_keywords": ["keyword5"],
                "secondary_keywords": ["keyword6", "keyword7"]
            }}
        ]
    }}
    """
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        raw_response = response.content.strip()

        # Clean and parse JSON response
        clean_response = raw_response.strip("```json").strip("```").strip()
        categorized_data = json.loads(clean_response)

        # Validate structure
        if not isinstance(categorized_data, dict) or "categories" not in categorized_data or "topics" not in categorized_data:
            raise ValueError("Invalid response structure.")

        state["categorized_keywords"] = categorized_data["categories"]
        state["content_topics"] = categorized_data["topics"]
        logging.info(f"Categorized keywords: {state['categorized_keywords']}")
        logging.info(f"Content topics: {state['content_topics']}")
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Error parsing categorized keywords: {e}")
        logging.error(f"Raw AI response: {raw_response}")
        # Fallback: Put all keywords in 'Uncategorized'
        state["categorized_keywords"] = {"Uncategorized": analyzed_keywords}
        state["content_topics"] = [
            {
                "title": "General Topic",
                "primary_keywords": [],
                "secondary_keywords": analyzed_keywords
            }
        ]
    except Exception as e:
        logging.error(f"Unexpected error during categorization: {e}")
        state["categorized_keywords"] = {"Uncategorized": analyzed_keywords}
        state["content_topics"] = [
            {
                "title": "General Topic",
                "primary_keywords": [],
                "secondary_keywords": analyzed_keywords
            }
        ]

    return state

def product_research_tool(state: Dict[str, Any]) -> Dict[str, Any]:
    """Perform detailed product research using AI and Tavily API."""
    if "product_name" not in state:
        logging.error("Invalid state: 'product_name' is missing in state")
        state["product_report"] = ""
        return state

    product_name = state["product_name"]
    search_query = TavilySearchInput(sub_queries=[TavilyQuery(query=f"Research about {product_name}")])
    try:
        search_results = asyncio.run(tavily_search(search_query))
        prompt = f"""
        Write a detailed and structured product report in Vietnamese for '{product_name}'.
        Focus on:
        - Features
        - Functionalities
        - Unique selling points
        - Technical details
        Search Results:
        {search_results[:5]}
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        state["product_report"] = response.content.strip()
    except Exception as e:
        logging.error(f"Error in product research: {e}")
        state["product_report"] = ""
    return state



# Define System Message
system_message = SystemMessage(content="""
You are an SEO research agent that uses tools to fetch, generate, analyze, and categorize SEO data, as well as perform detailed product research.
Your main tasks are:
- Fetching keywords for a product
- Generating additional keywords using AI
- Analyze the best keywords for SEO purposes
- Categorizing keywords into specific SEO categories
- Performing detailed product research and reporting on features, functionality, and technical details.
Respond with accurate and concise information.
""")

def create_seo_graph():
    """Create a graph for the SEO keyword research and product research pipeline."""
    builder = StateGraph(dict)

    # Define nodes to run tools
    builder.add_node("fetch_keywords_tool", fetch_keywords_tool)
    builder.add_node("generate_keywords_tool", generate_keywords_tool)
    builder.add_node("analyze_keywords_tool", analyze_keywords_tool)
    builder.add_node("categorize_keywords_tool", categorize_keywords_tool)
    builder.add_node("product_research_tool", product_research_tool)

    # Define edges for sequential execution
    builder.add_edge(START, "fetch_keywords_tool")
    builder.add_edge("fetch_keywords_tool", "generate_keywords_tool")
    builder.add_edge("generate_keywords_tool", "analyze_keywords_tool")
    builder.add_edge("analyze_keywords_tool", "categorize_keywords_tool")
    builder.add_edge("categorize_keywords_tool", "product_research_tool")

    return builder.compile()


def log_and_run(state, tool_func):
    logging.info(f"Running {tool_func.__name__} with state: {state}")
    return tool_func(state)


async def run_pipeline(product_name: str) -> Dict[str, Any]:
    """Run the SEO keyword research and product research pipeline for a product name."""
    graph = create_seo_graph()

    # Initial state
    state = {"product_name": product_name}

    # Execute the graph
    result = await graph.ainvoke(state)

    # Extract relevant results
    # return {
    #     "product_name": product_name,
    #     "keywords": result.get("filtered_keywords", "N/A"),
    #     "product_report": result.get("product_report", "N/A"),
    # }
    return {
    "product_name": product_name,
    # "keywords": result.get("filtered_keywords", "N/A"),
    # "generated_keywords": result.get("generated_keywords", "N/A"),
    # "seed_keyword": result.get("seed_keyword", "N/A"),
    # "analyzed_keywords": result.get("analyzed_keywords", "N/A"),
    "categorized_keywords": result.get("categorized_keywords", "N/A"),
    "content_topics": result.get("content_topics", "N/A"),
    "product_report": result.get("product_report", "N/A"),
}


