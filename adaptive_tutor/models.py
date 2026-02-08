from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# LLM1 = ChatGoogleGenerativeAI(
#         model="gemini-3-flash-preview",
#         temperature=0.2
#     )

LLM = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    max_retries=6,        # 6 baar try karega
    timeout=None,
    # retry_on_rate_limit=True  # Ye automatically backoff karega
)