"""Settings shared by the indexing and chat scripts."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
INDEX_PATH = BASE_DIR / "index.npz"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_TOKENS = 180
OVERLAP_TOKENS = 30
TOP_K = 3
MAX_NEW_TOKENS = 512
HISTORY_TURNS = 3

# Start with a 4,096-token context in LM Studio. This conservative byte
# budget limits request size without downloading another tokenizer.
# It is not an exact token count; LM Studio enforces its context limit.
MAX_PROMPT_BYTES = 6000

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
