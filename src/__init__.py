import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

embedding_model_path = "models/embedding/all-MiniLM-L6-v2"
llm_model_path = "models/tinyllama"
text_file = 'data/qa.txt'
vector_db = "data/faiss_index"