import streamlit as st
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
import pickle

from src.ml_utils import search_similar_chunks, build_prompt
from src import llm_model_path, embedding_model_path, vector_db

# === Load FAISS and Embeddings ===
with open(f"{vector_db}/chunk_texts.pkl", "rb") as f:
    chunk_texts = pickle.load(f)

index = faiss.read_index(f"{vector_db}/faiss.index")

# === Load Embedding Tokenizer & Model (Local) ===
embed_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path)
embed_model = AutoModel.from_pretrained(embedding_model_path)

# === Load LLM Tokenizer & Model (Local) ===
llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_path)

generator = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, device=-1)

# === Streamlit UI ===
st.title("💬 Local RAG Chatbot")

user_input = st.text_input("Ask me anything...", "")

if user_input:
    top_docs = search_similar_chunks(query=user_input, tokenizer=embed_tokenizer, model=embed_model, index=index, chunk_texts=chunk_texts, top_k=3)
    prompt = build_prompt(user_input, top_docs)

    with st.spinner("Thinking..."):
        response = generator(prompt, max_new_tokens=300)[0]["generated_text"]
        final_response = response.split("Answer:")[-1].strip()

    st.markdown("**📚 Retrieved Chunks:**")
    for doc in top_docs:
        st.info(doc)

    st.markdown("**🤖 Answer:**")
    st.success(final_response)
