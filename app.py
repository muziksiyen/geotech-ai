# -------------------------------------------------
# app.py – geotech.ai (GROK GİBİ YAPAY ZEKA!)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import PyPDF2
import re
import matplotlib.pyplot as plt
from io import BytesIO
import os

# ---------- AI: LangChain + Mistral-7B ----------
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Hugging Face Token (Streamlit Secrets'ten)
if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

@st.cache_resource
def get_llm():
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 500}
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm = get_llm()
embeddings = get_embeddings()

# ---------- Streamlit config ----------
st.set_page_config(page_title="geotech.ai", page_icon="globe", layout="wide")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ---------- Header ----------
st.title("geotech.ai")
st.caption("Zemini anlayan YAPAY ZEKA – PDF/CSV yükle, soru sor, akıllı cevap al.")

# ---------- Sidebar: Dosya yükleme ----------
with st.sidebar:
    st.header("Veri Yükle")
    uploaded_files = st.file_uploader(
        "PDF, CSV, Excel", type=["pdf", "csv", "xlsx"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Verileri İşle"):
        docs = []
        for f in uploaded_files:
            if f.type == "application/pdf":
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                docs.append(text)
            else:
                df = pd.read_csv(f) if f.type == "text/csv" else pd.read_excel(f)
                docs.append(df.to_string())
        
        if docs:
            text_chunks = []
            for doc in docs:
                chunks = [doc[i:i+1000] for i in range(0, len(doc), 1000)]
                text_chunks.extend(chunks)
            
            st.session_state.vectorstore = FAISS.from_texts(text_chunks, embeddings)
            st.success("Veriler AI belleğine eklendi!")

# ---------- Sohbet geçmişi ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Kaynaklar"):
                for s in msg["sources"]:
                    st.caption(s)

# ---------- Kullanıcı girişi ----------
if prompt := st.chat_input("Sorunu sor… (örnek: Likefaksiyon riski var mı?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Yapay zeka düşünüyor…"):
            context = ""
            sources = []

            # 1. Yüklenen verilerden bilgi çek
            if st.session_state.vectorstore and prompt:
                docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
                context = "\n\n".join([d.page_content for d in docs])
                sources = ["Yüklenen veri"] * len(docs)

            # 2. AI ile cevap üret
            prompt_template = """
            Sen geoteknik mühendisisin. Verilere ve bilginle soruya kısa, doğru, formüllü ve kaynaklı cevap ver.

            Veri: {context}
            Soru: {question}

            Cevap:
            """
            chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))
            answer = chain.run({"context": context, "question": prompt})

            # 3. Ekstra: Oturma formülü
            if any(w in prompt.lower() for w in ["oturma", "settlement"]):
                answer += "\n\n**Oturma formülü:**\n"
                st.latex(r"s = \frac{q B (1-\nu^2)}{E_s} \cdot I")
                answer += "\n> *Eₛ ≈ 8 × SPT (Terzaghi & Peck, 1967)*"

            st.markdown(answer)
            if sources:
                with st.expander("Kaynaklar"):
                    for s in sources:
                        st.caption(s)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
