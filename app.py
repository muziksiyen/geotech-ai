# -------------------------------------------------
# app.py – geotech.ai (ÇALIŞIR! EK-12 RAPOR + AI + HIZLI)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import PyPDF2
import re
import matplotlib.pyplot as plt
import os
import io
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# LangChain
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# AI Model
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
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Streamlit
st.set_page_config(page_title="geotech.ai", page_icon="globe", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("geotech.ai")
st.caption("Ek-12 uyumlu geoteknik rapor oluşturucu")

# Sidebar
with st.sidebar:
    st.header("Ek-12 Rapor Oluştur")
    pdf_file = st.file_uploader("Veri Raporu PDF Yükle", type="pdf", key="ek12")
    
    if pdf_file and st.button("Rapor Oluştur"):
        with st.spinner("Rapor hazırlanıyor..."):
            # PDF'den metin çıkar
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            # Veri çıkar
            depths = re.findall(r'Derinlik\D*(\d+\.?\d*)', text, re.I)
            spt_vals = re.findall(r'SPT\D*(\d+)', text, re.I)
            soil_types = re.findall(r'(Kil|Kum|Çakıl|Tın|Organik)', text, re.I)
            cohesion = re.findall(r'Kohezyon\D*(\d+\.?\d*)', text, re.I)
            friction = re.findall(r'Sürtünme\D*(\d+\.?\d*)', text, re.I)
            
            min_len = min(len(depths), len(spt_vals), len(soil_types))
            df = pd.DataFrame({
                'Derinlik (m)': depths[:min_len],
                'SPT': spt_vals[:min_len],
                'Zemin Tipi': soil_types[:min_len],
                'Kohezyon (kPa)': cohesion[:min_len] if cohesion else ['-'] * min_len,
                'Sürtünme Açısı (°)': friction[:min_len] if friction else ['-'] * min_len
            })
            
            st.subheader("Çıkarılan Veri")
            st.dataframe(df)
            
            # AI ile risk analizi
            context = df.to_string()
            rag_chain = (
                {"context": lambda x: context, "question": RunnablePassthrough()}
                | PromptTemplate.from_template("Verilere göre likefaksiyon riski nedir? {context}\nCevap:")
                | llm
                | StrOutputParser()
            )
            risk = rag_chain.invoke("Risk analizi")
            st.write("AI Risk Tahmini:", risk)
            
            # Ek-12 Rapor PDF
            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                story.append(Paragraph("ZEMİN VE TEMEL ETÜDÜ RAPORU (EK-12)", styles['Title']))
                story.append(Spacer(1, 12))
                story.append(Paragraph("1. GİRİŞ<br/>Proje: Örnek Proje<br/>Amaç: Temel tasarımı", styles['Normal']))
                story.append(Spacer(1, 12))
                
                data = [['Derinlik', 'SPT', 'Zemin']] + df[['Derinlik (m)', 'SPT', 'Zemin Tipi']].values.tolist()
                table = Table(data)
                story.append(table)
                
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"2. RİSK: {risk}", styles['Normal']))
                
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()
            
            pdf_bytes = create_pdf()
            st.download_button("Ek-12 Rapor PDF İndir", pdf_bytes, "ek12_rapor.pdf", "application/pdf")

# Ana Sohbet
with st.container():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Sorunu sor…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI düşünüyor..."):
                rag_chain = (
                    {"context": lambda x: "", "question": RunnablePassthrough()}
                    | PromptTemplate.from_template("Geoteknik sorusu: {question}\nCevap:")
                    | llm
                    | StrOutputParser()
                )
                answer = rag_chain.invoke(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
