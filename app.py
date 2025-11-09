# -------------------------------------------------
# app.py – geotech.ai (EK-12 RAPOR + CIVILS.AI ÖZELLİKLERİ)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import PyPDF2
import re
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
import easyocr
import cv2
import numpy as np
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

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
ocr_reader = easyocr.Reader(['en', 'tr'], gpu=False)

# Streamlit
st.set_page_config(page_title="geotech.ai", page_icon="globe", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

st.title("geotech.ai")
st.caption("Civils.ai gibi geoteknik AI – Ek-12 rapor oluştur, borehole çıkar!")

# Sidebar: Geoteknik Araçlar
with st.sidebar:
    st.header("Geoteknik Araçlar")
    
    # 1. Borehole Digitizer + Ek-12 Rapor
    if st.button("Ek-12 Geoteknik Rapor Oluştur"):
        pdf_file = st.file_uploader("Veri Raporu PDF Yükle", type="pdf", key="ek12")
        if pdf_file:
            with st.spinner("Veri parse ediliyor ve Ek-12 raporu oluşturuluyor..."):
                # Veri Parse (Civils.ai gibi)
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                # OCR
                for page in reader.pages:
                    if page.images:
                        img_data = page.images[0].data
                        img = Image.open(io.BytesIO(img_data))
                        ocr_result = ocr_reader.readtext(np.array(img))
                        ocr_text = " ".join([res[1] for res in ocr_result])
                        text += " " + ocr_text
                
                # Veri Çıkar
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
                    'Kohezyon (kPa)': cohesion[:min_len],
                    'Sürtünme Açısı (°)': friction[:min_len]
                })
                
                st.subheader("Parse Edilen Veri")
                st.dataframe(df)
                
                # AI ile Eksik Tamamla (Ek-12 için)
                context = df.to_string()
                rag_chain = (
                    {"context": lambda x: context, "question": RunnablePassthrough()}
                    | PromptTemplate.from_template("Ek-12'ye göre eksik zemin parametrelerini tamamla: {context}\nSoru: {question}\nCevap:")
                    | llm
                    | StrOutputParser()
                )
                completed_data = rag_chain.invoke("Eksik parametreleri tamamla (örneğin, E_s, φ).")
                st.write("AI Tamamlaması:", completed_data)
                
                # Ek-12 Rapor Oluştur (Detaylı Şablon)
                st.subheader("Ek-12 Geoteknik Rapor Önizlemesi")
                
                # Rapor Bölümleri (Ek-12 Formatı)
                rapor = """
                <h2>ZEMİN VE TEMEL ETÜDÜ RAPORU (EK-12 UYUMLU)</h2>
                <p><strong>1. GİRİŞ</strong></p>
                <p>Proje: [Proje Adı]<br>
                Lokasyon: [Koordinatlar]<br>
                Amaç: Temel tasarımı için zemin etüdü.</p>
                
                <p><strong>2. SAHA ÇALIŞMALARI</strong></p>
                <p>Sondaj Sayısı: {sondaj_sayisi}<br>
                Testler: SPT, CPT.</p>
                """.format(sondaj_sayisi=len(df))
                
                st.markdown(rapor, unsafe_allow_html=True)
                
                st.dataframe(df)  # Saha verisi tablosu
                
                # 3. LABORATUVAR TESTLERİ
                st.subheader("3. LABORATUVAR TESTLERİ")
                st.write("Granülometri, Atterberg Sınırları, Kesme Dayanımı.")
                
                # 4. ZEMİN PARAMETRELERİ
                st.subheader("4. ZEMİN VE KAYA PARAMETRELERİ")
                if spt_vals:
                    avg_spt = sum(map(int, spt_vals)) / len(spt_vals)
                    e_s = avg_spt * 8  # Korelasyon
                    st.write(f"Ortalama SPT: {avg_spt:.1f}, E_s ≈ {e_s:.1f} MPa")
                
                # 5. TEMEL TASARIMI
                st.subheader("5. TEMEL TASARIMI")
                st.write("Taşıma Kapasitesi: q_ult = c N_c + γ D N_q + 0.5 γ B N_γ")
                st.latex(r"q_{ult} = c N_c + \gamma D N_q + 0.5 \gamma B N_\gamma")
                
                # 6. RİSK DEĞERLENDİRMESİ
                st.subheader("6. RİSK DEĞERLENDİRMESİ")
                if avg_spt < 10:
                    st.error("Likefaksiyon Riski: YÜKSEK")
                else:
                    st.success("Likefaksiyon Riski: DÜŞÜK")
                
                # 7. SONUÇ VE ÖNERİLER
                st.subheader("7. SONUÇ VE ÖNERİLER")
                st.write("Temel Tipi: Yüzeysel / Kazık. Ek test önerisi: CPT.")
                
                # PDF Export
                def create_pdf_report(df):
                    buffer = io.BytesIO()
                    doc = SimpleDocTemplate(buffer, pagesize=letter)
                    styles = getSampleStyleSheet()
                    story = []
                    
                    # Başlık
                    title = Paragraph("ZEMİN VE TEMEL ETÜDÜ RAPORU (EK-12)", styles['Title'])
                    story.append(title)
                    story.append(Spacer(1, 12))
                    
                    # Veri Tablosu
                    data = [['Derinlik', 'SPT', 'Zemin Tipi']] + df.values.tolist()
                    table = Table(data)
                    story.append(table)
                    
                    doc.build(story)
                    buffer.seek(0)
                    return buffer.getvalue()
                
                pdf_bytes = create_pdf_report(df)
                st.download_button("Ek-12 Rapor PDF İndir", pdf_bytes, "ek12_geoteknik_rapor.pdf", "application/pdf")
    
    # Diğer Araçlar (Civils.ai İlhamı)
    if st.button("Takeoff Ölçümü"):
        st.write("CAD PDF yükle, alan hesapla (OpenCV ile).")
    
    if st.button("Sözleşme Arama"):
        st.write("Sözleşme PDF yükle, madde ara.")

# Ana Sohbet (AI)
# (Önceki kodun sohbet kısmı – kısaltılmış)
with st.container():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Sorunu sor…"):
        # RAG zinciri (önceki gibi)
        rag_chain = (
            {"context": lambda x: "", "question": RunnablePassthrough()}  # Context boş, genişlet
            | PromptTemplate.from_template("Geoteknik sorusu: {question}\nCevap:")
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke(prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
