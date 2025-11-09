# -------------------------------------------------
# app.py – geotech.ai (ÇALIŞIR! TÜM SORULARA CEVAP + RİSK + RAPOR)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import PyPDF2
import re
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# LangChain
from langchain_huggingface import HuggingFaceEndpoint

# Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# AI Model
@st.cache_resource
def get_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational",
        temperature=0.3,
        max_new_tokens=500
    )

llm = get_llm()

# Streamlit
st.set_page_config(page_title="geotech.ai", page_icon="globe", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None

st.title("geotech.ai")
st.caption("Veri raporu eklendiğinde OTOMATİK risk analizi + Ek-12 rapor! (Tüm sorulara cevap)")

# Sidebar
with st.sidebar:
    st.header("Veri Raporu Yükle")
    pdf_file = st.file_uploader("PDF Yükle", type="pdf")
    
    if pdf_file:
        with st.spinner("Rapor işleniyor..."):
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
            
            # EŞİT UZUNLUK YAP
            max_len = max(len(depths), len(spt_vals), len(soil_types), len(cohesion), len(friction))
            def pad_list(lst, length):
                return lst + ['-'] * (length - len(lst))
            
            depths = pad_list(depths, max_len)
            spt_vals = pad_list(spt_vals, max_len)
            soil_types = pad_list(soil_types, max_len)
            cohesion = pad_list(cohesion, max_len)
            friction = pad_list(friction, max_len)
            
            df = pd.DataFrame({
                'Derinlik (m)': depths,
                'SPT': spt_vals,
                'Zemin Tipi': soil_types,
                'Kohezyon (kPa)': cohesion,
                'Sürtünme Açısı (°)': friction
            })
            
            st.subheader("Çıkarılan Veri")
            st.dataframe(df)
            
            # OTOMATİK RİSK ANALİZİ (CHAT FORMATI!)
            messages = [
                {"role": "user", "content": f"Bu geoteknik veriler için likefaksiyon riski, oturma tahmini, taşıma kapasitesi ve temel önerisi nedir?\n{df.to_string()}"}
            ]
            try:
                risk_answer = llm.invoke(messages)
            except Exception as e:
                risk_answer = "AI hatası: " + str(e)
            
            st.subheader("OTOMATİK RİSK ANALİZİ")
            st.markdown(risk_answer)
            
            # Ek-12 Rapor PDF
            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                story.append(Paragraph("ZEMİN VE TEMEL ETÜDÜ RAPORU (EK-12)", styles['Title']))
                story.append(Spacer(1, 12))
                story.append(Paragraph("1. GİRİŞ\nProje: Örnek Proje\nAmaç: Temel tasarımı", styles['Normal']))
                story.append(Spacer(1, 12))
                
                data = [['Derinlik', 'SPT', 'Zemin', 'Kohezyon', 'Sürtünme']] + df.values.tolist()
                table = Table(data)
                story.append(table)
                
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"2. RİSK ANALİZİ:\n{risk_answer}", styles['Normal']))
                
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()
            
            pdf_bytes = create_pdf()
            st.download_button("Ek-12 Rapor PDF İndir", pdf_bytes, "ek12_rapor.pdf", "application/pdf")
            
            # Son raporu sakla
            st.session_state.last_report = {
                'df': df,
                'risk': risk_answer
            }

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
                # SELAM
                if "selam" in prompt.lower():
                    answer = "Selam! geotech.ai burada. PDF yükle, otomatik rapor al! 🚀"
                else:
                    # Son rapor varsa bağlam ekle
                    context = ""
                    if st.session_state.last_report:
                        context = f"Son rapor verileri:\n{st.session_state.last_report['df'].to_string()}\nRisk: {st.session_state.last_report['risk']}\n"
                    
                    messages = [
                        {"role": "user", "content": f"{context}Geoteknik sorusu: {prompt}"}
                    ]
                    try:
                        answer = llm.invoke(messages)
                    except Exception as e:
                        answer = "AI hatası: " + str(e)
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
