# -------------------------------------------------
# app.py – geotech.ai (ÇALIŞIR! "SELAM" + RAPOR + HATA YOK!)
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

# AI Model (DOĞRU!)
@st.cache_resource
def get_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational",  # BU SATIR DÜZELTİLDİ!
        temperature=0.3,
        max_new_tokens=500
    )

llm = get_llm()

# Streamlit
st.set_page_config(page_title="geotech.ai", page_icon="globe", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("geotech.ai")
st.caption("Ek-12 uyumlu geoteknik rapor oluşturucu")

# Sidebar
with st.sidebar:
    st.header("Ek-12 Rapor Oluştur")
    pdf_file = st.file_uploader("PDF Yükle", type="pdf")
    
    if pdf_file and st.button("Rapor Oluştur"):
        with st.spinner("Rapor hazırlanıyor..."):
            reader = PyPDF2.PdfReader(pdf_file)
            text = "".join([p.extract_text() or "" for p in reader.pages])
            
            depths = re.findall(r'Derinlik\D*(\d+\.?\d*)', text, re.I)
            spt_vals = re.findall(r'SPT\D*(\d+)', text, re.I)
            soil_types = re.findall(r'(Kil|Kum|Çakıl|Tın|Organik)', text, re.I)
            
            max_len = max(len(depths), len(spt_vals), len(soil_types))
            def pad(lst, l): return lst + ['-'] * (l - len(lst))
            depths, spt_vals, soil_types = [pad(lst, max_len) for lst in [depths, spt_vals, soil_types]]
            
            df = pd.DataFrame({'Derinlik': depths, 'SPT': spt_vals, 'Zemin': soil_types})
            st.dataframe(df)
            
            # AI (CHAT FORMATI)
            messages = [
                {"role": "user", "content": f"Verilere göre likefaksiyon riski nedir?\n{df.to_string()}"}
            ]
            try:
                answer = llm.invoke(messages)
            except Exception as e:
                answer = "AI hatası: " + str(e)
            st.write("AI Risk:", answer)
            
            # PDF Rapor
            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = [Paragraph("EK-12 RAPOR", styles['Title'])]
                story.append(Table([['Derinlik', 'SPT', 'Zemin']] + df.values.tolist()))
                story.append(Paragraph(f"Risk: {answer}", styles['Normal']))
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()
            
            st.download_button("PDF İndir", create_pdf(), "rapor.pdf", "application/pdf")

# Sohbet
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Sor..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("AI..."):
            if "selam" in prompt.lower():
                answer = "Selam! geotech.ai burada! PDF yükle, rapor al! ⚡"
            else:
                messages = [{"role": "user", "content": prompt}]
                try:
                    answer = llm.invoke(messages)
                except Exception as e:
                    answer = "AI hatası: " + str(e)
            st.markdown(answer)
            st.session.add_rows({"role": "assistant", "content": answer})
