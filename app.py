# -------------------------------------------------
# app.py – geotech.ai (GERÇEK ÜRÜN! CANLI!)
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
import hashlib
import uuid

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

# Kullanıcı Sistemi
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "reports" not in st.session_state:
    st.session_state.reports = []
if "is_pro" not in st.session_state:
    st.session_state.is_pro = False

# Streamlit
st.set_page_config(page_title="geotech.ai", page_icon="globe", layout="centered")

# Header
st.title("geotech.ai")
st.caption("Profesyonel Geoteknik AI – Ek-12 Rapor + Otomatik Analiz")

# Kullanıcı Girişi
with st.expander("Kullanıcı Girişi / Kayıt", expanded=not st.session_state.is_pro):
    email = st.text_input("Email")
    if st.button("Giriş / Kayıt"):
        st.session_state.email = email
        st.success("Hoş geldin! Ücretsiz 3 rapor hakkın var.")
        if "@" in email and "pro" in email:
            st.session_state.is_pro = True
            st.balloons()

# Rapor Limiti
report_count = len(st.session_state.reports)
if not st.session_state.is_pro and report_count >= 3:
    st.warning("Ücretsiz limit doldu. Pro için: pro@geotech.ai")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Veri Raporu Yükle")
    pdf_file = st.file_uploader("PDF Yükle", type="pdf")
    
    if pdf_file:
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
            
            # OTOMATİK RİSK
            messages = [{"role": "user", "content": f"Verilere göre risk analizi?\n{df.to_string()}"}]
            risk = llm.invoke(messages)
            st.markdown("**OTOMATİK RİSK:**")
            st.markdown(risk)
            
            # PDF Rapor
            def create_pdf():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = [Paragraph("EK-12 RAPOR", styles['Title'])]
                story.append(Table([['Derinlik', 'SPT', 'Zemin']] + df.values.tolist()))
                story.append(Paragraph(f"Risk: {risk}", styles['Normal']))
                doc.build(story)
                buffer.seek(0)
                return buffer.getvalue()
            
            pdf_bytes = create_pdf()
            st.download_button("PDF İndir", pdf_bytes, "rapor.pdf", "application/pdf")
            
            # Raporu kaydet
            report_id = hashlib.md5(pdf_file.read()).hexdigest()[:8]
            st.session_state.reports.append({"id": report_id, "risk": risk})
            share_link = f"https://app.geotech.ai/?report={report_id}"
            st.code(share_link, language=None)
            st.caption("Paylaşım linki (kopyala)")

# Ana Sohbet
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
                answer = "Selam! geotech.ai burada. PDF yükle, rapor al! 🚀"
            else:
                messages = [{"role": "user", "content": prompt}]
                answer = llm.invoke(messages)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.markdown("**geotech.ai** – Gerçek mühendisler için gerçek AI | [Pro Ol](mailto:pro@geotech.ai)")
