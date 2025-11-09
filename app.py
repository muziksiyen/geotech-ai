# -------------------------------------------------
# app.py – geotech.ai (Grok-style sohbet + PDF/CSV)
# -------------------------------------------------
import streamlit as st
import pandas as pd
import PyPDF2
import re
import matplotlib.pyplot as plt
from io import BytesIO

# ---------- Streamlit config ----------
st.set_page_config(page_title="geotech.ai", page_icon="globe", layout="wide")

# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Header ----------
st.title("geotech.ai")
st.caption("Zemini anlayan yapay zeka – PDF/CSV yükle, soru sor, kaynak gör.")

# ---------- Sidebar: Dosya yükleme ----------
with st.sidebar:
    st.header("Veri Yükle")
    uploaded_files = st.file_uploader(
        "PDF, CSV, Excel", type=["pdf", "csv", "xlsx"], accept_multiple_files=True
    )

# ---------- Sohbet geçmişi ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("Kaynaklar"):
                for s in msg["sources"]:
                    st.caption(s)

# ---------- Kullanıcı girişi ----------
if prompt := st.chat_input("Sorunu sor… (örnek: SPT ortalaması?)"):
    # Kullanıcı mesajı
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI cevabı
    with st.chat_message("assistant"):
        with st.spinner("Analiz ediliyor…"):
            answer = ""
            sources = []

            # ---------- 1) Yüklenen dosyalarda arama ----------
            if uploaded_files:
                for f in uploaded_files:
                    name = f.name
                    # ---- PDF ----
                    if f.type == "application/pdf":
                        reader = PyPDF2.PdfReader(f)
                        txt = ""
                        for page in reader.pages:
                            txt += page.extract_text() or ""
                        # SPT bul
                        spt = re.findall(r"SPT\D*(\d+)", txt, re.I)
                        if spt:
                            avg = sum(map(int, spt)) / len(spt)
                            answer += f"**{name}** → SPT ortalaması: **{avg:.1f}**\n"
                            sources.append(f"{name} (PDF)")
                    # ---- CSV / Excel ----
                    else:
                        df = pd.read_csv(f) if f.type == "text/csv" else pd.read_excel(f)
                        if "SPT" in df.columns:
                            fig, ax = plt.subplots()
                            df["SPT"].hist(ax=ax, bins=15, edgecolor="black")
                            ax.set_title(f"SPT Dağılımı – {name}")
                            st.pyplot(fig)
                            answer += f"**{name}** → SPT ortalaması: **{df['SPT'].mean():.1f}**\n"
                            sources.append(f"{name} (Tablo)")

            # ---------- 2) Genel formül örnekleri ----------
            if any(w in prompt.lower() for w in ["oturma", "settlement"]):
                answer += "\n**Oturma formülü**\n"
                st.latex(r"s = \frac{q B (1-\nu^2)}{E_s} \cdot I")
                answer += "\n> *Eₛ ≈ 8 × SPT (korelasyon)*\n"
                sources.append("Terzaghi & Peck (1967)")

            # ---------- 3) Sonuç ----------
            st.markdown(answer or "Veri bulunamadı.")
            if sources:
                with st.expander("Kaynaklar"):
                    for s in sources:
                        st.caption(s)

        # Kaydet
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
