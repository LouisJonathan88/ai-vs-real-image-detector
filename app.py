import streamlit as st
import pickle
import cv2
import numpy as np
import numpy as np
from streamlit_option_menu import option_menu
from predict_utils import build_feature
import os
import base64         
import numpy as np

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Gambar AI vs Real",
    page_icon="🖼️",
    layout="wide"
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 300px !important;
    }
    .hero {
        padding: 35px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0ea5e9, #22c55e);
        color: white !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .hero h1, .hero p { color: white !important; }

    .card, .metric-card, .member-card {
        background-color: var(--background-color);
        color: var(--text-color);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128, 128, 128, 0.3);
        height: 100%;
        margin-bottom: 20px;
    }

    .metric-card {
        text-align: center;
        padding: 20px;
    }

    .member-card {
        text-align: center;
        transition: transform 0.2s ease;
    }
    .member-card:hover {
        transform: translateY(-5px);
        border-color: #0ea5e9;
    }
            
    h1, h2, h3, h4, p, li {
        color: var(--text-color);
    }

    .footer {
        text-align: center;
        color: gray;
        font-size: 14px;
        margin-top: 40px;
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Memuat model klasifikasi
@st.cache_resource(show_spinner=False)
def load_model():
    with open("svm_ai_vs_real_model.sav", "rb") as file:
        return pickle.load(file)

placeholder = st.empty()

with placeholder.container():
    st.markdown(
        """
        <div style="text-align:center; padding:40px;">
            <h4>🔄 Memuat Model Klasifikasi</h4>
            <p>Mohon tunggu, sistem sedang menyiapkan model!</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    model = load_model()

placeholder.empty()

# Sidebar navigasi menu
with st.sidebar:
    selected = option_menu(
        "Main Menu",
        ["Beranda", "Prediksi Gambar"],
        icons=["house", "image", "bar-chart"],
        menu_icon="cast",
        default_index=0
    )

# HALAMAN: BERANDA
if selected == "Beranda":
    import os
    import streamlit as st
    import base64

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🖼️ Deteksi Gambar AI vs Gambar Nyata</h1>
        <p>Sistem klasifikasi citra berbasis Machine Learning
        menggunakan algoritma <b>Support Vector Machine (SVM)</b></p>
    </div>
    """, unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        st.markdown("""
        <div class="card">
            <h3>📌 Deskripsi Aplikasi</h3>
            <p> Aplikasi ini dirancang untuk membedakan gambar hasil kecerdasan buatan (AI) dan gambar nyata secara otomatis melalui analisis karakteristik visual pada citra digital. </p>
            <p> Dengan memanfaatkan pendekatan Machine Learning, sistem mengekstraksi fitur-fitur penting dari setiap citra dan melakukan proses klasifikasi secara objektif dan konsisten. </p>
            <p> Aplikasi ini dapat digunakan sebagai alat bantu analisis citra, khususnya dalam menghadapi potensi penyalahgunaan teknologi AI di ranah visual. </p>
        </div>
        """, unsafe_allow_html=True)

    with colB:
        st.markdown("""
        <div class="card">
            <h3>🔄 Alur Machine Learning</h3>
            <ul>
                <li>📂 Dataset gambar AI dan gambar nyata</li>
                <li>📊 Pembagian data latih dan data uji</li>
                <li>🔁 Augmentasi (Flip Horizontal, Rotasi, Zoom)</li>
                <li>🔧 Pra-pemrosesan (resize dan normalisasi)</li>
                <li>🎨 Ekstraksi fitur (RGB Histogram, LBP, dan HOG)</li> 
                <li>🌳 Pelatihan dan pengujian model SVM</li>
                <li>📈 Evaluasi menggunakan Confusion Matrix</li>
                <li>🖼️ Sistem deteksi gambar AI dan gambar nyata</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    col1.markdown("""
    <div class="metric-card">
        <h2>📂</h2>
        <h3>Dataset</h3>
        <p>Gambar AI & Nyata</p>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown("""
    <div class="metric-card">
        <h2>🤖</h2>
        <h3>Algoritma</h3>
        <p>Support Vector Machine (SVM)</p>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown("""
    <div class="metric-card">
        <h2>🎯</h2>
        <h3>Ekstraksi Fitur</h3>
        <p>RGB Histogram, LBP, dan HOG</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("👥 Anggota Kelompok")

    def svg_avatar():
        svg = """
        <svg xmlns="http://www.w3.org/2000/svg" width="130" height="130" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="12" fill="#e5e7eb"/>
            <circle cx="12" cy="9" r="4" fill="#9ca3af"/>
            <path d="M4 20c1.5-4 14.5-4 16 0" fill="#9ca3af"/>
        </svg>
        """
        b64 = base64.b64encode(svg.encode()).decode()
        return f"<img src='data:image/svg+xml;base64,{b64}' width='130'/>"

    def show_member(photo, name, nim):
        if os.path.exists(photo):
            img = f"<img src='{photo}' width='130' style='border-radius:50%;'>"
        else:
            img = svg_avatar()

        st.markdown(f"""
        <div class="member-card">
            {img}
            <h4>{name}</h4>
            <p>{nim}</p>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        show_member("assets/anggota1.jpg", "M Insani I U", "NIM 10122352")
    with col2:
        show_member("assets/anggota2.jpg", "Louis Jonathan", "NIM 10122362")
    with col3:
        show_member("assets/anggota3.jpg", "Atam Kartam", "NIM 10122367")

    st.markdown("""
    <div class="footer">
        © 2026 | Pemrosesan Citra <br>
        Sistem Deteksi Gambar AI vs Gambar Nyata
    </div>
    """, unsafe_allow_html=True)

# HALAMAN : PREDIKSI GAMBAR
elif selected == "Prediksi Gambar":

    import streamlit as st
    import numpy as np
    import cv2

    # Bagian hero
    st.markdown("""
    <div class="hero">
        <h1>🔍 Prediksi Gambar AI dan Gambar Nyata</h1>
        <p>
            Unggah citra digital untuk mendeteksi apakah gambar merupakan hasil kecerdasan buatan (AI) atau gambar nyata
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Unggah gambar
    uploaded = st.file_uploader(
        "Format yang didukung: JPG, JPEG, dan PNG. "
        "Pastikan gambar memiliki resolusi yang memadai.",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded is None:
        st.info("Silakan unggah gambar terlebih dahulu untuk memulai proses klasifikasi.")

    # Proses jika gambar tersedia
    if uploaded is not None:
        try:
            # Membaca dan mengonversi gambar menjadi citra digital
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Pratinjau gambar
            col = st.columns([1, 2, 1])
            with col[1]:
                st.image(image, use_container_width=True)

            # Proses klasifikasi citra
            with st.spinner("🔎 Sistem sedang melakukan klasifikasi citra..."):
                X = build_feature(image)
                pred = model.predict(X)[0]
                proba = model.predict_proba(X)[0]
                confidence = np.max(proba) * 100

            label = "Gambar Hasil AI" if pred == 0 else "Gambar Nyata"

            # Hasil klasifikasi
            result_html = f"""
            <div class="card">
                <h3>📊 Hasil Klasifikasi</h3>

                <div style="
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 20px;
                ">
                    <div class="metric-card">
                        <h2>🏷️</h2>
                        <h4>Label Prediksi</h4>
                        <p><b>{label}</b></p>
                    </div>

                    <div class="metric-card">
                        <h2>🎯</h2>
                        <h4>Tingkat Akurasi</h4>
                        <p><b>{confidence:.2f}%</b></p>
                    </div>
                </div>

                <p style="margin-top:20px;">
                    Berdasarkan hasil analisis citra, sistem
                    mengklasifikasikan citra sebagai
                    <b>{label}</b> dengan tingkat akurasi
                    sebesar <b>{confidence:.2f}%</b>.
                </p>
            </div>
            """
            st.html(result_html)

            if confidence < 60:
                st.info(
                    "ℹ️ Tingkat kepercayaan model masih relatif rendah. "
                    "Disarankan menggunakan gambar dengan kualitas visual yang lebih baik."
                )

        except Exception:
            st.error("❌ Terjadi kesalahan saat memproses citra.")
            st.warning(
                "Pastikan file yang diunggah merupakan gambar valid "
                "dengan format JPG atau PNG."
            )

    # Footer
    st.markdown("""
    <div class="footer">
        © 2026 | Pemrosesan Citra <br>
        Sistem Deteksi Gambar AI vs Gambar Nyata
    </div>
    """, unsafe_allow_html=True)