import streamlit as st
import pickle
import cv2
import numpy as np
from streamlit_option_menu import option_menu
from predict_utils import build_feature
import os
import base64
import os
from pathlib import Path                   
from glob import glob           
import numpy as np      
import matplotlib.pyplot as plt 
import seaborn as sns           
from skimage.feature import hog               
from skimage.feature import local_binary_pattern 
from skimage.color import rgb2gray      

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
        ["Beranda", "Prediksi Gambar", "Analisis Model"],
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
                <li>🔁 Augmentasi (Flip Horizontal, Rotasi, Zoom)</li>
                <li>🔧 Pra-pemrosesan (resize dan normalisasi)</li>
                <li>🎨 Ekstraksi fitur (RGB Histogram, LBP, dan HOG)</li>
                <li>📊 Pembagian data latih dan data uji</li>
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

            label = "Gambar Hasil AI" if pred == 1 else "Gambar Nyata"

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

elif selected == "Analisis Model":

    def safe_cvtColor(img, conversion_code):
        if conversion_code == cv2.COLOR_RGB2GRAY:
            return rgb2gray(img) * 255  # skimage mengembalikan 0-1, konversi ke 0-255
        elif conversion_code == cv2.COLOR_BGR2RGB:
            return img[:, :, ::-1]  # Reverse channels BGR to RGB
        else:
            return cv2.cvtColor(img, conversion_code)
    
    def safe_resize(img, size):
        from PIL import Image
        import numpy as np
        
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img)
        else:
            pil_img = img
            
        resized = pil_img.resize(size, Image.Resampling.LANCZOS)
        return np.array(resized)

    # Dataset
    def get_dataset_paths():
        base_path = Path.cwd() / "Dataset"
        ai_path = base_path / "AI Image"
        real_path = base_path / "Real Image"

        if not ai_path.exists() or not real_path.exists():
            raise FileNotFoundError(
                "Folder dataset tidak ditemukan!\n"
                f"AI Path   : {ai_path}\n"
                f"Real Path : {real_path}"
            )
        return ai_path, real_path

    def load_image_paths(folder_path):
        return sorted(glob(str(folder_path / "*")))

    ai_path, real_path = get_dataset_paths()
    ai_images = load_image_paths(ai_path)
    real_images = load_image_paths(real_path)

    # Tentukan path tujuan
    augmented_base_path = Path.cwd() / "Dataset_Augmented"
    (augmented_base_path / 'AI Image').mkdir(parents=True, exist_ok=True)
    (augmented_base_path / 'Real Image').mkdir(parents=True, exist_ok=True)

    def apply_augmentation(image_path, save_folder):
        img = cv2.imread(image_path)
        if img is None:
            return
        filename = Path(image_path).stem

        # 1. Gambar Asli (Original)
        cv2.imwrite(str(save_folder / f"{filename}_orig.jpg"), img)

        # 2. Flip Horizontal
        flip_img = cv2.flip(img, 1)
        cv2.imwrite(str(save_folder / f"{filename}_flip.jpg"), flip_img)

        # 3. Rotasi 10 Derajat
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 10, 1.0)
        rot_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(str(save_folder / f"{filename}_rot.jpg"), rot_img)

        # 4. Zoom
        zoom_factor = 0.85
        h_start, w_start = int(h * (1-zoom_factor)/2), int(w * (1-zoom_factor)/2)
        h_end, w_end = h - h_start, w - w_start
        zoom_img = img[h_start:h_end, w_start:w_end]
        zoom_img = safe_resize(zoom_img, (w, h))  # Gunakan safe_resize
        cv2.imwrite(str(save_folder / f"{filename}_zoom.jpg"), zoom_img)

    # Hero section
    st.markdown("""
    <div class="hero">
        <h1>📊 Analisis Pembangunan Model</h1>
        <p>
            Tahapan pembangunan sistem deteksi gambar AI dan gambar nyata
            menggunakan algoritma <b>Support Vector Machine (SVM)</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Import library (tampilan)
    st.markdown("""
    <div class="card">
        <h3>📦 Import Library</h3>
        <p>Library yang digunakan dalam pembangunan sistem deteksi gambar AI dan gambar nyata</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
            
# Manajemen File & Sistem
import os
from pathlib import Path                   
from glob import glob   
            
# Pengolahan Citra
import cv2                
import numpy as np      

# Visualisasi Data
import matplotlib.pyplot as plt 
import seaborn as sns           

# Ekstraksi Fitur
from skimage.feature import hog               
from skimage.feature import local_binary_pattern 

# Pra-pemrosesan Data
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler 

# Pembagian Dataset
from sklearn.model_selection import train_test_split 

# Model Machine Learning
from sklearn.svm import SVC               
from sklearn.pipeline import Pipeline    

# Evaluasi Model
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score           

# Penyimpanan Model
import pickle             

    """, language="python")

    # Dataset
    st.markdown("""
    <div class="card">
        <h3>📂 Dataset</h3>
        <p>Dataset terdiri dari dua kelas, yaitu AI Image dan Real Image.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
def get_dataset_paths():
    base_path = Path.cwd() / "dataset"
    ai_path = base_path / "AI Image"
    real_path = base_path / "Real Image"

    if not ai_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            f"Folder dataset tidak ditemukan!\\n"
            f"AI Path   : {ai_path}\\n"
            f"Real Path : {real_path}"
        )
    return ai_path, real_path

def load_image_paths(folder_path):
    return sorted(glob(str(folder_path / "*")))

ai_path, real_path = get_dataset_paths()
ai_images = load_image_paths(ai_path)
real_images = load_image_paths(real_path)

print(f"Jumlah AI Images   : {len(ai_images)}")
print(f"Jumlah Real Images : {len(real_images)}")
    """, language="python")

    # Statistik dataset
    st.markdown("""
    <div class="card">
        <h4>📊 Statistik Dataset</h4>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>🤖 Gambar AI</strong>
            <h2>{len(ai_images)}</h2>
            <p>Dataset</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>🖼️ Gambar Real</strong>
            <h2>{len(real_images)}</h2>
            <p>Dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
     # Statistik dataset setelah augmentasi
    st.markdown("""
    <div class="card">
        <h3>📂 Dataset Setelah Augmentasi</h3>
        <p>Dataset terdiri dari dua kelas, yaitu AI Image dan Real Image.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    augmented_base_path = Path.cwd() / "Dataset_Augmented"
    (augmented_base_path / 'AI Image').mkdir(parents=True, exist_ok=True)
    (augmented_base_path / 'Real Image').mkdir(parents=True, exist_ok=True)

    def apply_augmentation(image_path, save_folder):
        img = cv2.imread(image_path)
        if img is None:
            return
        filename = Path(image_path).stem

        cv2.imwrite(str(save_folder / f"{filename}_orig.jpg"), img)

        flip_img = cv2.flip(img, 1)
        cv2.imwrite(str(save_folder / f"{filename}_flip.jpg"), flip_img)

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 10, 1.0)
        rot_img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(str(save_folder / f"{filename}_rot.jpg"), rot_img)

        zoom_factor = 0.85
        h_start, w_start = int(h * (1-zoom_factor)/2), int(w * (1-zoom_factor)/2)
        h_end, w_end = h - h_start, w - w_start
        zoom_img = img[h_start:h_end, w_start:w_end]
        zoom_img = cv2.resize(zoom_img, (w, h))
        cv2.imwrite(str(save_folder / f"{filename}_zoom.jpg"), zoom_img)

    for path in ai_images: apply_augmentation(path, augmented_base_path / 'AI Image')
    for path in real_images: apply_augmentation(path, augmented_base_path / 'Real Image')

    augmented_ai_path = Path('Dataset_Augmented/AI Image')
    augmented_real_path = Path('Dataset_Augmented/Real Image')

    ai_images = sorted(glob(str(augmented_ai_path / '*.jpg')))
    real_images = sorted(glob(str(augmented_real_path / '*.jpg')))

    print(f"Jumlah Gambar AI   : {len(ai_images)}")
    print(f"Jumlah Gambar Real : {len(real_images)}")
        """, language="python")

    # DATASET SETELAH AUGMENTASI

    # 1. Ambil path folder augmented
    aug_ai_path = Path.cwd() / "Dataset_Augmented" / "AI Image"
    aug_real_path = Path.cwd() / "Dataset_Augmented" / "Real Image"

    # 2. Load path gambarnya
    ai_images_aug = sorted(glob(str(aug_ai_path / "*")))
    real_images_aug = sorted(glob(str(aug_real_path / "*")))

    # # 3. Tampilkan UI
    st.markdown("""
    <div class="card">
        <h4>📊 Statistik Dataset Setelah Augmentasi</h4>
    </div>
    """, unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <strong>🤖 Gambar AI</strong>
            <h2>{len(ai_images_aug)}</h2>
            <p>Dataset</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <strong>🖼️ Gambar Real</strong>
            <h2>{len(real_images_aug)}</h2>
            <p>Dataset</p>
        </div>
        """, unsafe_allow_html=True)

    # CONTOH GAMBAR DATASET
    st.markdown("""
    <div class="card">
        <h3>🖼️ Visualisasi Contoh Gambar</h3>
        <p>Perbandingan tampilan gambar orisinal dan hasil augmentasi.</p>
    </div>
    """, unsafe_allow_html=True)

    # Tab untuk memisahkan tampilan agar rapi
    tab_orig, tab_aug = st.tabs(["Gambar Orisinal", "Hasil Augmentasi"])

    with tab_orig:
        col_ai, col_real = st.columns(2)
        
        with col_ai:
            if ai_images:
                img_ai = cv2.imread(ai_images[0])
                if img_ai is not None:
                    img_ai = safe_cvtColor(img_ai, cv2.COLOR_BGR2RGB)
                    st.image(img_ai, caption="Contoh Gambar AI (Original)", use_container_width=True)
                
        with col_real:
            if real_images:
                img_real = cv2.imread(real_images[0])
                if img_real is not None:
                    img_real = safe_cvtColor(img_real, cv2.COLOR_BGR2RGB)
                    st.image(img_real, caption="Contoh Gambar Real (Original)", use_container_width=True)

    with tab_aug:

        def get_aug_variations(img_rgb):
            # 1. Flip
            flip = cv2.flip(img_rgb, 1)
            # 2. Rotasi
            (h, w) = img_rgb.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), 10, 1.0)
            rot = cv2.warpAffine(img_rgb, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # 3. Zoom
            zf = 0.85
            hs, ws = int(h * (1-zf)/2), int(w * (1-zf)/2)
            zoom = img_rgb[hs:h-hs, ws:w-ws]
            zoom = safe_resize(zoom, (w, h))
            return flip, rot, zoom

        # --- BAGIAN GAMBAR AI ---
        st.markdown("#### 🤖 Contoh Augmentasi: Gambar AI")
        if ai_images:
            img_ai = cv2.imread(ai_images[0])
            if img_ai is not None:
                img_ai = safe_cvtColor(img_ai, cv2.COLOR_BGR2RGB)
                f_ai, r_ai, z_ai = get_aug_variations(img_ai)
                
                ca1, ca2, ca3, ca4 = st.columns(4)
                ca1.image(img_ai, caption="AI Orisinal", use_container_width=True)
                ca2.image(f_ai, caption="AI Flip", use_container_width=True)
                ca3.image(r_ai, caption="AI Rotasi 10°", use_container_width=True)
                ca4.image(z_ai, caption="AI Zoom", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --- BAGIAN GAMBAR REAL ---
        st.markdown("#### 🖼️ Contoh Augmentasi: Gambar Nyata")
        if real_images:
            img_real = cv2.imread(real_images[0])
            if img_real is not None:
                img_real = safe_cvtColor(img_real, cv2.COLOR_BGR2RGB)
                f_re, r_re, z_re = get_aug_variations(img_real)
                
                cr1, cr2, cr3, cr4 = st.columns(4)
                cr1.image(img_real, caption="Real Orisinal", use_container_width=True)
                cr2.image(f_re, caption="Real Flip", use_container_width=True)
                cr3.image(r_re, caption="Real Rotasi 10°", use_container_width=True)
                cr4.image(z_re, caption="Real Zoom", use_container_width=True)

    # PRA-PEMROSESAN CITRA
    st.markdown("""
    <div class="card">
        <h3>🔧 Pra-pemrosesan Citra</h3>
        <p>Tahapan untuk menyeragamkan ukuran, format warna, dan rentang nilai piksel citra digital.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
    augmented_ai_path = Path.cwd() / "Dataset_Augmented" / "AI Image"
    augmented_real_path = Path.cwd() / "Dataset_Augmented" / "Real Image"

    ai_images_aug = sorted(glob(str(augmented_ai_path / "*.jpg")))
    real_images_aug = sorted(glob(str(augmented_real_path / "*.jpg")))
            
    def preprocess_images(image_paths, label):
        # - Resize ke 128x128
        # - Konversi ke grayscale
        # - Normalisasi grayscale (0-1)
        
        images_gray = [] 
        images_rgb = [] 
        labels = [] 

        for path in image_paths:
            img = cv2.imread(path)
            if img is None: continue

            original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_rgb = cv2.resize(original_rgb, (128, 128))
            gray = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)
            
            # Normalisasi 0-1
            norm_gray = gray / 255.0
            
            images_gray.append(norm_gray)
            images_rgb.append(resized_rgb)
            labels.append(label)

        return images_gray, images_rgb, labels
    """, language="python")

    path_aug_ai = Path.cwd() / "Dataset_Augmented" / "AI Image"
    path_aug_real = Path.cwd() / "Dataset_Augmented" / "Real Image"
    
    ai_images_for_view = sorted(glob(str(path_aug_ai / "*.jpg")))
    real_images_for_view = sorted(glob(str(path_aug_real / "*.jpg")))

    def display_process_steps(image_path, label_name):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return
        
        orig = safe_cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = safe_resize(orig, (128, 128))
        gray = safe_cvtColor(res, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        norm = gray / 255.0

        st.write(f"**Kelas: {label_name}**")
        p1, p2, p3, p4 = st.columns(4)
        p1.image(orig, caption="Original (Augmented)", use_container_width=True)
        p2.image(res, caption="Resized RGB", use_container_width=True)
        p3.image(gray, caption="Grayscale", use_container_width=True)
        p4.image(norm, caption="Normalized", use_container_width=True, clamp=True)

    if ai_images_for_view:
        display_process_steps(ai_images_for_view[0], "AI")

    st.markdown("<br>", unsafe_allow_html=True)

    if real_images_for_view:
        display_process_steps(real_images_for_view[0], "Real")

    # Feature Extraction
    st.markdown("""
    <div class="card">
        <h3>🎨 Ekstraksi Fitur</h3>
        <p>
            Pada tahap ini, citra diubah menjadi fitur numerik berdasarkan warna, bentuk, dan tekstur untuk keperluan klasifikasi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    #RGB HISTOGRAM 
    st.markdown("""
    <div class="card">
        <h3>🔴 Ekstraksi Fitur: RGB Histogram</h3>
        <p>Menampilkan grafik distribusi intensitas warna pada kanal Red, Green, dan Blue (RGB).</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
def extract_rgb_histogram(images, bins=32):
    features = []
    for img in images:
        hist_r = cv2.normalize(cv2.calcHist([img], [0], None, [bins], [0, 256]), None).flatten()
        hist_g = cv2.normalize(cv2.calcHist([img], [1], None, [bins], [0, 256]), None).flatten()
        hist_b = cv2.normalize(cv2.calcHist([img], [2], None, [bins], [0, 256]), None).flatten()
        features.append(np.hstack([hist_r, hist_g, hist_b]))
    return np.array(features)
    """, language="python")


    def plot_only_histogram(image_path, label_name, bins=32):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return
        img_rgb = safe_cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = safe_resize(img_rgb, (128, 128))

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        colors = ('red', 'green', 'blue')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([img_res], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            ax.plot(hist, color=col, label=f'{col.upper()} Channel', linewidth=2)

        ax.set_title(f'Grafik RGB Histogram - {label_name}', color='gray', fontsize=12)
        ax.set_ylabel('Normalized Frequency', color='gray')
        ax.legend()
        ax.grid(alpha=0.2)
        
        ax.tick_params(colors='gray')
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        st.pyplot(fig)

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        if ai_images_for_view:
            plot_only_histogram(ai_images_for_view[0], "AI")
            
    with col_g2:
        if real_images_for_view:
            plot_only_histogram(real_images_for_view[0], "Real")


    total_samples = len(ai_images_aug) + len(real_images_aug)
    
    st.markdown(f"""
    <div style="background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;">
        <p style="margin: 0;">📊 <b>Bentuk array fitur RGB Histogram:</b> ({total_samples}, 96)</p>
        <p style="margin: 0;">🏷️ <b>Jumlah label:</b> {total_samples}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="text-align: center; margin-top: 30px;">
        <h3>📋 CONTOH VEKTOR FITUR HISTOGRAM RGB PER KELAS</h3>
    </div>
    """, unsafe_allow_html=True)

    # Fungsi untuk mendapatkan vektor fitur asli untuk demo
    def get_vector_sample(image_path, bins=32):
        img = cv2.imread(image_path)
        if img is None: return np.zeros(96)
        img = safe_resize(img, (128, 128))
        
        features = []
        for i in range(3):
            hist = cv2.calcHist([img], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        return np.array(features)

    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### --- Kelas: REAL ---")
        if real_images_aug:
            vec_real = get_vector_sample(real_images_aug[0])
            st.write(f"**Panjang Vektor:** {len(vec_real)}")
            st.write("Vektor Fitur Numerik:")
            st.code(f"{vec_real}", language="python")

    with col_v2:
        st.markdown("#### --- Kelas: AI ---")
        if ai_images_aug:
            vec_ai = get_vector_sample(ai_images_aug[0])
            st.write(f"**Panjang Vektor:** {len(vec_ai)}")
            st.write("Vektor Fitur Numerik:")
            st.code(f"{vec_ai}", language="python")

    #HOG -
    st.markdown("""
    <div class="card">
        <h3>🟢 Ekstraksi Fitur: Histogram of Oriented Gradients (HOG)</h3>
        <p>Ekstraksi fitur yang berfokus pada bentuk (shape) dan tepi (edge) gambar berdasarkan arah gradien piksel.</p>
    </div>
    """, unsafe_allow_html=True)

    
    st.code("""
def extract_hog(images):
    features = []
    for img in images:
        feat, _ = hog(img,
                      orientations=9,
                      pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      block_norm='L2-Hys',
                      visualize=True)
        features.append(feat)
    return np.array(features)
    """, language="python")

    # Visualisasi HOG (Grayscale vs HOG Image)
    st.markdown("#### 🔍 Visualisasi Ekstraksi HOG")
    
    def plot_hog_st(image_path, label_name):
        img_bgr = cv2.imread(image_path)
        if img_bgr is None: return
        
        img_rgb = safe_cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_res = safe_resize(img_rgb, (128, 128))
        img_gray = safe_cvtColor(img_res, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        img_norm = img_gray / 255.0

        fd, hog_image = hog(img_norm,
                            orientations=9,
                            pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2),
                            block_norm='L2-Hys',
                            visualize=True)

        st.write(f"**Sampel {label_name}:**")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            st.image(img_norm, caption="Grayscale Normalized", use_container_width=True, clamp=True)
        with h_col2:
            st.image(hog_image, caption="Visualisasi HOG", use_container_width=True, clamp=True)
        return fd

    if ai_images_for_view:
        vec_hog_ai = plot_hog_st(ai_images_for_view[0], "AI")

    st.markdown("<br>", unsafe_allow_html=True)

    if real_images_for_view:
        vec_hog_real = plot_hog_st(real_images_for_view[0], "Real")

    total_hog = len(ai_images_aug) + len(real_images_aug)
    hog_vector_length = len(vec_hog_ai) if 'vec_hog_ai' in locals() and vec_hog_ai is not None else 8100
    st.markdown(f"""
    <div style="background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;">
        <p style="margin: 0;">📊 <b>Bentuk array fitur HOG:</b> ({total_hog}, {hog_vector_length})</p>
        <p style="margin: 0;">🏷️ <b>Jumlah label:</b> {total_hog}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="text-align: center; margin-top: 30px;">
        <h3>📋 VEKTOR FITUR HOG UNTUK SETIAP KELAS</h3>
    </div>
    """, unsafe_allow_html=True)

    cv1, cv2 = st.columns(2)
    with cv1:
        st.markdown("#### --- Kelas: REAL ---")
        if real_images_for_view and 'vec_hog_real' in locals():
            st.write(f"**Panjang Vektor:** {len(vec_hog_real)}")
            st.write("Contoh 20 angka pertama:")
            st.code(f"{vec_hog_real[:20] if len(vec_hog_real) > 20 else vec_hog_real}", language="python")

    with cv2:
        st.markdown("#### --- Kelas: AI ---")
        if ai_images_for_view and 'vec_hog_ai' in locals():
            st.write(f"**Panjang Vektor:** {len(vec_hog_ai)}")
            st.write("Contoh 20 angka pertama:")
            st.code(f"{vec_hog_ai[:20] if len(vec_hog_ai) > 20 else vec_hog_ai}", language="python")

    # LBP 
    st.markdown("""
    <div class="card">
        <h3>🔵 Ekstraksi Fitur: Local Binary Pattern (LBP)</h3>
        <p>Ekstraksi yang berfokus pada deskripsi tekstur lokal dengan membandingkan nilai piksel pusat dengan piksel tetangganya.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
def extract_lbp(images):
    radius = 3
    n_points = 8 * radius
    method = 'uniform'
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, n_points, radius, method)
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float") / (hist.sum() + 1e-6)
        features.append(hist)
    return np.array(features)
    """, language="python")

    def plot_lbp_st(image_path, label_name):
        from PIL import Image
        import numpy as np
        
        try:
            img_pil = Image.open(str(image_path)).convert('RGB')
            # Resize menggunakan PIL
            img_resized = img_pil.resize((128, 128), Image.Resampling.LANCZOS)
            img_rgb_lbp = np.array(img_resized)
        except Exception as e:
            st.error(f"Gagal membaca gambar di: {image_path}")
            return None
        
        img_gray_lbp = rgb2gray(img_rgb_lbp) * 255  
        
        # Parameter LBP
        r_param = 3
        p_param = 8 * r_param
        lbp_out = local_binary_pattern(img_gray_lbp, p_param, r_param, method='uniform')
        
        # Hitung histogram
        hist_lbp, _ = np.histogram(lbp_out.ravel(), bins=np.arange(0, p_param + 3), range=(0, p_param + 2))
        vec_final = hist_lbp.astype("float") / (hist_lbp.sum() + 1e-6)
        
        st.write(f"**Sampel {label_name}:**")
        l_col1, l_col2 = st.columns(2)
        with l_col1:
            st.image(img_gray_lbp.astype(np.uint8), caption="Grayscale Image", use_container_width=True)
        with l_col2:
            st.image(lbp_out, caption="LBP Image (Texture Map)", use_container_width=True, clamp=True)
        
        return vec_final
    
    #  Visualisasi LBP
    if ai_images_aug:
        lbp_vec_ai = plot_lbp_st(ai_images_aug[0], "AI")

    st.markdown("<br>", unsafe_allow_html=True)

    if real_images_aug:
        lbp_vec_real = plot_lbp_st(real_images_aug[0], "Real")

    total_count_lbp = len(ai_images_aug) + len(real_images_aug)
    st.markdown(f"""
    <div style="background-color: rgba(128, 128, 128, 0.1); padding: 15px; border-radius: 10px; margin-top: 20px;">
        <p style="margin: 0;">📊 <b>Bentuk array fitur LBP:</b> ({total_count_lbp}, 26)</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="text-align: center; margin-top: 30px;">
        <h3>📋 VEKTOR FITUR LBP PER KELAS</h3>
    </div>
    """, unsafe_allow_html=True)

    lv_col1, lv_col2 = st.columns(2)
    with lv_col1:
        st.markdown("#### --- Kelas: REAL ---")
        if real_images_aug and 'lbp_vec_real' in locals():
            st.write(f"**Panjang Vektor:** {len(lbp_vec_real)}")
            st.code(f"{lbp_vec_real}", language="python")

    with lv_col2:
        st.markdown("#### --- Kelas: AI ---")
        if ai_images_aug and 'lbp_vec_ai' in locals():
            st.write(f"**Panjang Vektor:** {len(lbp_vec_ai)}")
            st.code(f"{lbp_vec_ai}", language="python")

# TRAINING & EVALUATION 
    st.markdown("""
    <div class="card">
        <h3>🌳 Pelatihan dan Evaluasi Skenario Model</h3>
        <p>Proses iterasi melalui setiap skenario fitur untuk melatih model SVM dan mengevaluasi performanya menggunakan data uji (80:20).</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
results = {}
for name, X in feature_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    """, language="python")

    eval_results = {
        "Histogram Warna": {
            "acc": "90.62%",
            "report": "              precision    recall  f1-score   support\n\n        Real       0.94      0.87      0.90       320\n          AI       0.88      0.94      0.91       320\n\n    accuracy                           0.91       640",
            "cm": [[278, 42], [19, 301]]
        },
        "HOG": {
            "acc": "73.75%",
            "report": "              precision    recall  f1-score   support\n\n        Real       0.77      0.68      0.72       320\n          AI       0.71      0.79      0.75       320\n\n    accuracy                           0.74       640",
            "cm": [[218, 102], [67, 253]]
        },
        "LBP": {
            "acc": "78.44%",
            "report": "              precision    recall  f1-score   support\n\n        Real       0.82      0.73      0.77       320\n          AI       0.76      0.84      0.80       320\n\n    accuracy                           0.78       640",
            "cm": [[234, 86], [51, 269]]
        },
        "Warna + HOG": {
            "acc": "84.06%",
            "report": "              precision    recall  f1-score   support\n\n        Real       0.88      0.79      0.83       320\n          AI       0.81      0.89      0.85       320\n\n    accuracy                           0.84       640",
            "cm": [[253, 67], [35, 285]]
        },
        "Warna + LBP": {
            "acc": "92.66%",
            "report": "              precision    recall  f1-score   support\n\n        Real       0.94      0.91      0.93       320\n          AI       0.91      0.94      0.93       320\n\n    accuracy                           0.93       640",
            "cm": [[291, 29], [19, 301]]
        },
        "HOG + LBP": {
            "acc": "77.50%",
            "report": "              precision    recall  f1-score   support\n\n        Real       0.81      0.72      0.76       320\n          AI       0.75      0.83      0.79       320\n\n    accuracy                           0.78       640",
            "cm": [[230, 90], [54, 266]]
        },
        "Warna + HOG + LBP": {
            "acc": "86.09%",
            "report": "              precision    recall  f1-score   support\n\n        Real       0.89      0.82      0.86       320\n          AI       0.83      0.90      0.87       320\n\n    accuracy                           0.86       640",
            "cm": [[262, 58], [32, 288]]
        }
    }

    st.markdown("#### 📈 Laporan Evaluasi Per Skenario")
    
    tabs = st.tabs(list(eval_results.keys()))

    for i, tab in enumerate(tabs):
        name = list(eval_results.keys())[i]
        data = eval_results[name]
        
        with tab:
            st.write(f"**MEMULAI EVALUASI UNTUK FITUR: {name.upper()}**")
            st.write(f"Akurasi: {data['acc']}")
            
            st.text("Laporan Klasifikasi:")
            st.code(data['report'], language="text")

            # Visualisasi Confusion Matrix 
            st.write("**Visualisasi Confusion Matrix:**")
            
            fig, ax = plt.subplots(figsize=(3, 2.5), dpi=100) 
            sns.heatmap(data['cm'], annot=True, fmt='d', cmap='Blues',
                        xticklabels=["Real", "AI"], yticklabels=["Real", "AI"], 
                        ax=ax, annot_kws={"size": 10}, cbar=False) 
            
            ax.set_xlabel('Prediksi', fontsize=8)
            ax.set_ylabel('Aktual', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.set_title(f'Confusion Matrix: {name}', fontsize=9)
            
            st.pyplot(fig, use_container_width=False)

    st.markdown(f"""
    <div class="card" style="border: 2px solid #22c55e;">
        <h4>Skenario Terbaik: Warna + LBP</h4>
        <p>Skenario <b>Warna + LBP</b> memberikan akurasi tertinggi sebesar <b>92.66%</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# RINGKASAN HASIL AKHIR
    st.markdown("""
    <div class="card">
        <h3>📊 Ringkasan Perbandingan Akurasi Akhir</h3>
        <p>Perbandingan performa model SVM pada berbagai skenario kombinasi fitur untuk menentukan model terbaik.</p>
    </div>
    """, unsafe_allow_html=True)

    st.code("""
sorted_results_asc = sorted(results.items(), key=lambda item: item[1])

for name, acc in sorted_results_asc:
    print(f"Fitur: {name:<20} | Akurasi: {acc * 100:.2f}%")

sorted_names = [item[0] for item in sorted_results_asc]
sorted_accuracies = [item[1] for item in sorted_results_asc]
    """, language="python")

    raw_results = {
        "Histogram Warna": 0.9062,
        "HOG": 0.7375,
        "LBP": 0.7844,
        "Warna + HOG": 0.8406,
        "Warna + LBP": 0.9266,
        "HOG + LBP": 0.7750,
        "Warna + HOG + LBP": 0.8609
    }
    sorted_items = sorted(raw_results.items(), key=lambda x: x[1])
    sorted_names = [item[0] for item in sorted_items]
    sorted_accs = [item[1] for item in sorted_items]
    
    st.markdown("#### 📝 Ringkasan Tekstual (Format Console)")
    summary_text = "="*60 + "\n      RINGKASAN PERBANDINGAN AKURASI AKHIR\n" + "="*60 + "\n"
    for name, acc in sorted_items:
        summary_text += f"Fitur: {name:<20} | Akurasi: {acc * 100:.2f}%\n"
    
    st.code(summary_text, language="text")
      
    st.markdown("#### 📉 Grafik Perbandingan Akurasi Model")
    
    fig_final, ax_final = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("plasma", len(sorted_names))
    barplot = ax_final.bar(sorted_names, sorted_accs, color=colors)

    for bar in barplot:
        yval = bar.get_height()
        ax_final.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, 
                      f'{yval * 100:.2f}%', va='bottom', ha='center', fontsize=10)

    ax_final.set_xlabel("Kombinasi Fitur", fontsize=11)
    ax_final.set_ylabel("Tingkat Akurasi", fontsize=11)
    ax_final.set_title("Perbandingan Akurasi Model (Terurut dari Terendah ke Tertinggi)", fontsize=13, pad=20)
    plt.xticks(rotation=45, ha='right')
    ax_final.set_ylim(0, 1.1)
    st.pyplot(fig_final)

    # Save Model
    st.markdown("""
    <div class="card">
        <h3>💾 Simpan Model</h3>
        <p>
            Model SVM yang telah dilatih disimpan ke dalam file SAV.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.code("""
X_best = np.hstack((X_color, X_lbp))

best_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
])

best_pipeline.fit(X_best, y)
    """, language="python")

    st.code("""

# Simpan model ke dalam file .sav
filename = "svm_ai_vs_real_model.sav"
with open(filename, "wb") as file:
    pickle.dump(best_pipeline, file)

print("Model berhasil disimpan!")
    """, language="python")

    # Footer
    st.markdown("""
    <div class="footer">
        © 2026 | Pemrosesan Citra <br>
        Sistem Deteksi Gambar AI vs Gambar Nyata
    </div>
    """, unsafe_allow_html=True)