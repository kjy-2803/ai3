# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1uj2lD8goJDLo9uSg_8HcT4bxnl2trPc8")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {

     labels[0]: {
    "texts": ["치킨은 나날이 비싸지는 메뉴입니다"],
       "images": ["data:image/jpeg;base64,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"],
       "videos": ["https://www.youtube.com/watch?v=rqvDqRKO4dE"]
     },
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")
