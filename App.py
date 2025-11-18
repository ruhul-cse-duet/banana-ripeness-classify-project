import streamlit as st
import time
import logging
logging.basicConfig(level=logging.INFO)
import warnings
warnings.filterwarnings('ignore')
import os

import torch
import numpy as np
from PIL import Image

from src.custom_resnet import prediction_img

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_labels = ['overripe', 'ripe', 'rotten', 'unripe']

stage_descriptions = {
    'Unripe': (
        "🍃 Mostly green peel, high starch content.",
        "Best for storage or recipes needing firmer texture."
    ),
    'Overripe': (
        "🌑 Yellow peel covered with brown speckles.",
        "Great for smoothies, baking, and sweet dishes."
    ),
    'Rotten': (
        "⚠️ Black/brown peel with mushy or moldy spots.",
        "Discard immediately — unsafe for consumption."
    ),
    'Ripe': (
        "🍌 Bright yellow peel with a soft bite.",
        "Ready to enjoy fresh or slice into meals."
    )
}

st.set_page_config(page_title="Banana Ripeness Classification System", layout="wide")
with open("assets/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0
app_mode = st.session_state['page']
if 'pred_label' not in st.session_state:
    st.session_state['pred_label'] = None
if 'probabilities' not in st.session_state:
    st.session_state['probabilities'] = None



if(app_mode == "home"):
    st.markdown('<h1 class="title"; style="text-align:center; margin: 0.5rem 0;">Banana Ripeness Classification</h1>', unsafe_allow_html=True)
    st.markdown('<style>[data-testid="stSidebar"]{display:none;}</style>', unsafe_allow_html=True)
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.markdown(
            """
            <div class="hero">
              <div>
                <p class="hero-t">
                    Monitor banana supply chains with a lightweight CNN trained on the Kaggle Banana Ripeness dataset.
                    Upload any banana photo and receive a stage prediction (unripe, ripe, overripe, or rotten) with confidence.
                    The interface is optimized for agronomists, QA teams, and hobby growers who need fast visual inspection.
                    This demo is provided for educational purposes—always double-check critical decisions.
                </p>
                <div class="cta"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Ripeness stages we detect", expanded=True):
            st.markdown(
                "- **Unripe**: green peel, firm texture.\n"
                "- **Ripe**: yellow peel, ready to eat.\n"
                "- **Overripe**: yellow with several brown speckles.\n"
                "- **Rotten**: mostly brown/black, discard safely."
            )

        start = st.button("Analyze Ripeness", type="primary")
        if start:
            st.session_state['page'] = 'analysis'
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.session_state['probabilities'] = None
            st.rerun()
        st.markdown(
            """
            <div class="card-grid">
                <div class="card"><h3>Edge Ready</h3><p>
                    Custom CNN distilled from ResNet ideas; runs comfortably on CPU and accelerates with CUDA.</p>
                </div>
                <div class="card"><h3>Quality Insights</h3><p>
                    Confidence scores plus recommended handling tips per stage.</p>
                </div>
            </div>
            """,
             unsafe_allow_html=True,
        )
    with colB:
        sample_image_path = "test_img/ripe-1.jpg"
        if os.path.exists(sample_image_path):
            st.image(sample_image_path, caption="Sample Banana Image", width='stretch')
        else:
            st.info("Sample image not found. Please upload a banana photo to test the application.")

    st.markdown(
        """
             <div class="footer">For demo use only — verify before large-scale deployment.</div>
        """, unsafe_allow_html=True,
    )
elif(app_mode=="analysis"):
    nav_cols = st.columns([0.2, 0.6, 0.2])
    with nav_cols[0]:
        if st.button("Home"):
            st.session_state['page'] = 'home'
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.session_state['probabilities'] = None
            st.rerun()
    with nav_cols[1]:
        st.markdown('<div id="ripeness-analysis-anchor"></div>', unsafe_allow_html=True)
    with nav_cols[2]:
        pass

    st.markdown('<div id="ripeness-analysis"><p>Banana Ripeness Classification</p></div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload Banana Image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state['uploader_key']}"
    )

    if uploaded is None:
        st.info("Please upload a banana photo (JPEG/PNG) to begin the analysis.")
    else:
        image = Image.open(uploaded).convert('RGB')
        display_image = image.resize((400,400))
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(display_image, caption="Uploaded Image", width=400)
            c1, c2 = st.columns(2)
            with c1:
                predict_clicked = st.button("Predict", type="primary")
            with c2:
                clear_clicked = st.button("Clear")

        if clear_clicked:
            st.session_state['uploader_key'] += 1
            st.session_state['pred_label'] = None
            st.session_state['probabilities'] = None
            st.rerun()

        if predict_clicked:
            image_resized = image.resize((384,384))
            img_array = np.array(image_resized).astype(np.float32) / 255.0
            if img_array.ndim == 2:
                img_array = np.stack([img_array]*3, axis=-1)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std

            def to_device(data, device):
                if isinstance(data, (list,tuple)):
                    return [to_device(x, device) for x in data]
                return data.to(device, non_blocking=True)

            x = img_tensor.unsqueeze(0)
            x = to_device(x, device)

            with st.spinner("Running prediction..."):
                start = time.time()
                pred_idx_tensor, prob_tensor = prediction_img(x)

                if isinstance(pred_idx_tensor, torch.Tensor):
                    pred_idx_tensor = pred_idx_tensor.squeeze()
                if isinstance(pred_idx_tensor, (list, tuple)):
                    pred_idx = pred_idx_tensor[0]
                else:
                    pred_idx = pred_idx_tensor

                pred_idx = int(pred_idx)

                pred_label = class_labels[pred_idx]
                st.session_state['pred_label'] = pred_label

                if isinstance(prob_tensor, torch.Tensor):
                    prob_vector = prob_tensor.squeeze().tolist()
                else:
                    prob_vector = prob_tensor

                if isinstance(prob_vector, float):
                    prob_vector = [prob_vector]

                st.session_state['probabilities'] = {
                    label: float(prob_vector[i]) if i < len(prob_vector) else 0.0
                    for i, label in enumerate(class_labels)
                }

                end = time.time()
                logging.info(f"Prediction Response Time: {end - start:.4f} sec")

        with col2:
            if st.session_state['pred_label']:
                label = st.session_state['pred_label']
                st.image(display_image, caption=f"Prediction: {label}", width=400)
                summary, action = stage_descriptions.get(label, ("", ""))
                if summary:
                    st.success(summary)
                if action:
                    st.info(action)
                probs = st.session_state.get('probabilities')
                if probs:
                    confidence = probs.get(label, 0.0)
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                    st.write("Stage probabilities")
                    for lbl, prob in probs.items():
                        st.progress(int(min(max(prob, 0.0), 1.0) * 100), text=lbl)
                        st.caption(f"{lbl}: {prob*100:.1f}%")
            else:
                st.image(display_image, caption="Prediction pending", width=400)
                st.info("Click Predict to estimate ripeness.")
    st.markdown(
        """
             <div class="footer">For demo use only — verify before large-scale deployment.</div>
        """, unsafe_allow_html=True,
    )

# streamlit run App.py