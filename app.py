import streamlit as st
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="AI-Powered Phishing Email Detection", page_icon="🛡️", layout="wide")

# ============= SIDEBAR =============
with st.sidebar:
    st.title("Project Information")
    st.markdown("---")
    st.markdown("**Project:**")
    st.write("AI-Powered Phishing Email Detection")

    st.markdown("**Domain:**")
    st.write("Artificial Intelligence & Machine Learning")

    st.markdown("**Best Model:**")
    st.write("LinearSVC")

    st.markdown("**Feature Extraction:**")
    st.write("TF-IDF + LSA + Metadata")

    st.markdown("**Final Dataset:**")
    st.write("82,486 unique emails")

    st.markdown("**Test Accuracy:**")
    st.write("98.52%")

    st.markdown("**Phishing Recall:**")
    st.write("99.00%")

    st.markdown("**Models Evaluated:**")
    st.write("KNN, Logistic Regression, LinearSVC, RandomForest")

    st.markdown("---")
    st.caption("Developed as part of an AI & ML Academic Project")
# ===================================

# CSS - same as before
st.markdown("""
<style>
.metric-card { background-color: #262730; padding: 20px; border-radius: 10px; text-align: center; }
.safe-box { background-color: #0d4d1f; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745; }
.phishing-box { background-color: #5a1a1a; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545; }
.stButton>button { background-color: #1f77b4; color: white; font-weight: bold; height: 3em; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = joblib.load('best_phishing_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer_final.pkl')
    svd = joblib.load('svd_embedder.pkl')
    return model, tfidf, svd

model, tfidf, svd = load_models()

# HEADER - main page same
st.title("🛡️ AI-Powered Phishing Email Detection")
st.markdown("### Classify emails as Phishing or Safe using Machine Learning and NLP")

# METRICS ROW - same as before
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card"><h3>82,486</h3><p>Total Emails</p></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>LinearSVC</h3><p>Best Model</p></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>98.52%</h3><p>Test Accuracy</p></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-card"><h3>TF-IDF+LSA</h3><p>Features</p></div>', unsafe_allow_html=True)

st.markdown("---")

# MAIN SECTION - same as before
st.subheader("Analyze an Email")

email_subject = st.text_input("Email Subject (Optional)", placeholder="Enter the email subject...")
email_body = st.text_area("Email Content", height=250, placeholder="Paste the complete email content here including body...")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'link', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def extract_meta(text):
    has_link = 1 if "http" in text or "www" in text else 0
    has_attachment = 1 if "attachment" in text.lower() else 0
    urgency_flag = 1 if any(word in text.lower() for word in ['urgent', 'verify', 'suspended', 'click', 'account', 'password']) else 0
    return [has_link, has_attachment, urgency_flag]

if st.button("🔍 Analyze Email", type="primary", use_container_width=True):
    if email_body.strip()!= "":
        with st.spinner('Analyzing email content...'):
            full_text = email_subject + " "+ email_body
            cleaned = clean_text(full_text)
            X_tfidf = tfidf.transform([cleaned])
            X_emb = svd.transform(X_tfidf)
            X_meta = np.array([extract_meta(full_text)])
            X_final = hstack([X_tfidf, csr_matrix(X_emb), csr_matrix(X_meta)])
            pred = model.predict(X_final)[0]

            st.markdown("### Prediction Result:")
            if pred == 1:
                st.markdown('<div class="phishing-box"><h4>⚠️ ALERT: PHISHING EMAIL DETECTED</h4><p><b>Recommendation:</b> Do NOT click any links, do NOT download attachments, and do NOT reply. Delete this email immediately.</p></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="safe-box"><h4>✅ SAFE EMAIL</h4><p><b>Recommendation:</b> This email appears to be legitimate based on the trained model.</p></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter email content to analyze.")

st.markdown("---")
st.info("**Important Disclaimer:** This application provides a machine-learning prediction based on textual patterns learned from the training dataset. It does not guarantee 100% accuracy and should not be the sole basis for security decisions. Always verify suspicious emails through official channels.")

st.caption("AI-Powered Phishing Email Detection | Machine Learning & NLP Project")