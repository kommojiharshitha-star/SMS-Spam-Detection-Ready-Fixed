import streamlit as st
import pandas as pd
import pickle
import os
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from ai_agent import SMSAIAgent


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI SMS Spam Detection",
    page_icon="🛡️",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🛡️ AI Agent for SMS Spam Detection")
st.write(
    "Intelligent SMS classification using Machine Learning, "
    "BiLSTM, DistilBERT, BERT and AI-based filtering."
)

st.divider()


# =====================================================
# LOAD AI AGENT
# =====================================================

agent = SMSAIAgent()


# =====================================================
# LOAD DISTILBERT
# =====================================================

@st.cache_resource
def load_distilbert():

    model_path = "distilbert_model"

    if not os.path.exists(model_path):
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


distilbert_tokenizer, distilbert_model = load_distilbert()


# =====================================================
# DISTILBERT PREDICTION
# =====================================================

def predict_distilbert(message):

    inputs = distilbert_tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = distilbert_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        prediction
    ].item()

    return prediction, confidence


# =====================================================
# LOAD BERT
# =====================================================

@st.cache_resource
def load_bert():

    model_path = "bert_model"

    if not os.path.exists(model_path):
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.eval()

    return tokenizer, model


bert_tokenizer, bert_model = load_bert()


# =====================================================
# BERT PREDICTION
# =====================================================

def predict_bert(message):

    inputs = bert_tokenizer(
        message,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = bert_model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        prediction
    ].item()

    return prediction, confidence


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select AI Model",
    [
        "DistilBERT",
        "BERT"
    ]
)

st.sidebar.info(
    "DistilBERT achieved the best overall "
    "F1-score in our evaluation."
)


# =====================================================
# MESSAGE INPUT
# =====================================================

st.subheader("📩 Enter SMS Message")

message = st.text_area(
    "Type or paste a message below:",
    height=150,
    placeholder="Example: Congratulations! You won a prize. Click now!"
)


# =====================================================
# ANALYZE BUTTON
# =====================================================

if st.button(
    "🔍 Analyze Message",
    type="primary"
):

    if not message.strip():

        st.warning(
            "Please enter an SMS message."
        )

    else:

        with st.spinner(
            "Analyzing message..."
        ):

            # -----------------------------------------
            # MODEL PREDICTION
            # -----------------------------------------

            if model_choice == "DistilBERT":

                if distilbert_model is None:

                    st.error(
                        "DistilBERT model not found."
                    )

                    st.stop()

                prediction, confidence = (
                    predict_distilbert(
                        message
                    )
                )

            else:

                if bert_model is None:

                    st.error(
                        "BERT model not found."
                    )

                    st.stop()

                prediction, confidence = (
                    predict_bert(
                        message
                    )
                )


            # -----------------------------------------
            # AI AGENT
            # -----------------------------------------

            result = agent.analyze_message(
                message,
                prediction,
                confidence
            )


        st.divider()

        st.subheader(
            "🤖 AI Agent Analysis"
        )


        # -----------------------------------------
        # RESULT COLUMNS
        # -----------------------------------------

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Prediction",
                result["prediction"]
            )


        with col2:

            st.metric(
                "Confidence",
                f"{result['confidence']:.2f}%"
            )


        with col3:

            st.metric(
                "Risk Score",
                f"{result['risk_score']:.2f}"
            )


        with col4:

            st.metric(
                "Action",
                result["action"]
            )


        # -----------------------------------------
        # ACTION MESSAGE
        # -----------------------------------------

        if result["action"] == "BLOCK":

            st.error(
                "🚨 SPAM MESSAGE — BLOCKED"
            )

        elif result["action"] == "WARN":

            st.warning(
                "⚠️ SUSPICIOUS MESSAGE — REVIEW"
            )

        elif result["action"] == "ALLOW":

            st.success(
                "✅ NORMAL MESSAGE — ALLOWED"
            )


        # -----------------------------------------
        # EXPLANATION
        # -----------------------------------------

        st.subheader(
            "🧠 Why did the AI Agent decide this?"
        )

        st.info(
            result["reason"]
        )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.divider()

st.subheader(
    "📊 Model Performance Comparison"
)


comparison_data = {

    "Model": [
        "BiLSTM",
        "DistilBERT",
        "BERT"
    ],

    "Accuracy": [
        "98.65%",
        "99.37%",
        "99.19%"
    ],

    "Precision": [
        "98.55%",
        "99.31%",
        "97.30%"
    ],

    "Recall": [
        "91.28%",
        "95.97%",
        "96.64%"
    ],

    "F1-Score": [
        "94.77%",
        "97.61%",
        "96.97%"
    ]
}


comparison_df = pd.DataFrame(
    comparison_data
)


st.dataframe(
    comparison_df,
    use_container_width=True,
    hide_index=True
)


st.success(
    "🏆 Best Overall Model: DistilBERT "
    "(99.37% Accuracy, 97.61% F1-score)"
)


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Agent for SMS Spam Detection and "
    "Intelligent Message Filtering | "
    "Data Science Project"
)