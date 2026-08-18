import streamlit as st
import pandas as pd
import pickle

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
# LOAD MACHINE LEARNING MODEL
# =====================================================

@st.cache_resource
def load_ml_model():

    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


ml_model, vectorizer = load_ml_model()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("⚙️ Settings")

st.sidebar.info(
    "Cloud deployment uses the trained Machine Learning "
    "model for reliable SMS prediction."
)

model_choice = "Machine Learning"


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
            # MACHINE LEARNING PREDICTION
            # -----------------------------------------

            message_vector = vectorizer.transform(
                [message]
            )

            prediction = ml_model.predict(
                message_vector
            )[0]

            # Confidence
            if hasattr(ml_model, "predict_proba"):

                probabilities = ml_model.predict_proba(
                    message_vector
                )[0]

                confidence = (
                    max(probabilities) * 100
                )

            else:

                confidence = 100.0


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