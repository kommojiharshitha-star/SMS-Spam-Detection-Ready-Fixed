
import pickle
import streamlit as st
from preprocess import preprocess_text

model = pickle.load(open('spam_model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

st.set_page_config(page_title='SMS Spam Detection', page_icon='📩')
st.title('📩 SMS Spam Detection')
st.write('Enter an SMS message and check whether it is Spam or Ham.')

message = st.text_area('Enter SMS Message')

if st.button('Predict'):
    if message.strip():
        processed = preprocess_text(message)
        vector = vectorizer.transform([processed])
        pred = model.predict(vector)[0]
        prob = model.predict_proba(vector)[0]
        if pred == 1:
            st.error('Prediction: SPAM')
            st.metric('Confidence', f'{prob[1]*100:.2f}%')
        else:
            st.success('Prediction: HAM (Not Spam)')
            st.metric('Confidence', f'{prob[0]*100:.2f}%')
    else:
        st.warning('Please enter a message.')
