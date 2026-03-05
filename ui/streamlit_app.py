import requests
import streamlit as st

API_URL = st.secrets["API_URL"]

st.set_page_config(page_title="News Classifier", page_icon="📰")
st.title("📰 News Classifier")
st.write(
    "Enter a news snippet below to classify it into: **World, Sports, Business, or Sci/Tech**."
)

user_text = st.text_area("Article Text", height=200, placeholder="Paste news text here...")

if st.button("Classify", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # 15-second timeout in case the free Hugging Face space is waking up
                response = requests.post(API_URL, json={"text": user_text}, timeout=15)

                if response.status_code == 200:
                    data = response.json()

                    st.success(f"**Prediction:** {data['label']}")
                    st.info(f"**Confidence:** {data['confidence']:.1%}")

                    # Show off the custom middleware tracing ID
                    st.caption(f"Trace ID: `{data.get('request_id', 'N/A')}`")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")

            except requests.exceptions.Timeout:
                st.error(
                    "The API is taking a while to respond (the server might be waking up from sleep). Please try again!"
                )
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
