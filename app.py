import streamlit as st

from src.app import process_query

st.set_page_config(page_title="Smart Support Desk", layout="wide")
st.title("Smart Support Desk")
st.write("Ask the support assistant about company policies, technical issues, billing, or escalation requests.")

query = st.text_area("Enter your question:", height=180)
user_id = st.text_input("User ID (optional)")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Processing your question..."):
            result = process_query(query, user_id=int(user_id) if user_id.isdigit() else None)

        st.subheader("Response")
        st.markdown(result["response"])

        st.subheader("Classifier")
        st.json({
            "persona": result["persona"],
            "intent": result["intent"],
            "confidence": result["confidence"],
        })

        st.subheader("Escalation")
        st.json(result["escalation"])

        if result["escalation"]["escalate"]:
            st.warning("This interaction was marked for escalation.")

        st.subheader("Retrieved Context")
        st.text_area("Context used for generation", value=result["context"], height=240)
