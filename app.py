import streamlit as st
import os
import tempfile
from review_tool import extract_text_from_file, evaluate_dissertation, export_to_csv, export_to_pdf

st.set_page_config(page_title="Dissertation Reviewer Tool", layout="wide")
st.title("📘 Dissertation Reviewer Tool")

st.markdown("This tool evaluates dissertation or capstone files for grammar, APA citation alignment, and content structure based on rubric scoring. Upload multiple files to review them sequentially.")

with st.sidebar:
    st.header("Reviewer Info")
    project_type = st.selectbox("Select Project Type", ["Dissertation", "Capstone"])
    reviewer_id = st.text_input("Faculty Reviewer Name or ID")
    discipline = st.selectbox("Select Discipline", [
        "public administration", "applied psychology", "behavior analysis",
        "business", "counseling", "education", "human services",
        "public service", "psychology", "information technology",
        "emergency management", "clinical psychology", "higher education leadership",
        "nursing", "public health", "social work"
    ])
    uploaded_files = st.file_uploader("Upload Document(s)", type=["pdf", "docx", "txt"], accept_multiple_files=True)

if uploaded_files and reviewer_id and discipline:
    for uploaded_file in uploaded_files:
        st.markdown(f"### 📄 Reviewing: `{uploaded_file.name}`")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        text = extract_text_from_file(temp_path)
        result = evaluate_dissertation(text, discipline, project_type)

        st.subheader("Decision")
        st.success(f"Final Recommendation: {result['decision']}")

        st.subheader("Faculty Comment")
        st.info(result["rubric"].get("Faculty Comment", "No comment generated."))

        st.subheader("Rubric Scores")
        for key, value in result["rubric"].items():
            if not key.startswith("_DEBUG") and key != "Faculty Comment":
                st.write(f"**{key}**: {value}")

        st.subheader("Scores")
        st.json(result["scores"])

        st.subheader("Grammar Issues (Top 10)")
        for issue in result["details"]["grammar_issues"][:10]:
            st.markdown(f"- {issue['message']}")
            st.caption(f"Context: {issue['context']}")

        st.subheader("Citation Issues")
        for issue in result["details"]["citation_issues"]:
            st.error(issue)

        if st.button(f"✅ Save review to CSV for `{uploaded_file.name}`"):
            export_to_csv(temp_path, discipline, result, reviewer=reviewer_id)
            st.success("Review saved to dissertation_reviews.csv")

        if st.button(f"📄 Export PDF for `{uploaded_file.name}`"):
            export_to_pdf(temp_path, discipline, result, reviewer=reviewer_id)
            st.success("PDF exported.")

        os.remove(temp_path)

elif not reviewer_id or not uploaded_files:
    st.warning("Please enter your reviewer ID and upload one or more files to begin.")
