import streamlit as st
import PyPDF2
import docx
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pandas as pd
import io

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def calculate_similarity(jd_text, resume_text):
    # Preprocess both texts
    jd_processed = preprocess_text(jd_text)
    resume_processed = preprocess_text(resume_text)
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([jd_processed, resume_processed])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity * 100  # Convert to percentage

def extract_skills(text):
    # This is a simple skill extractor - you might want to enhance it
    skills = [
        'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
        'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask',
        'machine learning', 'deep learning', 'ai', 'data analysis', 'sql', 'nosql',
        'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'git', 'agile', 'scrum'
    ]
    
    found_skills = []
    for skill in skills:
        if skill in text.lower():
            found_skills.append(skill)
    return found_skills

def plot_skill_match(match_percentage):
    # Create a smaller figure with just the pie chart
    fig, ax = plt.subplots(figsize=(2, 2))  # Made it even smaller (2x2)
    
    # Pie Chart with smaller text
    ax.pie(
        [match_percentage, 100 - match_percentage],
        labels=["Matched", "Not Matched"],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#4CAF50', '#F44336'],
        textprops={'fontsize': 8},  # Even smaller font
        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
        labeldistance=0.8  # Bring labels closer to the pie
    )
    ax.axis('equal')
    plt.title('Skill Match', pad=5, fontsize=9)  # Smaller title and padding
    
    # Adjust layout to minimize whitespace
    plt.tight_layout(pad=0.5)
    
    return fig


def main():
    st.set_page_config(page_title="Resume Matcher", layout="wide")
    st.title("Resume to Job Description Matcher")
    st.write("Upload your resume and paste the job description to see how well they match!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload Your Resume")
        resume_file = st.file_uploader("Choose a file", type=['pdf', 'docx'])
        
        if resume_file is not None:
            file_extension = resume_file.name.split('.')[-1].lower()
            
            try:
                if file_extension == 'pdf':
                    resume_text = extract_text_from_pdf(resume_file)
                elif file_extension == 'docx':
                    resume_text = extract_text_from_docx(resume_file)
                else:
                    st.error("Unsupported file format. Please upload a PDF or DOCX file.")
                    return
                
                st.session_state['resume_text'] = resume_text
                st.success("Resume uploaded successfully!")
                
                # Display extracted skills
                skills = extract_skills(resume_text)
                if skills:
                    st.subheader("Skills found in your resume:")
                    st.write(", ".join(skills))
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with col2:
        st.header("Job Description")
        jd_text = st.text_area("Paste the job description here", height=300)
        
        if st.button("Analyze Match"):
            if 'resume_text' not in st.session_state or not jd_text:
                st.warning("Please upload a resume and enter a job description.")
                return
            
            with st.spinner('Analyzing...'):
                # Calculate similarity
                similarity = calculate_similarity(jd_text, st.session_state['resume_text'])
                
                # Extract skills from JD
                jd_skills = extract_skills(jd_text)
                resume_skills = extract_skills(st.session_state['resume_text'])
                
                # Find matching skills
                matched_skills = set(resume_skills) & set(jd_skills)
                unmatched_skills = set(jd_skills) - set(resume_skills)
                
                # Calculate match percentage based on JD skills
                if jd_skills:
                    match_percentage = (len(matched_skills) / len(jd_skills)) * 100
                else:
                    match_percentage = 0
                st.markdown(
                    """
                    <h2 style='text-align: center; color:#222; margin-top:30px;'>
                        Match Analysis
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                # Center the percentage
                st.markdown(
                    f"""
                    <h1 style='text-align: center; font-size: 42px; color:#000;'>
                        {match_percentage:.1f}% Match
                    </h1>
                    """,
                    unsafe_allow_html=True
                )
                # Create a centered container for the pie chart
                st.markdown("""
                <div style='text-align: center; width: 100%; margin: 10px 0;'>
                """, unsafe_allow_html=True)
                
                # Display the pie chart with a smaller size
                fig = plot_skill_match(match_percentage)
                st.pyplot(fig, use_container_width=False)
                
                # Close the container
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Skills section in two columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Matched Skills")
                    if matched_skills:
                        st.write(", ".join(matched_skills))
                    else:
                        st.write("No matching skills found.")
                
                with col2:
                    st.subheader("Missing Skills")
                    if unmatched_skills:
                        st.write(", ".join(unmatched_skills))
                    else:
                        st.write("No missing skills found!")
                
                # Display similarity score
                st.subheader("Text Similarity")
                st.write(f"The overall text similarity between your resume and the job description is: {similarity:.1f}%")
                
                # Add some styling
                st.markdown("""
                <style>
                .stProgress > div > div > div > div {
                    background-color: #66b3ff;
                }
                </style>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
