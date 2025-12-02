# Resume to Job Description Matcher

This application helps you analyze how well your resume matches a job description by providing a visual representation of skill matches and overall similarity.

## Features

- Upload your resume in PDF or DOCX format
- Paste the job description text
- View skill match percentage
- See matched and missing skills
- Visual representation with pie charts and bar graphs
- Overall text similarity score

## Installation

1. Make sure you have Python 3.8+ installed
2. Clone this repository
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   streamlit run app.py
   ```
2. Open your browser and navigate to `http://localhost:8501`
3. Upload your resume and paste the job description
4. Click "Analyze Match" to see the results

## How It Works

1. The application extracts text from your resume (PDF or DOCX)
2. Processes both the resume and job description text
3. Uses TF-IDF vectorization to calculate text similarity
4. Extracts and matches skills from both texts
5. Displays visual representations of the match

## Note

This is a basic implementation. For more accurate results, you might want to:
- Expand the skills dictionary
- Implement more sophisticated text processing
- Add support for more file formats
- Include machine learning models for better matching
