from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import spacy
import re
import os
from datetime import datetime
from werkzeug.utils import secure_filename
import pdfplumber
import logging

app = Flask(__name__)

# --- Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        app.logger.info(f"Created upload folder: {UPLOAD_FOLDER}")
    except OSError as e:
        app.logger.error(f"Could not create upload folder {UPLOAD_FOLDER}: {e}")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model Initialization ---
summarizer_pipeline = None
nlp = None

try:
    app.logger.info("Initializing summarization pipeline...")
    summarizer_pipeline = pipeline(
        "summarization",
        model="google/flan-t5-small",
        device=-1, # Use CPU
        truncation=True
    )
    app.logger.info("Summarizer pipeline (google/flan-t5-small) initialized successfully.")
except Exception as e:
    app.logger.warning(f"Could not initialize summarizer pipeline: {e}. Summarization will be disabled.")

try:
    app.logger.info("Initializing spaCy NER model...")
    nlp = spacy.load("en_core_web_sm", exclude=["parser", "tagger", "lemmatizer"])
    app.logger.info("spaCy NER model (en_core_web_sm) loaded successfully.")
except OSError:
    app.logger.warning("spaCy model 'en_core_web_sm' not found. Attempting to download...")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm", exclude=["parser", "tagger", "lemmatizer"])
        app.logger.info("spaCy model downloaded and loaded successfully.")
    except Exception as e:
        app.logger.error(f"Could not download or load spaCy model: {e}. NER features will be limited.")

# --- In-Memory Storage ---
parsed_resumes = [] # Stores parsed data for current session

# --- Helper Functions ---
def chunk_text_for_model(text, model_max_length=1024, sentence_split=True):
    chunks = []
    if not text or not text.strip(): return chunks
    if sentence_split:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= model_max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk.strip(): chunks.append(current_chunk.strip())
                if len(sentence) > model_max_length: # Hard split for very long sentence
                    for i in range(0, len(sentence), model_max_length): chunks.append(sentence[i:i+model_max_length])
                else: current_chunk = sentence + " "
        if current_chunk.strip(): chunks.append(current_chunk.strip())
    else: # Simple hard split
        for i in range(0, len(text), model_max_length): chunks.append(text[i:i+model_max_length])
    
    final_chunks = [] # Ensure all chunks are within limit
    for chunk in chunks:
        if len(chunk) > model_max_length:
            for i in range(0, len(chunk), model_max_length): final_chunks.append(chunk[i:i+model_max_length])
        elif chunk.strip(): final_chunks.append(chunk)
    return final_chunks

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                app.logger.warning(f"No pages found in PDF: {pdf_path}")
                return None
            for i, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text(x_tolerance=1, y_tolerance=3, layout=False) # layout=False can sometimes help with dense text
                    if page_text:
                        page_text = re.sub(r'\s*\n\s*', '\n', page_text) 
                        page_text = re.sub(r'-\n(\w)', r'\1', page_text) # De-hyphenate words split across lines
                        text += page_text.strip() + "\n\n" 
                except Exception as e:
                    app.logger.warning(f"Could not extract text from page {i+1} in {pdf_path}: {e}")
        if not text.strip():
            app.logger.warning(f"No text extracted from PDF {pdf_path}. It might be image-based or corrupted.")
            return None
        return text.strip()
    except Exception as e:
        app.logger.error(f"Error processing PDF {pdf_path}: {e}", exc_info=True)
        return None

def extract_basic_info(text):
    info = {'name': 'N/A', 'email': 'N/A', 'phone': 'N/A'}
    if not text: return info
    
    name_found = False
    # Try to get name from the first few lines using a more specific pattern
    # Pattern: Two or more capitalized words, potentially with a middle initial/name.
    name_pattern = r"^([A-Z][a-z']+(?:\s[A-Z][a-z']\.?)*(?:\s[A-Z][a-z']+)+)"
    lines = text.split('\n')
    for line in lines[:5]: # Check first 5 lines
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) > 70: continue # Skip empty or very long lines
        if '@' in line_stripped or "http" in line_stripped or re.search(r'\d{3,}', line_stripped): continue # Skip lines with email/url/many numbers
        
        match = re.match(name_pattern, line_stripped)
        if match and len(match.group(1).split()) >= 2 : # Ensure at least two words for a name
            info['name'] = match.group(1)
            name_found = True
            break
    if not name_found and lines and lines[0].strip() and len(lines[0].strip().split()) >=2 and len(lines[0].strip().split()) < 5: # Fallback
         info['name'] = lines[0].strip()


    email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    if email_match: info['email'] = email_match.group(0)

    phone_match = re.search(r'(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(\s*\d{2,4}\s*\)|\d{2,4})[-.\s]?\d{3,4}[-.\s]?\d{3,4}(?:[-.\s]?\d{1,4})?\b|\b\d{10}\b)', text)
    if phone_match: info['phone'] = phone_match.group(0).strip()
    return info

def extract_skills_keywords(text):
    if not text: return []
    skills = set()
    skill_keywords = [ # Using the expanded list from previous iterations
        'python', 'java', 'javascript', 'c\+\+', 'c#', 'c', 'ruby', 'php', 'swift', 'kotlin', 'golang', 'go', 'rust', 'scala', 'typescript',
        'sql', 'mysql', 'postgresql', 'mongodb', 'nosql', 'oracle', 'sqlite', 'cassandra', 'redis', 't-sql', 'pl/sql',
        'html', 'html5', 'css', 'css3', 'react', 'reactjs', 'react.js', 'angular', 'angularjs', 'vue', 'vuejs', 'vue.js', 'jquery', 'bootstrap', 'tailwind', 'sass', 'less',
        'node.js', 'nodejs', 'express.js', 'expressjs', 'next.js', 'nestjs',
        'django', 'flask', 'ruby on rails', 'laravel', 'spring framework', 'spring boot', '.net', 'asp.net',
        'android', 'ios', 'react native', 'flutter', 'xamarin', 'swiftui', 'jetpack compose',
        'aws', 'azure', 'gcp', 'google cloud platform', 'amazon web services', 'oracle cloud', 'ibm cloud', 'oci',
        'ec2', 's3', 'lambda', 'fargate', 'rds', 'aurora', 'dynamodb', 'elasticache', 'beanstalk', 'ecs', 'eks', 'vpc', 'route 53', 'iam', 'cloudformation', 'cloudwatch', 'sns', 'sqs', 'api gateway', 'aws bedrock', 'sagemaker',
        'azure functions', 'azure blob storage', 'azure sql database', 'azure cosmos db', 'azure kubernetes service', 'aks', 'azure devops',
        'vertex ai', 'google compute engine', 'gce', 'google kubernetes engine', 'gke', 'bigquery', 'cloud functions', 'pub/sub', 'google cloud storage', 'gcs',
        'docker', 'kubernetes', 'k8s', 'openshift', 'terraform', 'ansible', 'puppet', 'chef', 'jenkins', 'ci/cd', 'gitlab ci', 'github actions', 'circleci',
        'git', 'github', 'gitlab', 'bitbucket', 'svn', 'jira', 'confluence', 'slack', 'trello', 'asana',
        'agile', 'scrum', 'kanban', 'lean', 'devops', 'mlops', 'sre', 'site reliability engineering',
        'machine learning', 'ml', 'deep learning', 'dl', 'artificial intelligence', 'ai', 'neural networks', 'ann',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'opencv', 'nltk', 'spacy',
        'data analysis', 'data analytics', 'data science', 'data visualization', 'data mining', 'data engineering', 'business intelligence', 'bi', 'tableau', 'power bi', 'qlik', 'excel', 'vba',
        'statistics', 'nlp', 'natural language processing', 'computer vision', 'cv', 'svm', 'support vector machine', 'random forest', 'gradient boosting', 'xgboost', 'lightgbm', 'decision trees', 'knn', 'k-nearest neighbors', 'clustering', 'regression', 'classification',
        'cnn', 'convolutional neural network', 'rnn', 'recurrent neural network', 'lstm', 'long short-term memory', 'transformer', 'bert', 'gpt', 'gan', 'generative adversarial network', 'resnet', 'yolo',
        'cybersecurity', 'information security', 'infosec', 'penetration testing', 'pentest', 'network security', 'siem', 'soc', 'firewall', 'encryption', 'cryptography', 'owasp',
        'linux', 'unix', 'windows server', 'macos', 'bash', 'powershell', 'shell scripting',
        'api', 'rest', 'restful apis', 'graphql', 'microservices', 'soa', 'rpc', 'grpc', 'soap',
        'big data', 'hadoop', 'spark', 'apache spark', 'kafka', 'apache kafka', 'flink', 'elasticsearch', 'logstash', 'kibana', 'elk stack', 'splunk',
        'object-oriented programming', 'oop', 'functional programming', 'fp',
        'system design', 'software architecture', 'algorithms', 'data structures', 'problem solving', 'optimization',
        'jupyter', 'jupyter notebook', 'vs code', 'visual studio code', 'colab', 'google colab', 'r language', 'matlab', 'sas'
    ]
    text_lower = text.lower()
    for skill_pattern in skill_keywords:
        # Escape special regex characters in skill_pattern for safe use in re.search
        # except for `+` if it's for c++ or c# (handled by explicit `c\+\+`)
        escaped_skill = re.escape(skill_pattern)
        pattern = r'\b' + escaped_skill + r'\b'
        try:
            if re.search(pattern, text_lower):
                skills.add(skill_pattern.replace(r'\+', '+').replace(r'\.', '.')) # Add original unescaped skill
        except re.error:
            app.logger.warning(f"Regex error for skill pattern: {skill_pattern}")
    return sorted(list(skills))


def extract_experience_section_text(text):
    section_text = ""
    # Prioritize "WORK EXPERIENCE" as it's common and specific
    headers = [r"WORK EXPERIENCE", r"EXPERIENCE", r"EMPLOYMENT HISTORY", r"PROFESSIONAL EXPERIENCE", r"CAREER HISTORY"]
    
    for header in headers:
        # Match header at the beginning of a line, possibly indented
        match = re.search(r'^[ \t]*' + header + r'[ \t]*\n', text, re.IGNORECASE | re.MULTILINE)
        if match:
            start_index = match.end()
            # Define terminators for the experience section
            terminators = [
                r"EDUCATION", r"QUALIFICATIONS", r"SKILLS", r"TECHNICAL SKILLS", r"PROJECTS", 
                r"PERSONAL PROJECTS", r"CERTIFICATIONS", r"AWARDS", r"ACHIEVEMENTS", 
                r"PUBLICATIONS", r"LANGUAGES", r"REFERENCES", r"ADDITIONAL INFORMATION"
            ]
            min_end_index = len(text)
            for term in terminators:
                term_match = re.search(r'^[ \t]*' + term + r'[ \t]*\n', text[start_index:], re.IGNORECASE | re.MULTILINE)
                if term_match:
                    min_end_index = min(min_end_index, start_index + term_match.start())
            
            section_text = text[start_index:min_end_index].strip()
            if section_text:
                app.logger.info(f"Extracted Work Experience section using header: '{header}'")
                return section_text
    
    app.logger.info("Work Experience section not clearly demarcated by main headers.")
    return "" # Return empty if no clear section found

def extract_experience_entries(text):
    experience_section = extract_experience_section_text(text)
    if not experience_section:
        return []

    entries = []
    # Regex to capture job title, company, and date range. This is complex and resume-dependent.
    # This pattern tries to find a block starting with a potential job title,
    # then a line that could be a company (often with " - "), and a date range.
    # Assumes job entries are separated by at least one blank line or a new job title pattern.
    
    # Split section into paragraphs or blocks
    blocks = re.split(r'\n\s*\n', experience_section) # Split by blank lines
    
    current_entry = {}
    date_pattern = re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)[\s.,]*\d{4}\s*(?:-|to|–|—)\s*(?:Present|Current|Till Date|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)[\s.,]*\d{4})\b', re.IGNORECASE)
    
    # For the user's specific format: "Job Title - Company Name \n Date Range" or "Job Title - Company Name Date Range"
    # This is a simplified heuristic based on the user's sample.
    # A more robust parser would use NLP techniques or more sophisticated grammars.
    
    # Let's try to capture entries that have a clear structure:
    # Line 1: Job Title - Company (or just Job Title)
    # Line 2 (optional): Company (if not on line 1)
    # Line with Date Range
    # Lines following: Description (bullet points often start with •, *, -, or are indented)

    # Iterating through lines of the experience section:
    lines = experience_section.split('\n')
    entry_buffer = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped: # Potential end of an entry
            if entry_buffer:
                entries.append(" ".join(entry_buffer))
                entry_buffer = []
            continue
        entry_buffer.append(line_stripped)
    if entry_buffer: # Add last entry
        entries.append(" ".join(entry_buffer))

    # Filter and format the entries based on common patterns
    # This is a very basic filter.
    formatted_entries = []
    for entry_text in entries:
        # Heuristic: an entry should have at least a few words and likely a date
        if len(entry_text.split()) > 3 and date_pattern.search(entry_text):
            # Attempt to structure it a bit for display
            # Find date first
            date_match = date_pattern.search(entry_text)
            dates = date_match.group(0) if date_match else "Dates N/A"
            
            # Everything before the date could be title/company/description
            before_dates = entry_text[:date_match.start()].strip() if date_match else entry_text
            
            # Simple split for title/company if " - " is present in the first part
            title_company_parts = before_dates.split(" - ", 1)
            if len(title_company_parts) == 2:
                job_title = title_company_parts[0].strip()
                company_and_desc = title_company_parts[1].strip()
                # Simple assumption: if company part is short, it's company, rest is start of desc.
                company_parts = company_and_desc.split(None, 2) # Split first two words
                company = " ".join(company_parts[:2]) if len(company_parts)>1 else company_and_desc
                description_start = company_parts[2] if len(company_parts)>2 else ""

                formatted_entries.append(f"**{job_title}** at **{company}** ({dates}). {description_start}".strip())
            else: # Could not clearly separate title and company
                 # Try to extract a potential job title from the beginning of 'before_dates'
                first_few_words = " ".join(before_dates.split()[:4]) # e.g., "Machine Learning Engineer Intern"
                formatted_entries.append(f"**{first_few_words}...** ({dates}). {before_dates[len(first_few_words):]}".strip())

    if formatted_entries:
        return formatted_entries
    
    # Fallback if above fails, return raw valid entries
    return [e for e in entries if len(e.split()) > 3 and date_pattern.search(e)] if entries else []


def perform_ner(text, skills_list):
    ner_entities = {}
    if not nlp or not text: return ner_entities

    doc_chunks = chunk_text_for_model(text, nlp.max_length - 100, sentence_split=False)
    skills_lower = [skill.lower() for skill in skills_list]

    for chunk in doc_chunks:
        try:
            doc = nlp(chunk)
            for ent in doc.ents:
                ent_text = ent.text.strip()
                ent_text_lower = ent_text.lower()
                label = ent.label_

                if not ent_text or len(ent_text) < 2 and ent_text_lower not in ['c', 'r']: continue # Skip empty or very short
                if ent_text_lower in skills_lower and label in ["ORG", "PERSON", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT", "FAC"]: continue
                
                common_headers_lower = ["summary", "experience", "education", "skills", "projects", "awards", "honors", "certifications", "references", "contact", "profile"]
                if ent_text_lower in common_headers_lower and label != "MISC": continue # Allow MISC for headers

                # Avoid adding numbers or date-like strings if they are misclassified as ORG/PERSON etc.
                if label in ["ORG", "PERSON", "GPE", "LOC"] and (ent_text.isdigit() or re.match(r"^\d{1,2}/\d{4}$|^\d{4}$", ent_text)):
                    continue

                if label not in ner_entities: ner_entities[label] = []
                
                is_present = any(existing_ent.lower() == ent_text_lower for existing_ent in ner_entities[label])
                if not is_present: ner_entities[label].append(ent_text)
        except Exception as e:
            app.logger.error(f"Error during NER processing of a chunk: {e}")
    for label in ner_entities: ner_entities[label].sort()
    return ner_entities

def extract_professional_summary_section_text(text):
    headers = [r"PROFESSIONAL SUMMARY", r"SUMMARY", r"CAREER OBJECTIVE", r"OBJECTIVE", r"PROFILE", r"ABOUT ME"]
    for header in headers:
        match = re.search(r'^[ \t]*' + header + r'[ \t]*\n', text, re.IGNORECASE | re.MULTILINE)
        if match:
            start_index = match.end()
            content_after_summary = text[start_index:].lstrip()
            
            end_delimiters_patterns = [ # Patterns that mark the end of the summary section
                r'^[ \t]*(?:WORK EXPERIENCE|EXPERIENCE|EMPLOYMENT HISTORY|PROFESSIONAL EXPERIENCE|CAREER HISTORY)[ \t]*\n',
                r'^[ \t]*(?:EDUCATION|ACADEMIC QUALIFICATIONS|QUALIFICATIONS)[ \t]*\n',
                r'^[ \t]*(?:SKILLS|TECHNICAL SKILLS|KEY SKILLSET)[ \t]*\n',
                r'^[ \t]*(?:PROJECTS|PERSONAL PROJECTS)[ \t]*\n',
                r'\n\s*\n\s*\n' # Three or more newlines often separate major sections
            ]
            
            min_end_pos = len(content_after_summary)
            for pattern in end_delimiters_patterns:
                end_match = re.search(pattern, content_after_summary, re.IGNORECASE | re.MULTILINE)
                if end_match:
                    min_end_pos = min(min_end_pos, end_match.start())
            
            section_text = content_after_summary[:min_end_pos].strip()
            if section_text:
                app.logger.info(f"Identified Professional Summary section using header: '{header}'")
                return section_text
    app.logger.info("Dedicated Professional Summary section not explicitly found by header.")
    return None

def generate_summary_from_text(text_to_summarize, max_len=100, min_len=30): # Increased default max_len
    if not summarizer_pipeline or not text_to_summarize or not text_to_summarize.strip() or len(text_to_summarize.split()) < min_len : # Check word count
        return "Summary not available (text too short or summarizer disabled)."

    input_chunks = chunk_text_for_model(text_to_summarize, model_max_length=600, sentence_split=True) # Smaller chunks for better focus
    if not input_chunks: return "Text could not be chunked for summarization."

    summaries = []
    for i, chunk in enumerate(input_chunks):
        if not chunk.strip() or len(chunk.split()) < 10: continue # Skip very short chunks
        try:
            # Dynamically adjust max_length based on chunk length
            # For flan-t5, it's more about input token limit (~512 for small) than output length directly dictating input
            current_max_len = max(min_len, int(len(chunk.split()) * 0.6)) # Target 60% of words
            current_max_len = min(max_len, current_max_len) # Cap at overall max_len

            result = summarizer_pipeline(
                chunk,
                max_length=current_max_len,
                min_length=min_len,
                num_beams=4,
                early_stopping=True,
                # no_repeat_ngram_size=2 # Can help reduce repetition
            )
            if result and isinstance(result, list) and result[0] and 'summary_text' in result[0]:
                summaries.append(result[0]['summary_text'].strip())
            else:
                app.logger.warning(f"Summarization pipeline returned unexpected result for chunk {i+1}.")
        except Exception as e:
            app.logger.error(f"Error summarizing chunk {i+1}: {e}", exc_info=True)
    
    if not summaries: return "Could not generate summary from provided text."
    
    final_summary = ' '.join(summaries)
    # Basic post-processing: capitalize first letter, ensure ends with a period.
    if final_summary:
        final_summary = final_summary[0].upper() + final_summary[1:]
        if not final_summary.endswith('.'): final_summary += '.'
    if len(input_chunks) > 1 and len(summaries) > 0 : final_summary += " (Compiled)"
    return final_summary


def generate_overall_summary(full_text):
    professional_summary_text = extract_professional_summary_section_text(full_text)
    if professional_summary_text:
        summary = generate_summary_from_text(professional_summary_text)
        if not summary.startswith("Summary not available") and not summary.startswith("Could not generate"):
            return summary
    
    app.logger.info("Falling back to summarizing the beginning of the document for overall summary.")
    first_meaningful_block = ""
    lines = full_text.split('\n')
    temp_lines = []
    word_count = 0
    for line in lines:
        stripped_line = line.strip()
        if stripped_line: # Consider non-empty lines
            temp_lines.append(stripped_line)
            word_count += len(stripped_line.split())
            if word_count > 200: # Roughly first 200 words
                break
    first_meaningful_block = "\n".join(temp_lines)

    if not first_meaningful_block.strip():
        return "Summary not available: Document too short or no suitable content found."
    return generate_summary_from_text(first_meaningful_block)

def parse_resume_text_content(full_text, filename="N/A"):
    if not full_text or not full_text.strip(): # Handle empty or whitespace-only text
        return { 'filename': filename, 'error': 'Input text was empty or null.', 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S') }

    contact_info = extract_basic_info(full_text)
    skills = extract_skills_keywords(full_text)
    experience_entries = extract_experience_entries(full_text)
    ner_entities = perform_ner(full_text, skills)
    summary = generate_overall_summary(full_text)

    return {
        'filename': filename,
        'contact_info': contact_info,
        'skills': skills,
        'experience_phrases': experience_entries, # Ensure frontend uses this key
        'ner_entities': ner_entities,
        'summary': summary,
        'original_text_length': len(full_text),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume_route(): # Renamed to avoid conflict with any potential 'upload' variable
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part', 'details': 'Please select a file to upload.'}), 400
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({'error': 'No selected file', 'details': 'Please select a file to upload.'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            app.logger.error(f"Error saving file '{filename}': {e}", exc_info=True)
            return jsonify({'error': 'File save failed', 'details': f'Could not save file: {str(e)}'}), 500

        resume_text_content = None
        file_ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

        if file_ext == 'pdf': resume_text_content = extract_text_from_pdf(filepath)
        elif file_ext in ['txt', 'md', 'rtf']:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: resume_text_content = f.read()
            except Exception as e:
                return jsonify({'error': 'Text file read failed','details': f'Could not read: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Unsupported file type','details': f'Supports PDF, TXT, MD, RTF. Got: {file_ext}'}), 400

        if not resume_text_content:
            return jsonify({'error': 'No text extracted','details': 'File empty, image-based, or corrupted.'}), 400

        parsed_data = parse_resume_text_content(resume_text_content, filename)
        if 'error' in parsed_data: # Check if parsing itself returned an error object
             return jsonify({'error': 'Parsing failed', 'details': parsed_data['error']}), 500

        # Store results for later access (in-memory for this version)
        # Avoid duplicates by filename for simplicity in this session
        parsed_resumes[:] = [r for r in parsed_resumes if r.get('filename') != parsed_data.get('filename')]
        parsed_resumes.append(parsed_data)
        
        app.logger.info(f"Resume '{filename}' parsed. Total stored: {len(parsed_resumes)}")
        return jsonify({
            'message': 'Resume parsed successfully!',
            'data': parsed_data,
            'debug_info': {
                'file_size_bytes': os.path.getsize(filepath) if os.path.exists(filepath) else -1,
                'extracted_text_length': len(resume_text_content),
                'num_skills_found': len(parsed_data.get('skills', [])),
                'num_experience_entries': len(parsed_data.get('experience_phrases', [])) # Changed key
            }
        }), 200
    except Exception as e:
        app.logger.error(f"Unexpected error in /upload: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@app.route('/search', methods=['GET'])
def search_resumes_route():
    try:
        query_term = request.args.get('query', '').lower().strip() # Renamed to avoid conflict
        if not query_term:
            return jsonify({'results': [], 'total': 0, 'message': 'Please provide a search query.'}), 400

        results = []
        for resume in parsed_resumes:
            # Build a searchable string (ensure all fields exist)
            content = [
                str(resume.get('filename', '')).lower(),
                str(resume.get('contact_info', {}).get('name', '')).lower(),
                str(resume.get('contact_info', {}).get('email', '')).lower(),
                ' '.join(resume.get('skills', [])).lower(), # Skills are already lowercased
                ' '.join(str(exp) for exp in resume.get('experience_phrases', [])).lower(), # Use experience_phrases
                str(resume.get('summary', '')).lower()
            ]
            ner_texts = []
            for entities_list in resume.get('ner_entities', {}).values():
                ner_texts.extend([str(entity).lower() for entity in entities_list])
            content.append(' '.join(ner_texts))
            full_search_text = ' '.join(filter(None, content))

            if query_term in full_search_text:
                results.append({ # Return a subset for search results display
                    'filename': resume.get('filename'),
                    'name': resume.get('contact_info', {}).get('name'),
                    'email': resume.get('contact_info', {}).get('email'),
                    'skills_preview': resume.get('skills', [])[:5], # First 5 skills
                    'summary_preview': (resume.get('summary', '')[:150] + '...') if resume.get('summary') else 'N/A'
                })
        return jsonify({'query': query_term, 'results': results, 'total': len(results)}), 200
    except Exception as e:
        app.logger.error(f"Error during search: {e}", exc_info=True)
        return jsonify({'error': 'Search failed', 'details': str(e)}), 500

# --- Main Application Execution ---
if __name__ == '__main__':
    templates_dir = 'templates'
    index_html_path = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(templates_dir): os.makedirs(templates_dir)
    # Ensure 'templates/index.html' exists from the next code block (frontend)
    app.run(debug=False, host='0.0.0.0', port=5001)