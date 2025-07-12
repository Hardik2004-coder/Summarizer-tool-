import os
import json
import warnings
import re
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS

# For local summarization
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Ollama import (only for Q&A)
# Ensure Ollama is running and the model (e.g., llama3) is pulled:
# ollama pull llama3
import ollama

# Document processing imports
import PyPDF2
from docx import Document
import io
import base64
import traceback

# Imports essential for existing functionality and robustness
from werkzeug.utils import secure_filename
import uuid
import mimetypes

# MongoDB imports
from pymongo import MongoClient
from bson import ObjectId
import tempfile

app = Flask(__name__)
CORS(app)  # Allow all origins by default

# Initial setup
warnings.filterwarnings("ignore")

class SecureDocumentProcessor:
    def __init__(self):
        """Initialize with local models for secure processing"""
        self.setup_local_models()
        self.ollama_model = "llama3"  # Only for Q&A
        
    def setup_local_models(self):
        """Setup local models for summarization"""
        print("Loading local summarization models...")
        
        try:
            # Option 1: Use BART for summarization (recommended)
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            print(" BART summarization model loaded")
            
        except Exception as e:
            print(f" Failed to load BART: {e}")
            try:
                # Fallback: Smaller T5 model
                self.summarizer = pipeline(
                    "summarization",
                    model="t5-small",
                    device=0 if torch.cuda.is_available() else -1
                )
                print(" T5-small summarization model loaded (fallback)")
                
            except Exception as e2:
                print(f" Failed to load any summarization model: {e2}")
                self.summarizer = None

    def sent_tokenize(self, text):
        """Improved sentence tokenizer"""
        if not text.strip():
            return []
        
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        result = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                result.append(sentence)
        
        return result if result else [text.strip()]

    def smart_text_chunker(self, text, max_chunk_size=512, overlap=50):
        """Intelligently chunk text by sentences"""
        if not text.strip():
            return []

        sentences = self.sent_tokenize(text)
        if not sentences:
            return [text[:max_chunk_size]]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    words = current_chunk.split()
                    if len(words) > overlap:
                        overlap_text = " ".join(words[-overlap:])
                        current_chunk = overlap_text + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    words = sentence.split()
                    for i in range(0, len(words), max_chunk_size//10):
                        chunk_words = words[i:i + max_chunk_size//10]
                        chunks.append(" ".join(chunk_words))
                    current_chunk = ""
            else:
                current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def local_extractive_summary(self, text, num_sentences=5):
        """Create extractive summary using local processing only"""
        sentences = self.sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        
        # Simple scoring based on word frequency and position
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3 and word.isalpha():  # Filter meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            sentence_words = sentence.lower().split()
            score = 0
            word_count = 0
            
            for word in sentence_words:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            
            if word_count > 0:
                # Normalize by length and add position bonus for early sentences
                sentence_scores[i] = (score / word_count) + (1.0 / (i + 1)) * 0.1
        
        # Get top sentences
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_sentences.sort(key=lambda x: x[0])  # Sort by original order
        
        return " ".join([sentences[i] for i, _ in top_sentences])

    def secure_local_summarization(self, text, max_length=500, min_length=50):
        """Perform summarization entirely locally"""
        if not text.strip():
            return "No content to summarize."
        
        if len(text) < 100:
            return "Text too short to summarize meaningfully."
        
        # Method 1: Try transformer-based local summarization
        if self.summarizer:
            try:
                # Chunk text for local summarization
                chunks = self.smart_text_chunker(text, max_chunk_size=1024, overlap=100)
                summaries = []
                
                for chunk in chunks:
                    if len(chunk) > 50:
                        # Use local transformer model
                        result = self.summarizer(
                            chunk, 
                            max_length=min(max_length//len(chunks), 150),
                            min_length=min(min_length//len(chunks), 30),
                            do_sample=False
                        )
                        summaries.append(result[0]['summary_text'])
                
                if summaries:
                    combined = " ".join(summaries)
                    # If combined is too long, summarize again locally
                    if len(combined) > max_length * 1.5:
                        final_result = self.summarizer(
                            combined,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False
                        )
                        return final_result[0]['summary_text']
                    return combined
                    
            except Exception as e:
                print(f"Local transformer summarization failed: {e}")
        
        # Method 2: Fallback to extractive summarization (completely local)
        print("Using extractive summarization fallback")
        num_sentences = max(3, min(8, len(self.sent_tokenize(text)) // 4))
        return self.local_extractive_summary(text, num_sentences)

    def secure_qa_with_ollama(self, context, question):
        """Only send the specific question and minimal context to Ollama,
        with strict instructions to adhere to context."""
        try:
            # Find most relevant context chunk for the question
            question_words = set(question.lower().split())
            context_chunks = self.smart_text_chunker(context, max_chunk_size=2000, overlap=100)
            
            best_context = ""
            max_relevance = 0
            
            for chunk in context_chunks:
                chunk_words = set(chunk.lower().split())
                relevance = len(question_words.intersection(chunk_words))
                if relevance > max_relevance:
                    max_relevance = relevance
                    best_context = chunk
            
            if not best_context and context_chunks:
                best_context = context_chunks[0] # Fallback to first chunk if no relevance found
            
            # Limit context size to minimize data sent to Ollama
            if len(best_context) > 3000:
                best_context = best_context[:3000]
            
            # **Improved Prompt for Ollama**
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Your task is to answer the user's question STRICTLY based on the provided context. If the answer is not found within the context, state explicitly that the information is not available in the provided documents. Do NOT use any outside knowledge."},
                {"role": "user", "content": f"Context: {best_context}\n\nQuestion: {question}\n\nAnswer:"}
            ]
            
            response = ollama.chat(
                model=self.ollama_model,
                messages=messages,
                options={"temperature": 0.1} # Low temperature for factual answers
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            print(f"Ollama Q&A error: {e}")
            return f"Error generating answer: {str(e)}"

class MongoDBSummaryManager:
    def __init__(self, connection_string="", database_name=""):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client[database_name]
            self.user_collection = self.db.users
            self.collection = self.db.summaries
            self.qa_collection = self.db.qa_history  # New collection for Q&A history
            # Test connection
            self.client.admin.command('ping')
            print(f" Connected to MongoDB database: {database_name}")
        except Exception as e:
            print(f" Failed to connect to MongoDB: {e}")
            raise e
        
    def add_user(self, user):
        try:
            result = self.user_collection.insert_one(user)
            return str(result.inserted_id)
        except Exception as e:
            raise e
        
    def login_user(self, obj):
        try:
           print("obj: ", type(obj) )
           user = self.user_collection.find_one(
                {"email": obj["email"], "password": obj["password"]}
            )
           if user:
                return str(user.get('_id'))
           else:
                # No user found with these credentials
                return None
            # return str(result.inserted_id)
        except Exception as e:
            raise e
    
    def add_summary(self, filename, summary_text, text_length,userId):
        """Add a new summary to MongoDB"""
        try:
            document = {
                "userId":userId,
                "filename": filename,
                "summary_text": summary_text,
                "text_length": text_length,
                "timestamp": datetime.now(),
                "created_at": datetime.now().isoformat()
            }
            result = self.collection.insert_one(document)
            print(f" Summary stored in MongoDB with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            print(f" Error storing summary in MongoDB: {e}")
            raise e
    
    def get_all_summaries_text(self):
        """Get all summaries as combined text (equivalent to reading summaries.json)"""
        try:
            summaries = list(self.collection.find({}).sort("timestamp", 1))
            if not summaries:
                return ""
            
            combined_text = ""
            for summary in summaries:
                filename = summary.get('filename', 'Unknown')
                timestamp = summary.get('created_at', summary.get('timestamp', ''))
                summary_text = summary.get('summary_text', '')
                
                combined_text += f"--- Summary from {filename} ({timestamp}) ---\n"
                combined_text += summary_text + "\n\n"
            
            return combined_text
        except Exception as e:
            print(f" Error retrieving summaries from MongoDB: {e}")
            return ""
    
    def get_summaries_count_and_size(self):
        """Get count of summaries and approximate size"""
        try:
            count = self.collection.count_documents({})
            # Approximate size calculation
            total_text = self.get_all_summaries_text()
            size = len(total_text.encode('utf-8'))
            return count, size
        except Exception as e:
            print(f" Error getting summaries status: {e}")
            return 0, 0
    
    def clear_all_summaries(self):
        """Clear all summaries from MongoDB"""
        try:
            result = self.collection.delete_many({})
            print(f" Cleared {result.deleted_count} summaries from MongoDB")
            return result.deleted_count
        except Exception as e:
            print(f" Error clearing summaries from MongoDB: {e}")
            raise e
    
    def generate_summaries_file(self):
        """Generate a temporary summaries.json file for download"""
        try:
            combined_text = self.get_all_summaries_text()
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8')
            temp_file.write(combined_text)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            print(f" Error generating summaries file: {e}")
            raise e

    # New Q&A History Methods
    def add_qa_session(self, question, answer, userId, sessionId):
        """Add a new Q&A session or update existing one"""
        try:
            # Create a session ID based on the first question (truncated for display)
            session_name = question[:50] + "..." if len(question) > 50 else question
            
            # Check if this is the first Q&A or find existing session
            existing_session = self.qa_collection.find_one({
               "userId": userId,
               "sessionId": sessionId
            })
            
            qa_item = {
                "question": question,
                "answer": answer,
                "timestamp": datetime.now(),
                "created_at": datetime.now().isoformat()
            }
            
            def create_new_session():
                session_document = {
                    "userId" : userId,
                    "sessionId": sessionId,
                    "session_name": session_name,
                    "created_date": datetime.now().strftime("%Y-%m-%d"),
                    "created_at": datetime.now().isoformat(),
                    "qa_pairs": [qa_item]
                }
                result = self.qa_collection.insert_one(session_document)
                return str(result.inserted_id)
            
            if existing_session:
                # Add to existing session
                self.qa_collection.update_one(
                    {"_id": existing_session["_id"]},
                    {"$push": {"qa_pairs": qa_item}}
                )
                return str(existing_session["_id"])
            else:
                # Create new session
               return create_new_session()
            
        except Exception as e:
            print(f" Error storing Q&A in MongoDB: {e}")
            raise e
    
    def get_qa_sessions(self):
        """Get all Q&A sessions"""
        try:
            sessions = list(self.qa_collection.find().sort("created_at", -1))
            for session in sessions:
                session["_id"] = str(session["_id"])
            return sessions
        except Exception as e:
            print(f" Error retrieving Q&A sessions: {e}")
            return []
    
    def get_qa_session_by_id(self, session_id):
        """Get a specific Q&A session by ID"""
        try:
            session = self.qa_collection.find_one({"_id": ObjectId(session_id)})
            if session:
                session["_id"] = str(session["_id"])
            return session
        except Exception as e:
            print(f" Error retrieving Q&A session: {e}")
            return None
    
    def clear_all_qa_history(self):
        """Clear all Q&A history"""
        try:
            result = self.qa_collection.delete_many({})
            print(f" Cleared {result.deleted_count} Q&A sessions from MongoDB")
            return result.deleted_count
        except Exception as e:
            print(f" Error clearing Q&A history: {e}")
            raise e

# Initialize the secure processor and MongoDB manager
processor = SecureDocumentProcessor()
mongo_manager = MongoDBSummaryManager()

# Keep the summaries folder for backward compatibility (can be removed if not needed)
SUMMARIES_FOLDER = os.path.join(os.getcwd(), 'summaries')
os.makedirs(SUMMARIES_FOLDER, exist_ok=True)

# Document extraction functions (unchanged)
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = re.sub(r'\s+', ' ', page_text)
        text += page_text + " "
    return text.strip()

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text.strip() + " "
    return text.strip()

def extract_text_from_txt(txt_file):
    content = txt_file.read().decode('utf-8')
    content = re.sub(r'\s+', ' ', content)
    return content.strip()

# Route to serve the HTML frontend
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register_user():
    try:
        user_data = request.json
        user_id = mongo_manager.add_user(user_data)
        userId = request.headers.get("userId")
        return jsonify({"data": user_id, "userId": userId})
    except Exception as e:
        print(f"Login failed : {e}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500
    
@app.route("/login", methods=["POST"])
def login_user():
    try:
        user_data = request.json
        user_id = mongo_manager.login_user(user_data)
        return jsonify({"data": user_id})
    except Exception as e:
        print(f"Login failed : {e}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route("/process", methods=["POST"])
def process_document():
    """Secure document processing - all extraction and summarization local.
       Now stores summaries in MongoDB instead of local file."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        file_extension = os.path.splitext(filename)[1].lower()
        file_content = file.read()
        mime_type, _ = mimetypes.guess_type(filename)

        print(f" Secure processing: {filename} (local only)")

        # Extract text locally
        text_content = ""
        if mime_type == 'application/pdf' or file_extension == '.pdf':
            text_content = extract_text_from_pdf(io.BytesIO(file_content))
        elif mime_type == 'application/docx' or file_extension == '.docx':
            text_content = extract_text_from_docx(io.BytesIO(file_content))
        elif mime_type and mime_type.startswith('text/') or file_extension == '.json' or file_extension == '.txt':
            text_content = file_content.decode('utf-8')
        else:
            return jsonify({"error": f"Unsupported file type: {file_extension}"}), 400

        if len(text_content.strip()) < 10:
            return jsonify({"error": "Extracted text is too short or empty"}), 400

        print(f" Text extracted locally: {len(text_content)} characters")

        #  SECURE: Summarize entirely locally (no data sent to Ollama)
        summary_text = processor.secure_local_summarization(text_content)

        if not summary_text:
            return jsonify({"error": "Could not generate summary locally"}), 400

        # Store summary in MongoDB instead of local file
        userId=request.headers.get("userId")
        summary_id = mongo_manager.add_summary(filename, summary_text, len(text_content),userId)

        print(f" Secure processing complete for {filename}. Summary stored in MongoDB with ID: {summary_id}")
        print(f" No data sent to external services during processing")

        return jsonify({
            "message": "Document processed securely (local summarization)",
            "summary_text": summary_text,
            "text_length": len(text_content),
            "security_mode": "local_processing",
            "summary_id": summary_id,
            "combined_summaries_url": f"/download-combined-summaries"
        }), 200

    except Exception as e:
        print(f" Secure processing error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Processing error: {str(e)}"}), 500

@app.route('/generate-answer', methods=['POST'])
def generate_answer():
    """Secure Q&A - uses content from MongoDB summaries as context.
       Now also saves Q&A to history."""
    try:
        user_question = request.json.get('input')
        sessionId = request.json.get('sessionId')
        if not user_question:
            return jsonify({"error": "No question provided"}), 400

        userId = request.headers.get("userId")
        # Get all summaries from MongoDB as context
        context_text = mongo_manager.get_all_summaries_text()
        
        # --- START OF MODIFICATION ---
        if not context_text.strip():
            print(f"DEBUG: No summaries found in MongoDB for Q&A context. User asked: '{user_question}'")
            return jsonify({"error": "No summaries available for Q&A. Please process documents or ensure existing data in MongoDB."}), 400
        # --- END OF MODIFICATION ---

        print(f"DEBUG: Successfully retrieved {len(context_text)} characters from MongoDB summaries")
        print(f" Secure Q&A: Using MongoDB summaries ({len(context_text)} chars) as context for Ollama")
        print(f" Question: {user_question}")

        # Use secure Q&A method
        answer = processor.secure_qa_with_ollama(context_text, user_question)

        # Save Q&A to history
        
        session_id = mongo_manager.add_qa_session(user_question, answer,userId, sessionId)

        return jsonify({
            
            "question": user_question,
            "answer": answer,
            "session_id": session_id,
            "security_note": "Only question and minimal relevant chunk from MongoDB summaries sent to Ollama"
        }), 200

    except Exception as e:
        print(f" Q&A error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# New Q&A History Routes
@app.route('/qa-sessions', methods=['GET'])
def get_qa_sessions():
    """Get all Q&A sessions for history display"""
    try:
        userId = request.headers.get("userId")
        sessions = mongo_manager.get_qa_sessions()
        return jsonify(sessions), 200
    except Exception as e:
        print(f" Error getting Q&A sessions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/qa-session/<session_id>', methods=['GET'])
def get_qa_session(session_id):
    """Get a specific Q&A session by ID"""
    userId = request.headers.get("userId")
    try:
        session = mongo_manager.qa_collection.find_one({"_id": ObjectId(session_id)}) # Corrected to use qa_collection
        if session:
            session["_id"] = str(session["_id"])
        return jsonify(session), 200
    except Exception as e:
        print(f" Error getting Q&A session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/clear-qa-history', methods=['DELETE'])
def clear_qa_history():
    """Clear all Q&A history"""
    userId = request.headers.get("userId")
    try:
        deleted_count = mongo_manager.clear_all_qa_history()
        return jsonify({"message": f"Q&A history cleared successfully. Deleted {deleted_count} sessions."}), 200
    except Exception as e:
        print(f" Error clearing Q&A history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/summaries", methods=["GET"])
def list_summaries():
    """Lists available summary files (now from MongoDB)."""
    userId = request.headers.get("userId")
    try:
        count, size = mongo_manager.get_summaries_count_and_size()
        if count > 0:
            return jsonify([{"count": count, "total_size_bytes": size, "source": "mongodb"}]), 200
        else:
            return jsonify([]), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download-combined-summaries', methods=['GET'])
def download_combined_summaries():
    """Downloads the combined summaries from MongoDB as a file."""
    try:
        temp_file_path = mongo_manager.generate_summaries_file()
        return send_file(temp_file_path, as_attachment=True, download_name='summaries.json', mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/clear-all-summaries', methods=['DELETE'])
def clear_all_summaries():
    """Clears all summaries from MongoDB."""
    try:
        deleted_count = mongo_manager.clear_all_summaries()
        print(f" Cleared {deleted_count} summaries from MongoDB")
        return jsonify({"message": f"All summaries cleared from MongoDB successfully. Deleted {deleted_count} documents."}), 200
    except Exception as e:
        print(f" Error clearing summaries: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/check-summaries-status', methods=['GET'])
def check_summaries_status():
    """Checks MongoDB summaries status."""
    try:
        count, size = mongo_manager.get_summaries_count_and_size()
        exists = count > 0
        return jsonify({
            "exists": exists, 
            "size": size,
            "count": count,
            "storage": "mongodb"
        }), 200
    except Exception as e:
        print(f" Error checking summaries status: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" SECURE LOCAL DOCUMENT PROCESSOR (MongoDB Integration)")
    print("="*60)
    print(" Document extraction: 100% Local")
    print(" Summarization: 100% Local (Transformers/Extractive)")
    print(" Storage: MongoDB (summaries database)")
    print(" Q&A: Uses content from MongoDB as context for Ollama (local model)")
    print(" Q&A History: Saved to MongoDB with session management")
    print(" Maximum privacy and security (assuming local Ollama + MongoDB setup)")
    print("="*60)
    # Ensure the summaries folder exists for backward compatibility
    os.makedirs(SUMMARIES_FOLDER, exist_ok=True)
    app.run(host="0.0.0.0", port=5005, debug=False)