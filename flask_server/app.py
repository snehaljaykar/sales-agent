import os
import csv
from flask import Flask, jsonify, request
from datetime import datetime
import logging

from EventManager import EventHandler, EventSimulator
from DataLoader import DataLoader
from rag_engine import RAGEngine
import llm_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

DATA_DIR = os.getenv("DATA_DIR", "./data")  # folder where CSVs are stored
USERS_FILE = os.path.join(DATA_DIR, "users.csv")
FILES_FILE = os.path.join(DATA_DIR, "files.csv")
PARTICIPANTS_FILE = os.path.join(DATA_DIR, "file_participants.csv")

# Global instances
loader = None
rag_engine = None

def read_csv(file_path):
    with open(file_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/users/<user_id>", methods=["GET"])
def process_user_files(user_id: str):
    """Original endpoint for processing user files"""
    loader.process_user_files(user_id)
    return jsonify({"status": "completed"})

@app.route("/chat", methods=["POST"])
def chat_query():
    """Handle chat queries from the client"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        user_id = data.get("user_id")
        query_text = data.get("query", "").strip()
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        if not query_text:
            return jsonify({"error": "query is required"}), 400

        user_name = loader.users_df[loader.users_df['user_id'] == int(user_id)].iloc[0]['user_name']
        logger.info(f"Processing chat query for user {user_id} {user_name}: {query_text}")

        print(type(llm_utils.query_converter))
        
        # Ensure LLM services are initialized
        if llm_utils.query_converter is None or llm_utils.answer_generator is None:
            logger.warning("LLM services not initialized, initializing now...")
            llm_utils.initialize_llm_services()
        
        # # Step 1: Convert natural language query to search parameters
        # search_params = llm_utils.query_converter.convert_to_search_query(query_text, user_name)
        # logger.info(f"Search parameters: {search_params}")
        
        # Step 2: Search Qdrant for relevant documents
        retrieved_docs = rag_engine.search_transcripts(query_text, user_name, limit=10)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 3: Generate answer using LLM + retrieved docs
        answer = llm_utils.answer_generator.generate_answer(query_text, retrieved_docs, query_text)
        
        return jsonify({
            "answer": answer,
            "sources": len(retrieved_docs),
            "search_params": query_text,
            "documents_found": len(retrieved_docs)
        })
        
    except Exception as e:
        logger.error(f"Error in chat_query: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    logger.info("Starting Sales Agent Chatbot Server...")

    # EventHandler to read JSON file and create users, files and file_participants tables (csv files)
    handler = EventHandler()
    simulator = EventSimulator('data/event_queue.json', handler)
    simulator.run()
    handler.write_csvs()

    logger.info("CSV files generated: files.csv, users.csv, file_participants.csv")

    # Initialize DataLoader
    loader = DataLoader(data_dir="./data", qdrant_host="qdrant", qdrant_port=6333)

    # Initialize RAG Engine
    rag_engine = RAGEngine(loader)
    logger.info("RAG Engine initialized")

    # Initialize LLM services
    llm_utils.initialize_llm_services()
    logger.info("LLM services initialized")

    # Cron job / scheduled run: process all pending files
    loader.process_all_pending()

    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info("Server ready! Starting Flask app on port 5000...")
    app.run(host="0.0.0.0", port=5000)
