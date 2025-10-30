from flask import Flask, request, jsonify
from flask_cors import CORS
from main import rag_simple, rag_retriever, client  # Import from main.py

app = Flask(__name__)
CORS(app)

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query", "")
    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    try:
        answer = rag_simple(query_text, rag_retriever, client)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
