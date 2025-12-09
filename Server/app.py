from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import asyncio
import sys
import os
from main import scan_url  # Your existing scanner

app = Flask(__name__, static_folder='../website', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow frontend requests

@app.route('/')
def serve_index():
    return send_from_directory('../website', 'index.html')

@app.route('/scan.html')
def serve_scan():
    return send_from_directory('../website', 'scan.html')

@app.route('/scan', methods=['POST'])
def api_scan():
    data = request.json
    url = data.get('url', '')
    if not url:
        return jsonify({'error': 'URL required'}), 400
    
    try:
        result = asyncio.run(scan_url(url))
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../website', path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
