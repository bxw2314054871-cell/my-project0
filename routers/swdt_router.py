from flask import Flask, request, jsonify, send_file
import os
from src.plugins.swdt import analyze_text, generate_mindmap, parse_pdf, parse_docx
import spacy
from graphviz import Digraph

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
nlp = spacy.load("en_core_web_sm")

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return jsonify({"message": "文件上传成功", "path": file_path})
    else:
        return jsonify({"error": "未上传文件"}), 400

@app.route('/analyze', methods=['POST'])
def analyze_document():
    data = request.json
    file_path = "D:/test2.pdf"#data.get('path')
    if not file_path:
        return jsonify({"error": "未提供文件路径"}), 400

    try:
        text = parse_document(file_path)
        key_sentences, entities = analyze_text(text)
        mindmap_path = generate_mindmap(key_sentences, entities)
        return jsonify({"message": "分析完成，思维导图已生成", "mindmap_path": mindmap_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download', methods=['GET'])
def download_mindmap():
    mindmap_path = request.args.get('path')
    if not mindmap_path:
        return jsonify({"error": "未提供思维导图路径"}), 400

    if not os.path.exists(mindmap_path):
        return jsonify({"error": "文件不存在"}), 404

    return send_file(mindmap_path, as_attachment=True)

def parse_document(file_path):
    if file_path.endswith('.pdf'):
        return parse_pdf(file_path)
    elif file_path.endswith('.docx'):
        return parse_docx(file_path)
    else:
        raise ValueError("不支持的文件格式")

if __name__ == '__main__':
    app.run(debug=True, port=5001)