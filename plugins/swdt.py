import spacy
from graphviz import Digraph
import PyPDF2
from docx import Document

nlp = spacy.load("en_core_web_sm")

def parse_docx(file_path):
    doc = Document(file_path)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return '\n'.join(text)

def parse_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = []
        for page in reader.pages:
            text.append(page.extract_text())
        return '\n'.join(text)

def generate_mindmap(key_sentences, entities):
    dot = Digraph(comment="Mindmap")
    dot.node("Root", "Document Summary")

    for idx, sentence in enumerate(key_sentences):
        dot.node(f"sentence_{idx}", sentence)
        dot.edge("Root", f"sentence_{idx}")

    for idx, (ent, label) in enumerate(entities):
        dot.node(f"ent_{idx}", f"{ent} ({label})")
        dot.edge("Root", f"ent_{idx}")

    mindmap_path = "mindmap.pdf"
    dot.render(mindmap_path, view=False, format="pdf")
    return mindmap_path

def analyze_text(text):
    doc = nlp(text)
    key_sentences = [sent.text for sent in doc.sents if len(sent.text) > 50]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return key_sentences, entities