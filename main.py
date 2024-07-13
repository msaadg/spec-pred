import fitz  # PyMuPDF
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

nltk.download('stopwords')
nltk.download('wordnet')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_blocks = []
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text_blocks.append(block[4])
    return text_blocks

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def embed_sentence(sentence, max_length=512):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def calculate_similarity(text1, text2, max_length=512):
    # Split text1 into chunks if it's longer than max_length
    text1_chunks = [text1[i:i+max_length] for i in range(0, len(text1), max_length)]
    text2_chunks = [text2[i:i+max_length] for i in range(0, len(text2), max_length)]
    
    similarities = []
    for chunk1 in text1_chunks:
        for chunk2 in text2_chunks:
            vec1 = embed_sentence(chunk1, max_length)
            vec2 = embed_sentence(chunk2, max_length)
            similarity = cosine_similarity(vec1.detach().numpy(), vec2.detach().numpy())[0][0]
            similarities.append(similarity)
    
    # Return the average similarity score
    return sum(similarities) / len(similarities) if similarities else 0.0


def load_lookup_table(file_path):
    with open(file_path, 'r') as file:
        lookup_sentences = [line.strip() for line in file.readlines()]
    return lookup_sentences

def main():
    script_dir = os.path.dirname(__file__)
    pdf_file_path = os.path.join(script_dir, 'ML_Book.pdf')
    lookup_file_path = os.path.join(script_dir, 'lookup.txt')

    lookup_sentences = load_lookup_table(lookup_file_path)

    text_blocks = extract_text_from_pdf(pdf_file_path)
    preprocessed_blocks = [preprocess_text(block) for block in text_blocks]

    results = []
    for block in preprocessed_blocks:
        for lookup_sentence in lookup_sentences:
            similarity = calculate_similarity(block, lookup_sentence)
            if similarity > 0.7:
                results.append({
                    'block': block,
                    'lookup_sentence': lookup_sentence,
                    'similarity': similarity
                })

    for result in results:
        print(f"Block: {result['block']}")
        print(f"Lookup Sentence: {result['lookup_sentence']}")
        print(f"Similarity: {result['similarity']:.2f}\n")

if __name__ == '__main__':
    main()
