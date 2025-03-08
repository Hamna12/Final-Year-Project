import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from langchain.memory import SimpleMemory
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face token from environment variables
hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
if not hf_token:
    raise ValueError("Hugging Face token not set. Please set it in the environment variable 'HUGGINGFACE_HUB_TOKEN'.")

# Initialize the LaMini model pipeline
model_name = "MBZUAI/LaMini-T5-61M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print('Initializing LaMini pipeline')
# Set up the text generation pipeline using LaMini model
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available, else CPU
    max_length=500
)
print("LaMini model and pipeline initialized successfully.")

# Define prompt template
prompt_template = """
Act like you are an expert intelligent assistant and Data scientist focused on explaining socio-economic and educational data through visual representations and prediction insights.

{context}

Based on the data provided in the visualizations, please address the following:

1. Explain the trends in pass/fail ratios across different years.
2. Compare pass/fail ratios in tehsils, districts, urban, and rural areas.
3. Discuss differences in performance between government and non-government schools.
4. Analyze student performance in various subjects, and how it varies by gender and location.
5. Highlight any significant patterns or insights visible in the visual data.
6. What prediction results are representing?
7. How we can investigate output of the model?
8. What are the limitation of the model?

Question: {question}

Answer:
"""

# Define the context for the visuals (should be replaced with actual context for better results)
visual_context = """
1. The line chart on the dashboard shows trends in pass/fail ratios across different years.
2. The bar chart compares pass/fail ratios in tehsils, districts, urban, and rural areas.
3. The comparative bar charts illustrate the performance differences between government and non-government schools.
4. The pie charts highlight student performance in various subjects and differences by gender.
5. The map visualization depicts living standards across different districts.
6. Explain the insights that SHAP graph is displaying.
"""

# Custom function to generate response using the pipeline
def generate_response_with_pipe(context: str, question: str) -> str:
    input_text = f"{context}\n\nQuestion: {question}\n\nAnswer:"
    response = pipe(input_text, max_length=500)
    return response[0]['generated_text']

# Initialize prompt template
template = PromptTemplate.from_template(prompt_template)
memory = SimpleMemory()

# Load the dataset and process it
file_path = "Data/qa-chatbot.txt"
qa_pairs = []

with open(file_path, "r") as file:
    lines = file.readlines()

question, answer = "", ""
for line in lines:
    line = line.strip()
    if line.startswith("Q") and ":" in line:
        if question and answer:
            qa_pairs.append({"question": question, "answer": answer})
        question = line.split(": ", 1)[1] if ": " in line else line.split(":", 1)[1].strip()
        answer = ""
    elif line.startswith("A") and ":" in line:
        answer = line.split(": ", 1)[1] if ": " in line else line.split(":", 1)[1].strip()
    else:
        if line:
            answer += " " + line

if question and answer:
    qa_pairs.append({"question": question, "answer": answer})

df = pd.DataFrame(qa_pairs)
print(f"Loaded dataset with {len(df)} entries.")
print(df.head())  # Print the first few rows
chunks = [Document(page_content=qa) for qa in (df['question'] + " " + df['answer'])]

# Define a class to handle embedding
class SentenceTransformerWrapper:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return [embedding.tolist() for embedding in embeddings]  # Convert to list of lists
    
    def embed_query(self, query):
        embedding = self.model.encode([query], show_progress_bar=False)
        return embedding[0].tolist()  # Convert to list

# Set up Sentence Transformer for embeddings
embedding_wrapper = SentenceTransformerWrapper('sentence-transformers/all-MiniLM-L6-v2')

# Setup persistent storage directory
persist_directory = 'db'
client = chromadb.PersistentClient(path=persist_directory)

# Initialize Chroma and add documents
vectordb = Chroma(
    collection_name="qa_collection",
    embedding_function=embedding_wrapper,
    persist_directory=persist_directory,
    client=client
)

vectordb.add_texts([chunk.page_content for chunk in chunks])

# Function to generate response using the prompt template
def generate_response(question):
    # Retrieve context from Chroma
    context_docs = vectordb.similarity_search(question, k=3)
    context = " ".join([doc.page_content for doc in context_docs])
    
    inputs = {
        "context": context,
        "question": question
    }
    response = generate_response_with_pipe(context, question)
    return response

# Example question to test the chatbot
# if __name__ == "__main__":
#     test_question = "What is the prediction result it is showing?"
#     response = generate_response(test_question)
#     print(response)


