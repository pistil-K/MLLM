import torch
from datasets import load_from_disk, load_dataset
from transformers import pipeline, AutoProcessor, Qwen2VLForConditionalGeneration,AutoModelForImageTextToText
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from PIL import Image
import json
import os
from tqdm import tqdm

# Define evaluation functions
def evaluate_bleu(reference, generated):
    reference_tokens = reference.split()
    generated_tokens = generated.split()
    score = sentence_bleu([reference_tokens], generated_tokens)
    return score

def evaluate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

def evaluate_answer(reference, generated):
    bleu_score = evaluate_bleu(reference, generated)
    rouge_scores = evaluate_rouge(reference, generated)
    return bleu_score, rouge_scores

# Load dataset
dataset_path = "data1"
data = load_dataset(dataset_path)["train"]

# Create output directory
output_dir = 'RAG_Qwen'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load Qwen2VL model and processor
model_name = "qwen2.5"  # Replace with the correct model name or local path
processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Set up document embeddings using HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="./MiniLM")

# Create documents for FAISS indexing
documents = []
for entry in data:
    content = entry.get('text', '')
    document = Document(page_content=content)
    documents.append(document)

faiss_db = FAISS.from_documents(documents, embeddings)

# Setup Qwen pipeline
qwen_pipeline = pipeline(
    "text-generation",
    model=model,
    processor=processor,
    max_new_tokens=128,
    tokenizer=processor.tokenizer,  # Explicitly pass the tokenizer
)

qwen_llm = HuggingFacePipeline(pipeline=qwen_pipeline)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=qwen_llm,
    chain_type="stuff",
    retriever=faiss_db.as_retriever(),
)

generated_results_qwen = []

for idx in tqdm(range(len(data))):
    sample = data[idx]

    image = sample["image"]
    question = sample["question"]

    if image is None or (isinstance(image, Image.Image) and image.size == (0, 0)):
        print(f"Warning: Invalid image at index {idx}, skipping...")
        continue

    # Prepare inputs for Qwen2VL (text + image)
    inputs = processor(text=question, images=[image], return_tensors="pt").to(model.device)

    # Use RetrievalQA for Qwen inference
    try:
        qwen_output_text = qa_chain.invoke({"query": question})["result"]
    except Exception as e:
        print(f"Error during inference at index {idx}: {e}")
        continue

    # Compute evaluation metrics
    qwen_bleu_score, qwen_rouge_scores = evaluate_answer(sample["chosen"], qwen_output_text)

    # Store Qwen results
    result_qwen = {
        "idx": sample["idx"],
        "image_path": sample["image_path"],
        "question": sample["question"],
        "qwen_output": qwen_output_text,
        "chosen": sample["chosen"],
        "rejected": sample["rejected"],
        "qwen_bleu_score": qwen_bleu_score,
        "qwen_rouge1": qwen_rouge_scores['rouge1'].fmeasure,
        "qwen_rouge2": qwen_rouge_scores['rouge2'].fmeasure,
        "qwen_rougeL": qwen_rouge_scores['rougeL'].fmeasure,
    }
    generated_results_qwen.append(result_qwen)

    if idx % 100 == 0:
        with open(f"{output_dir}/generated_results_qwen_{idx}.json", 'w') as f:
            json.dump(generated_results_qwen, f, indent=4)

    del qwen_output_text
    torch.cuda.empty_cache()

# Save final results
with open(f"{output_dir}/generated_results_qwen_final.json", 'w') as f:
    json.dump(generated_results_qwen, f, indent=4)

print(f"Results saved to {output_dir}/generated_results_qwen_final.json")

# Print some results
for result_qwen in generated_results_qwen:
    print(f"Sample ID: {result_qwen['idx']}")
    print(f"Qwen BLEU Score: {result_qwen['qwen_bleu_score']}")
    print(f"Qwen ROUGE1 Score: {result_qwen['qwen_rouge1']}")
    print("-----")