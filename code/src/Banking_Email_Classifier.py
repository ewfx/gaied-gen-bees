from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from parseioc import extract_banking_context_with_parsio
from transformers import pipeline
import torch
from PyPDF2 import PdfReader
import docx
import email
from email import policy
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

classifier = pipeline("text-classification", model="nickprock/distilbert-base-uncased-banking77-classification")
ner_extractor = pipeline("ner", model="dslim/bert-base-NER")

# Extract text from various file formats
def extract_text_from_file(file: UploadFile) -> str:
    try:
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file.file)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif file.filename.endswith(".docx"):
            doc = docx.Document(file.file)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file.filename.endswith(".doc"):
            raise HTTPException(status_code=400, detail=".doc files are not supported. Please convert to .docx")
        elif file.filename.endswith(".eml"):
            msg = email.message_from_bytes(file.file.read(), policy=policy.default)
            return msg.get_body(preferencelist=('plain')).get_content() if msg.get_body() else ""
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        logging.error(f"Error extracting text from file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {file.filename}")

# Extract banking-related context using an enhanced hybrid approach
def extract_banking_context(text: str) -> List[str]:
    try:
        entities = ner_extractor(text)
        ner_contexts = [entity['word'] for entity in entities if entity['entity'].startswith('B-')]

        patterns = [
            r"(?:account|acct|a/c|iban|sort code|routing number)[\s#:]*\d{4,}",  # Account identifiers
            r"(?:loan|mortgage|policy|reference)[\s#:]*\d{4,}",     # Loan and policy numbers
            r"(?:credit card|debit card|cc|payment card|card number)[\s#:]*\d{4,}",  # Card details
            r"(?:transaction|txn|payment|amount|transfer|deposit)[\s#:]*\$?\d{1,},?\d*(?:\.\d{2})?",  # Transactions
            r"(?:OTP|PIN|password|security code)[\s#:]*\d{4,}",  # Sensitive info
            r"(?:refund|chargeback|overdraft|fee|interest|withdrawal|balance|statement|payment confirmation)"  # Banking actions
        ]

        # Extract matches without altering regex structure
        regex_contexts = [match.group() for pattern in patterns for match in re.finditer(pattern, text, re.IGNORECASE)]

        # Capture context around identified terms (2 sentences around matches)
        surrounding_contexts = []
        for match in re.finditer(r"([^.]*?(?:" + "|".join([p.replace("(", "(?:") for p in patterns]) + ")[^.]*\.)", text, re.IGNORECASE):
            surrounding_contexts.append(match.group().strip())

        banking_contexts = list(set(ner_contexts + regex_contexts + surrounding_contexts))

        if not banking_contexts:
            logging.warning("No banking context found, using default context.")
            banking_contexts.append("general banking inquiry")

        return banking_contexts
    except Exception as e:
        logging.error(f"Error during banking context extraction: {e}")
        raise HTTPException(status_code=500, detail="Error extracting banking context")

# Classify the extracted banking contexts with enhanced accuracy
def classify_email_contexts(contexts: List[str]):
    try:
        classifications = []
        for context in contexts:
            combined_input = " ".join(contexts)[:512]  # Combine contexts for richer input
            results = classifier(combined_input, top_k=3)

            for result in results:
                label_parts = result['label'].split(" - ")
                category = label_parts[0] if len(label_parts) > 0 else "Unknown"
                subcategory = label_parts[1] if len(label_parts) > 1 else "Unknown"
                confidence = round(result['score'] * 100, 2)  # Convert to percentage

                if confidence >= 60:  # Filter by confidence
                    classifications.append({
                        "requestType": category,
                        "subRequestType": subcategory,
                        "confidence": confidence,
                        "context": combined_input
                    })

        if not classifications:
            classifications.append({
                "requestType": "Unknown",
                "subRequestType": "Unknown",
                "confidence": 0.0,
                "context": "No relevant context found"
            })

        return classifications
    except Exception as e:
        logging.error(f"Error during email classification: {e}")
        raise HTTPException(status_code=500, detail="Error during classification")

@app.post("/classify-emails-mc/")
async def classify_emails(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            logging.info(f"Processing file: {file.filename}")
            email_text = extract_text_from_file(file)

            # Extract and classify contexts
            banking_contexts = extract_banking_context_with_parsio(email_text)
            classification = classify_email_contexts(email_text)

            results.append({"filename": file.filename, "classification": classification})
            logging.info(f"Successfully processed file: {file.filename}")
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {e}")
            results.append({"filename": file.filename, "error": str(e)})
    return results
