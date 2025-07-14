# RAG-based QA System for University Regulations

This project implements a Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about university academic documents.

## Overview

- Academic regulation documents in PDF format are processed via OCR (Optical Character Recognition).
- Extracted texts are segmented into overlapping chunks and vectorized using SentenceTransformer embeddings.
- A FAISS index enables semantic retrieval of relevant chunks based on user queries.
- Retrieved context is passed to a Turkish GPT-2 model (`ytu-ce-cosmos/turkish-gpt2`) to generate answers in natural language.

## Technologies Used

- Python 3.11
- pytesseract, pdf2image (OCR)
- sentence-transformers, faiss-cpu (Retrieval)
- transformers, torch (Text generation)

## Current Status

- OCR-to-FAISS pipeline is functional.
- Question answering with generation is under development; current integration with GPT-2 requires optimization due to memory constraints on local execution.
