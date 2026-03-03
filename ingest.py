import os
import glob
import torch
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_advanced_vector_store():
    print("🚀 Starting progress-tracked hardware ingestion pipeline...")
    
    # 1. FIND DATA
    pdf_files = glob.glob("./test_data/*.pdf")
    if not pdf_files:
        print("❌ No documents found in data/ folder!")
        return

    documents = []

    # 2. LOAD & PARSE
    for file_path in pdf_files:
        print(f"📖 Reading: {file_path}")
        try:
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            
            filename = os.path.basename(file_path).upper()
            mcu_tag = "GENERIC"
            
            if "ESP32" in filename: mcu_tag = "ESP32"
            elif "STM32" in filename: mcu_tag = "STM32"
            elif "ATMEGA" in filename or "328" in filename: mcu_tag = "ATmega328P"
            elif "RP2040" in filename or "PICO" in filename: mcu_tag = "RP2040"
            
            for doc in docs:
                doc.metadata["mcu"] = mcu_tag
                doc.metadata["source"] = os.path.basename(file_path)
                
            documents.extend(docs)
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")

    # 3. CHUNKING
    print(f"📑 Total pages loaded: {len(documents)}. Splitting into technical chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=250,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    all_chunks = text_splitter.split_documents(documents)
    total_chunks = len(all_chunks)
    print(f"✅ Generated {total_chunks} chunks.")

    # 4. INITIALIZE EMBEDDINGS (CPU)
    print("🧠 Initializing BGE-Large Embedding Model (CPU Mode)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5", 
        model_kwargs={'device': 'cpu'},     
        encode_kwargs={'normalize_embeddings': True} 
    )

    # 5. BATCHED VECTOR STORE CREATION (With Logging)
    print(f"⚡ Processing {total_chunks} chunks in batches of 50 to track progress...")
    
    # Initialize the FAISS index with the first chunk
    start_time = time.time()
    vectorstore = FAISS.from_documents([all_chunks[0]], embeddings)
    
    batch_size = 50
    # Process the rest in batches
    for i in range(1, total_chunks, batch_size):
        batch = all_chunks[i : i + batch_size]
        vectorstore.add_documents(batch)
        
        # Calculate progress and speed
        elapsed = time.time() - start_time
        chunks_done = i + len(batch)
        percent = (chunks_done / total_chunks) * 100
        print(f"⏳ Progress: {chunks_done}/{total_chunks} chunks ({percent:.1f}%) | Time elapsed: {elapsed:.1f}s", end="\r")

    # 6. SAVE
    print(f"\n\n💾 Saving Vector Database to './vectorstore'...")
    vectorstore.save_local("./vectorstore")
    print(f"✨ SUCCESS! {total_chunks} chunks are now indexed and ready for app.py.")

if __name__ == "__main__":
    build_advanced_vector_store()