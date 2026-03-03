# 🛠️ Smart Microcontroller RAG Assistant

An advanced, locally-hosted Retrieval-Augmented Generation (RAG) system engineered to navigate massive technical reference manuals for the STM32F10, ESP32-C6, ATmega328P, and RP2040 architectures.

Developed as an extension of the NVIDIA DLI: Building RAG Agents framework, this tool is designed for offline execution to ensure data privacy and zero API costs. It acts as a strict hardware auditor, transforming 1000+ page datasheets into an interactive, hallucination-free knowledge base.

------------------------------------------------------------------------

## ✨ Key Technical Features
- **Strict Auditor Guardrails:** Operates under a "Zero Hallucination" directive. If a user attempts to connect incompatible logic levels (e.g., a 5V sensor to a 3.3V-only pin), the system searches for "5V tolerant" flags and explicitly warns of permanent hardware damage if not found.

- **Dynamic Metadata Filtering:** Prevents cross-architectural contamination. The system detects the target MCU in the prompt and strictly locks the retriever to that specific manual, ensuring RP2040 limits are never accidentally applied to an ESP32-C6 design.

- **Hardware Multi-Query Retriever:** Intercepts user queries and automatically generates technical variations (hunting for specific table headers like "Absolute Maximum Ratings" or "Electrical Characteristics") to guarantee the retrieval of dense data tables.

- **Mandatory Citations:** Every technical claim, register name, or voltage threshold outputs with a strict citation format (e.g., [Source: ESP32-C6.pdf, Page: 1042]).

## 🚀 Quick Start: Installation

### 1. Prerequisites

-   Python 3.10+
-   Git
-   NVIDIA GPU (RTX 30, 40, or 50 series recommended)
-   Ollama (Download from ollama.com)

### 2. Clone & Setup Environment

``` bash
# Clone the repository
git clone https://github.com/your-username/mcu-rag-assistant.git
cd mcu-rag-assistant
```

### 3. Install Core Dependencies

```bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 🧠 Technical Architecture

### 🛡️ Hardware-Aware Guardrails

To effectively run a 2000+ page RAG system locally on an 8GB VRAM limit, this architecture utilizes a hybrid processing split between the CPU and GPU.

Vector Store (CPU): 

Chunking Strategy: RecursiveCharacterTextSplitter configured with large 1200-character chunks to prevent massive datasheet tables from fracturing during ingestion.

### 🧬 The RAG Pipeline

-   **Orchestration:** LangChain (LCEL)\
-   **LLM:** Llama 3.1 8B (4-bit quantized via Ollama). Fits within 5-6GB of VRAM, reserving the rest of the GPU for context processing.\
-   **Embeddings:** BAAI/bge-large-en-v1.5. Offloaded to the CPU to preserve VRAM. Exceptional at capturing the semantic meaning of dense engineering jargon.\
-   **Vector Store:** FFAISS (Facebook AI Similarity Search). Operates entirely in-memory for ruthlessly fast local CPU execution.
-   **Chunking Strategy:** RecursiveCharacterTextSplitter configured with large 1200-character chunks to prevent massive datasheet tables from fracturing during ingestion.\

------------------------------------------------------------------------

## 📊 Performance Benchmarks

**Tested on NVIDIA RTX 5060 (8GB VRAM)**

  Metric             Result
  ------------------ ------------------------------------------
  VRAM Footprint     \~6.2 GB (Stable)
  Response Latency   6s (min) - 15s (max)
  Memory Limit       Truncated at 12 messages for VRAM safety

------------------------------------------------------------------------

## ⚖️ Evaluation (Notebook 8 Standards)

The project abandons manual QA in favor of a ruthless, automated LLM-as-a-Judge (Human Preference Metric) system to benchmark accuracy.

-   **Synthetic Generation:** Randomly samples two chunks from the FAISS database to automatically generate a highly specific "Ground Truth" QA pair.

-   **Isolated Testing:** Feeds the synthetic question into the RAG pipeline with a blank memory state.
- **Regex-Parsed Grading:** A secondary LLM evaluates the RAG response against the Ground Truth. Using Regular Expressions, the script automatically parses the output, assigning a [1] (Fail) for dangerous hallucinations or missed flags, and a [2] (Pass) strictly when the core engineering data aligns.

## 💬 Sample Queries to Try
Once the app is running, try asking:

- "What is the absolute maximum voltage for GPIO 15 on the ESP32-C6?"

- "I want to connect a 5V sensor directly to GPIO 4 on the STM32F10. Is this pin 5V tolerant?"

- "On the ESP32-C6, what is the exact single-end input high threshold voltage when USB_SERIAL_JTAG_VREFH is configured to 2?"

------------------------------------------------------------------------

## 📂 Project Structure

    ├── data/               # MCU Reference Manuals here
    ├── vectorstore/        # Auto-generated FAISS index saves here
    ├── ingest.py           # Ingestion pipeline (Extract -> Chunk -> Embed)
    ├── app.py              # FastAPI backend + Gradio UI + Evaluation Logic
    └── MCU_RAG_Report.md   # Project Documentation

------------------------------------------------------------------------

## 🛠️ Usage

1.  Drop the PDF manuals into the `/data` folder.

2.  Run:

    ``` bash
    python ingest.py
    ```

3.  Start the assistant:

    ``` bash
    python app.py
    ```

4.  Access the UI at:

        http://127.0.0.1:8000


## 👨‍💻 Author
* **Krolus Samy Hebiesh** - [GitHub Profile](https://github.com/KrolusSamy)