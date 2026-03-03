import gradio as gr
import re
import ast
import random
from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- 1. SETUP LANGCHAIN & MODELS ---
print("Loading LLM and Embeddings...")
llm = ChatOllama(model="llama3.1", temperature=0.0)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

try:
    vectorstore = FAISS.load_local("./vectorstore", embeddings, allow_dangerous_deserialization=True)
    # Base retriever configuration
    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 8, "fetch_k": 20} 
    )
    print("FAISS Vectorstore loaded successfully!")
except Exception as e:
    print(f"Warning: No vectorstore found. ({e})")
    retriever = None

# --- 2. MULTI-QUERY & RAG PIPELINE ---
if retriever:
    # Custom prompt to force the LLM to hunt for hardware tables
    mq_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an expert hardware engineer. Generate 3 different technical search queries to find the answer to the user's question in a microcontroller datasheet.
        Always include variations that look for tables like "Absolute Maximum Ratings", "Electrical Characteristics", or "Pin Definitions".
        Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    
    # Wrap the base retriever in the MultiQueryRetriever
    hardware_mq_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=mq_prompt
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, formulate a standalone question. Do NOT answer it, just reformulate it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Use the advanced MultiQueryRetriever here
    history_aware_retriever = create_history_aware_retriever(llm, hardware_mq_retriever, contextualize_q_prompt)

    # Document Metadata Stamper
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "page"],
        template="[Source: {source}, Page: {page}]\n{page_content}\n"
    )

    # Strict Auditor System Prompt
    system_prompt = (
        "You are a strict Embedded Hardware Technical Auditor. "
        "Your ONLY function is to evaluate hardware using strictly the provided Context. "
        "You have zero outside knowledge. You must obey the following rules absolutely:\n\n"
        
        "RULE 1: THE KILL SWITCH\n"
        "IF the user asks about a microcontroller or component NOT explicitly named in the Context, "
        "THEN you must instantly stop and output EXACTLY: 'The datasheet for [Component] is not in my database.' Do not attempt to answer for a different microcontroller.\n\n"

        "CRULE 2: RITICAL RULES:\n"
        "1. ZERO OUTSIDE KNOWLEDGE: If the exact answer is not in the Context, you MUST output EXACTLY: 'Data not found in the retrieved datasheet chunks.' Do not guess. Do not provide general theory.\n"
        "2. NO META-COMMENTARY: Never say 'Based on the provided snippets...' or 'According to the context...'. Just answer the question directly.\n"
        "3. STRICT CITATIONS: Every technical claim MUST end with [Source: Filename, Page: X].\n"
        
        "RULE 3: LOGIC LEVEL GUARDRAILS\n"
        "IF a user asks about connecting a voltage (e.g., 5V) to a pin, you MUST search the Context for the exact phrase '5V tolerant'. "
        "IF that phrase is not explicitly in the Context for that specific pin, "
        "THEN you must output EXACTLY: 'WARNING: Pin is NOT confirmed 5V tolerant based on retrieved context; applying 5V may cause permanent hardware damage.'\n\n"
        
        "RULE 4: MANDATORY CITATIONS\n"
        "Every single technical claim, voltage, or formula MUST end with a citation using this exact format: [Source: Filename, Page: X]. "
        "If you do not know the page, you cannot state the fact.\n\n"
        
        "Context:\n{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt, document_prompt=document_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 3. HELPER FUNCTIONS ---
def get_mcu_filter(query: str):
    """Dynamic router to lock the retriever to a specific manual."""
    query_upper = query.upper()
    if "ESP32" in query_upper: return {"mcu": "ESP32"}
    elif "STM32" in query_upper: return {"mcu": "STM32"}
    elif "ATMEGA" in query_upper or "328" in query_upper or "ARDUINO" in query_upper: return {"mcu": "ATmega328P"}
    elif "RP2040" in query_upper or "PICO" in query_upper: return {"mcu": "RP2040"}
    return {} 

def extract_pure_text(msg):
    if isinstance(msg, dict):
        if "text" in msg: return str(msg["text"])
        if "content" in msg: return str(msg["content"])
    elif isinstance(msg, (list, tuple)) and len(msg) > 0:
        return str(msg[0])
    return str(msg)

def clean_llm_output(text):
    text = str(text).strip()
    if text.startswith("[{") and "'text':" in text:
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list) and len(parsed) > 0 and 'text' in parsed[0]:
                return parsed[0]['text']
        except:
            pass
    return text

# --- 4. GRADIO LOGIC ---
def chat_interface(message, history):
    if not retriever:
        return "System offline: No documents ingested. Run ingest.py first."
        
    clean_user_input = extract_pure_text(message)
    
    # Apply dynamic metadata filtering to block cross-manual contamination
    mcu_filter = get_mcu_filter(clean_user_input)
    search_kwargs = {"k": 8, "fetch_k": 20}
    if mcu_filter:
        search_kwargs["filter"] = mcu_filter
    retriever.search_kwargs = search_kwargs
        
    chat_history = []
    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            human, ai = item
            if human: chat_history.append(HumanMessage(content=extract_pure_text(human)))
            if ai: chat_history.append(AIMessage(content=extract_pure_text(ai)))
        elif isinstance(item, dict):
            role = item.get("role", "")
            content = extract_pure_text(item)
            if content:
                if role == "user": chat_history.append(HumanMessage(content=content))
                elif role in ["assistant", "model"]: chat_history.append(AIMessage(content=content))
    
    try:
        response = rag_chain.invoke({
            "input": clean_user_input,
            "chat_history": chat_history
        })
        return clean_llm_output(response["answer"])
    except Exception as e:
        return f"LangChain Error: {str(e)}"
def run_evaluation(num_questions):
    """Automated LLM-as-a-Judge using the NVIDIA DLI Preference Metric Style"""
    if not retriever or not vectorstore:
        return "Error: No vectorstore loaded. Run ingest.py first.", ""
    
    try:
        num_qs = int(num_questions)
    except:
        num_qs = 3

    # Access all the raw chunks currently stored in your FAISS database
    docs = list(vectorstore.docstore._dict.values())
    if len(docs) < 2:
        return "Error: Not enough documents in the database.", ""

    results_md = ""
    scores = []

    # 1. Setup Prompts (Directly from Notebook 08)
    simple_prompt = ChatPromptTemplate.from_messages([
        ('system', '{system}'), 
        ('user', 'INPUT: {input}')
    ])

    eval_prompt = ChatPromptTemplate.from_template("""INSTRUCTION: 
Evaluate the following Question-Answer pair. 
Assume Answer 1 is the Ground Truth and is factually correct.
Assume Answer 2 is the New Answer generated by an AI assistant.

SCORING CRITERIA:
[1] FAIL: Answer 2 directly contradicts the Ground Truth, completely misses the core question, or hallucinates dangerous hardware advice.
[2] PASS: Answer 2 correctly addresses the core question and aligns with the factual essence of the Ground Truth. It is completely OKAY if Answer 2 is longer, provides additional helpful context, or is formatted differently, as long as the primary facts are correct.

CRITICAL OUTPUT FORMAT:
You MUST start your response with EXACTLY the string "[1]" or "[2]".

{qa_trio}

EVALUATION: 
""")
    for i in range(num_qs):
        # 2. Sample TWO random documents like the DLI notebook
        doc1, doc2 = random.sample(docs, 2)
        source1 = doc1.metadata.get("source", "Unknown")
        source2 = doc2.metadata.get("source", "Unknown")
        
        # 3. Generate Synthetic Q&A Pair
        sys_msg = (
            "Use the documents provided by the user to generate an interesting question-answer pair."
            " Try to use both documents if possible, and rely more on the document bodies than the summary."
            " Use the format:\nQuestion: (good question, 1-3 sentences, detailed)\n\nAnswer: (answer derived from the documents)"
            " DO NOT SAY: \"Here is an interesting question pair\" or similar. FOLLOW FORMAT!"
        )
        usr_msg = (
            f"Document1: {doc1.page_content}\n\n"
            f"Document2: {doc2.page_content}"
        )

        qa_pair = (simple_prompt | llm).invoke({'system': sys_msg, 'input': usr_msg}).content
        
        try:
            synth_q = qa_pair.split("Answer:")[0].replace("Question:", "").strip()
            synth_a = qa_pair.split("Answer:")[1].strip()
        except Exception:
            continue # Skip if LLM formatted it wrong
            
        # 4. Run the RAG Pipeline
        try:
            # Apply dynamic metadata filtering to give the RAG agent a fair chance
            mcu_filter = get_mcu_filter(synth_q)
            search_kwargs = {"k": 8, "fetch_k": 20}
            if mcu_filter: 
                search_kwargs["filter"] = mcu_filter
            retriever.search_kwargs = search_kwargs

            rag_response = rag_chain.invoke({
                "input": synth_q,
                "chat_history": []
            })
            rag_a = clean_llm_output(rag_response["answer"]) 
        except Exception as e:
            rag_a = f"Pipeline Error: {str(e)}"
        
        # 5. LLM-as-a-Judge (Human Preference Metric)
        qa_trio = f"Question: {synth_q}\n\nAnswer 1 (Ground Truth): {synth_a}\n\n Answer 2 (New Answer): {rag_a}"
        eval_res = (eval_prompt | llm).invoke({'qa_trio': qa_trio}).content
        
        # 6. Score parsing (NEW REGEX LOGIC)
        # This hunts for the first number inside brackets, ignoring extra text like "Score:"
        match = re.search(r'\[.*?([12]).*?\]', eval_res[:25])
        if match and match.group(1) == '2':
            score_val = 2
        else:
            score_val = 1
            
        scores.append(score_val)
        
        # 7. Format the output for Gradio UI
        results_md += f"### QA Pair {i+1}\n"
        results_md += f"**Source Chunks:** {source1}, {source2}\n\n"
        results_md += f"**Question:** {synth_q}\n\n"
        results_md += f"**Ground Truth:** {synth_a}\n\n"
        results_md += f"**RAG Answer:** {rag_a}\n\n"
        results_md += f"**Preference Evaluation:** {eval_res}\n\n---\n"

    if not scores:
        return "Error: Failed to generate valid Q&A pairs. Try again.", ""

    # Calculate final preference percentage
    accuracy = sum([1 for s in scores if s == 2]) / len(scores) * 100
    stats_md = f"### 📊 Overall Preference Score: {accuracy:.1f}%\n"
    stats_md += f"*(The RAG pipeline achieved a preference score of [2] on {int(accuracy/100 * len(scores))} out of {len(scores)} questions)*"

    return stats_md, results_md
# --- 5. GRADIO UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft()) as gradio_app:
    gr.Markdown("# 🛠️ Smart Microcontroller Q&A Assistant")
    gr.Markdown("Ask questions about your uploaded hardware manuals. The system utilizes Multi-Query semantic search to auto-check logic levels, isolating correct microcontroller architectures and accurately citing page numbers.")
    
    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=chat_interface,
                chatbot=gr.Chatbot(height=500),
                textbox=gr.Textbox(placeholder="E.g., What is the absolute maximum voltage for GPIO 15 on the ESP32-C6?", container=False, scale=7),
            )
        
        with gr.Tab("Evaluation Metrics"):
            gr.Markdown("### ⚖️ LLM-as-a-Judge RAG Evaluation")
            with gr.Row():
                eval_slider = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Number of Test Questions")
                eval_btn = gr.Button("▶️ Run Evaluation Test", variant="primary")
                
            with gr.Row():
                with gr.Column():
                    eval_stats = gr.Markdown(label="Statistics")
                    eval_details = gr.Markdown(label="Detailed Breakdown")
            
            eval_btn.click(fn=run_evaluation, inputs=[eval_slider], outputs=[eval_stats, eval_details])

app = FastAPI(title="Microcontroller RAG API")
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    print("Starting server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)