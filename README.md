# SmartResearch

SmartResearch is a Streamlit-based web application that enables users to search, ingest, and summarize academic papers from arXiv. It leverages **LangChain**, **LangGraph**, **ChromaDB**, and **Groq** to provide a powerful research assistant capable of fetching papers, performing vector-based similarity searches, generating structured summaries, and creating multi-step research outputs like method comparisons and presentation slide outlines.

## Features
- **Paper Ingestion**: Fetch and ingest up to 200 papers from arXiv based on a user query.
- **Vector Search**: Perform similarity searches using ChromaDB and SentenceTransformer embeddings to retrieve the most relevant papers.
- **RAG Summarization**: Generate structured summaries of retrieved papers with sections like Contributions, Methods, Datasets, Key Results, Limitations, Citations, and Abstract.
- **Agentic Workflows**: Use a LangGraph-style planner to execute multi-step tasks such as summarizing papers, comparing methods, and generating slide outlines.
- **Interactive UI**: Built with Streamlit for an intuitive user experience, allowing users to input queries, adjust parameters, and view results.

## Installation

### Prerequisites
- Python 3.8+
- A Groq API key (set up in a `.env` file)
- Git (for cloning the repository)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/SmartResearch.git
   cd SmartResearch
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Create a `.env` file in the project root with the following:
   ```plaintext
   GROQ_API_KEY=<your-groq-api-key>
   GROQ_API_BASE=https://api.groq.com
   CHROMA_PERSIST_DIR=./data/chroma_db
   ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

   The app will open in your default web browser at `http://localhost:8501`.

## Usage
1. **Enter a Research Query**: Input a topic or query (e.g., "machine learning transformers") in the text box.
2. **Ingest Papers**: Specify the number of papers to fetch from arXiv (default: 50) and click "Ingest from arXiv" to populate the vector database.
3. **Retrieve & Summarize**: Use the "Retrieve & Summarize (RAG)" button to search the vector database and generate a structured summary of the top-k results (default: 5).
4. **Run Multi-Step Planner**: Click "Run multi-step planner" to execute tasks like summarizing papers, comparing methods, and generating a slide outline.

## Project Structure
- `app.py`: Main Streamlit application tying together ingestion, search, and summarization.
- `groq_llm.py`: Custom LangChain-compatible wrapper for the Groq API.
- `ingest.py`: Handles fetching and ingesting arXiv papers into the vector store.
- `vstore.py`: Manages the ChromaDB vector store and embeddings.
- `utils.py`: Utility functions for text chunking, prompt building, and title cleaning.
- `lc_agent.py`: Implements the LangChain agent and LangGraph planner for summarization and multi-step tasks.
- `requirements.txt`: Lists all Python dependencies.
- `.env`: Stores environment variables (not tracked in Git).

## Dependencies
- `streamlit`: For the web interface.
- `langchain`, `langgraph`, `langchain-community`: For agentic workflows and RAG.
- `chromadb`: For vector storage and similarity search.
- `sentence-transformers`: For generating embeddings.
- `arxiv`: For fetching papers from arXiv.
- `groq`: For LLM inference via the Groq API.
- Others: `python-dotenv`, `tqdm`, `pypdf`, `python-pptx`, `requests`, `rich`

## Notes
- Ensure your Groq API key is valid and has sufficient quota. Obtain one from [xAI](https://x.ai/api).
- The vector database is stored locally in `./data/chroma_db`. Ensure this directory exists or is writable.
- The application is designed for academic research and assumes access to arXiv papers' abstracts. PDF downloading and chunking are optional and not fully implemented in this version.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open an issue on GitHub or contact the maintainer at `<kasarashish89@gmai.com>`.
