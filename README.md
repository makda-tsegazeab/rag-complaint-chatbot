<<<<<<< HEAD
# 📊 Intelligent Complaint Analysis for Financial Services

## 🎯 Business Objective
CrediTrust Financial, a fast-growing digital finance company serving East African markets, receives thousands of customer complaints monthly across multiple channels. This project develops an internal AI tool that transforms unstructured complaint data into actionable insights, empowering product managers, support teams, and compliance officers to quickly identify trends and resolve issues.

### 📈 Key Performance Indicators (KPIs)
- Decrease trend identification time from **days to minutes**
- Empower **non-technical teams** with self-service analytics
- Shift from **reactive to proactive** problem-solving

## 🏗️ Project Architecture

```
rag-complaint-chatbot/
├── .vscode/                 # VS Code workspace settings
├── .github/                # CI/CD workflows
├── data/                   # Data storage
│   ├── raw/               # Original CFPB dataset
│   ├── processed/         # Cleaned and filtered data
│   └── sampled/           # Stratified samples for development
├── vector_store/          # FAISS/ChromaDB indices
│   ├── prebuilt/          # Pre-built vector store (1.37M chunks)
│   └── custom/            # Custom-built vector stores
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code modules
│   ├── data/             # Data preprocessing and sampling
│   ├── embedding/        # Text chunking and embedding
│   ├── rag/             # RAG pipeline components
│   └── evaluation/       # Performance evaluation
├── tests/                # Unit tests
├── app.py                # Gradio/Streamlit interface
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

## 📋 Project Tasks & Progress

### ✅ **Task 1: Exploratory Data Analysis and Preprocessing** - **COMPLETED**
**Objective**: Understand and prepare CFPB complaint data for the RAG pipeline.

#### 📊 Key Findings:
- **Dataset**: Consumer Financial Protection Bureau (CFPB) complaint dataset
- **Products Analyzed**: Credit Cards, Personal Loans, Savings Accounts, Money Transfers
- **Data Quality**: ~85% of complaints contain usable narrative text
- **Average Narrative Length**: ~150 words per complaint
- **Common Issues**: Billing disputes, unauthorized charges, poor customer service

#### 🛠️ Implementation:
- **Data Cleaning**: Lowercasing, boilerplate removal, special character handling
- **Product Filtering**: Isolated four key financial product categories
- **Output**: `data/processed/filtered_complaints.csv`

#### 📁 Deliverables:
- `notebooks/01_eda_preprocessing.ipynb` - Complete EDA pipeline
- `data/processed/filtered_complaints.csv` - Cleaned dataset
- EDA visualizations and statistical analysis

---

### ✅ **Task 2: Text Chunking, Embedding, and Vector Store Indexing** - **COMPLETED**
**Objective**: Convert complaint narratives into searchable vector representations.

#### 🎯 Sampling Strategy:
- **Sample Size**: 12,500 complaints (stratified)
- **Distribution**: Proportional to original product categories
- **Method**: Random sampling without replacement

#### 📝 Chunking Approach:
- **Chunk Size**: 500 characters (matches pre-built vector store)
- **Chunk Overlap**: 50 characters (preserves context)
- **Algorithm**: Sliding window with sentence awareness
- **Metadata Preservation**: Complaint ID, product category, issue type, dates

#### 🤖 Embedding Model:
- **Selected Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Rationale**: Matches pre-built store, efficient, accurate for financial text
- **Performance**: Optimized for semantic similarity tasks

#### 🗄️ Vector Store:
- **Technology**: ChromaDB with persistent storage
- **Metadata Schema**: Comprehensive complaint context
- **Location**: `vector_store/custom/`

#### 📁 Deliverables:
- `data/sampled/complaints_sample.csv` - Stratified sample
- `data/sampled/sampling_strategy.txt` - Documentation
- `vector_store/custom/` - Vector database structure
- Chunking and embedding implementation code

---

### 🔄 **Task 3: Building the RAG Core Logic and Evaluation** - **IN PROGRESS**
**Objective**: Develop and evaluate the Retrieval-Augmented Generation pipeline.

#### 🧠 RAG Pipeline Components:

##### 1. **Retriever Module** (`src/rag/retriever.py`)
- **Semantic Search**: Cosine similarity using pre-built embeddings
- **Query Processing**: Dynamic embedding generation for user questions
- **Filtering Support**: Product category, date range, issue type filters
- **Fallback Mechanism**: Graceful degradation when pre-built store unavailable

##### 2. **Generator Module** (`src/rag/generator.py`)
- **LLM Integration**: Support for HuggingFace models and OpenAI API
- **Prompt Engineering**: Specialized templates for financial analysis
- **Context Management**: Intelligent context formatting and truncation
- **Answer Generation**: Concise, evidence-based responses

##### 3. **Pipeline Orchestration** (`src/rag/pipeline.py`)
- **End-to-End Flow**: Query → Retrieval → Generation → Response
- **Performance Tracking**: Timing, source quality, answer relevance
- **Batch Processing**: Support for multiple simultaneous queries
- **Trend Analysis**: Automated complaint pattern detection

##### 4. **Evaluation Framework** (`src/evaluation/metrics.py`)
- **Test Suite**: 10 representative financial complaint questions
- **Quality Metrics**: Answer relevance, source accuracy, response time
- **Comparative Analysis**: Performance across product categories
- **Improvement Tracking**: Iterative development feedback

#### 📊 Evaluation Questions & Results:

| Question | Category | Quality Score | Key Insights |
|----------|----------|---------------|--------------|
| What are common credit card complaints? | Credit Cards | 8/10 | Unauthorized charges, billing errors main issues |
| Issues with savings accounts? | Savings Accounts | 7/10 | Fees and withdrawal restrictions common |
| Problems with money transfers? | Money Transfers | 9/10 | Failed transactions and delays frequent |
| Personal loan complaints? | Personal Loans | 8/10 | Application process and rates problematic |
| Compare issues across products | Comparison | 8/10 | Clear differentiation between product issues |

#### 📁 Deliverables:
- Complete RAG pipeline implementation
- Evaluation framework with scoring metrics
- Performance analysis report
- Sample responses and source verification

---

### ⏳ **Task 4: Creating an Interactive Chat Interface** - **UP NEXT**
**Objective**: Build user-friendly interface for non-technical stakeholders.

#### 🎨 Interface Requirements:
- **Natural Language Input**: Plain English questions about complaints
- **Source Transparency**: Display supporting complaint excerpts
- **Response Streaming**: Token-by-token answer generation
- **Filter Controls**: Product category, date range, company filters
- **Session Management**: Query history and result saving

#### 💻 Technology Stack:
- **Frontend Options**: Gradio (lightweight) or Streamlit (feature-rich)
- **Backend Integration**: REST API with RAG pipeline
- **Deployment**: Local server with potential cloud scaling

#### 🎯 User Experience Goals:
- **Asha (Product Manager)**: Identify credit card trends in minutes
- **Support Team**: Find similar complaints for faster resolution
- **Compliance Officer**: Detect fraud patterns proactively
- **Executive**: Dashboard view of emerging issues

#### 📁 Planned Deliverables:
- `app.py` - Complete chat interface application
- UI screenshots and demonstration video
- User documentation and quick-start guide
- Deployment instructions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 5GB disk space

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/rag-complaint-chatbot.git
cd rag-complaint-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup
1. Place CFPB dataset in `data/raw/complaints.csv`
2. Place pre-built embeddings in `data/raw/complaint_embeddings.parquet`
3. Run Task 1: `jupyter notebook notebooks/01_eda_preprocessing.ipynb`

### Running the Application
```bash
# Development mode
python app.py

# Or run individual tasks
python -m src.rag.pipeline  # Test RAG pipeline
python -m src.evaluation.metrics  # Run evaluation
```

## 🔧 Technical Implementation Details

### Data Pipeline
```
Raw CFPB Data → Cleaning → Filtering → Chunking → Embedding → Vector Store
```

### RAG Architecture
```
User Query → Query Embedding → Semantic Search → Context Retrieval → 
LLM Prompting → Answer Generation → Source Attribution → Response
```

### Performance Optimization
- **Caching**: Embedding and retrieval result caching
- **Batch Processing**: Parallel embedding generation
- **Index Tuning**: FAISS/ChromaDB parameter optimization
- **Model Selection**: Balance between accuracy and speed

## 📈 Evaluation Metrics

### Retrieval Quality
- **Recall@K**: Relevant documents in top K results
- **Mean Reciprocal Rank**: Ranking of first relevant document
- **Precision**: Proportion of relevant retrieved documents

### Generation Quality
- **ROUGE Scores**: Text similarity with ground truth
- **BERTScore**: Semantic similarity evaluation
- **Human Evaluation**: Expert rating of answer quality

### System Performance
- **Response Time**: End-to-end query processing
- **Throughput**: Queries processed per minute
- **Scalability**: Performance with increasing data volume

## 🏆 Learning Outcomes

Through this project, you will gain expertise in:

1. **RAG Systems**: Combining retrieval and generation for accurate Q&A
2. **Vector Databases**: Implementing semantic search with FAISS/ChromaDB
3. **Text Processing**: Handling noisy, unstructured customer feedback
4. **LLM Integration**: Prompt engineering and model selection
5. **Evaluation Methods**: Quantitative and qualitative system assessment
6. **Production Deployment**: Building scalable, user-friendly interfaces

## 📅 Project Timeline

| Date | Milestone | Status |
|------|-----------|---------|
| Dec 31, 2025 | Project Kickoff | ✅ |
| Jan 4, 2026 | Interim Submission (Tasks 1-2) | ✅ |
| Jan 13, 2026 | Final Submission (Tasks 3-4) | 🔄 |

## 👥 Team & Acknowledgments

**Facilitators**: Kerod, Mahbubah, Filimon, Smegnsh

**Data Source**: Consumer Financial Protection Bureau (CFPB)

**Special Thanks**: The open-source community for tools and libraries

## 📚 References

1. CFPB Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Sentence Transformers: https://www.sbert.net/
3. ChromaDB Documentation: https://docs.trychroma.com/
4. LangChain RAG Implementation: https://python.langchain.com/

## 📄 License

This project is developed for educational purposes as part of a data engineering challenge.

---

*Last Updated: January 4, 2026*  
*Project Status: Interim Submission Complete - Tasks 1 & 2*  
*Next Milestone: Final Submission (Jan 13, 2026) - Tasks 3 & 4*
=======
# 📊 Intelligent Complaint Analysis for Financial Services

## 🎯 Business Objective
CrediTrust Financial, a fast-growing digital finance company serving East African markets, receives thousands of customer complaints monthly across multiple channels. This project develops an internal AI tool that transforms unstructured complaint data into actionable insights, empowering product managers, support teams, and compliance officers to quickly identify trends and resolve issues.

### 📈 Key Performance Indicators (KPIs)
- Decrease trend identification time from **days to minutes**
- Empower **non-technical teams** with self-service analytics
- Shift from **reactive to proactive** problem-solving

## 🏗️ Project Architecture

```
rag-complaint-chatbot/
├── .vscode/                 # VS Code workspace settings
├── .github/                # CI/CD workflows
├── data/                   # Data storage
│   ├── raw/               # Original CFPB dataset
│   ├── processed/         # Cleaned and filtered data
│   └── sampled/           # Stratified samples for development
├── vector_store/          # FAISS/ChromaDB indices
│   ├── prebuilt/          # Pre-built vector store (1.37M chunks)
│   └── custom/            # Custom-built vector stores
├── notebooks/             # Jupyter notebooks for analysis
├── src/                   # Source code modules
│   ├── data/             # Data preprocessing and sampling
│   ├── embedding/        # Text chunking and embedding
│   ├── rag/             # RAG pipeline components
│   └── evaluation/       # Performance evaluation
├── tests/                # Unit tests
├── app.py                # Gradio/Streamlit interface
├── requirements.txt      # Python dependencies
└── README.md            # This documentation
```

## 📋 Project Tasks & Progress

### ✅ **Task 1: Exploratory Data Analysis and Preprocessing** - **COMPLETED**
**Objective**: Understand and prepare CFPB complaint data for the RAG pipeline.

#### 📊 Key Findings:
- **Dataset**: Consumer Financial Protection Bureau (CFPB) complaint dataset
- **Products Analyzed**: Credit Cards, Personal Loans, Savings Accounts, Money Transfers
- **Data Quality**: ~85% of complaints contain usable narrative text
- **Average Narrative Length**: ~150 words per complaint
- **Common Issues**: Billing disputes, unauthorized charges, poor customer service

#### 🛠️ Implementation:
- **Data Cleaning**: Lowercasing, boilerplate removal, special character handling
- **Product Filtering**: Isolated four key financial product categories
- **Output**: `data/processed/filtered_complaints.csv`

#### 📁 Deliverables:
- `notebooks/01_eda_preprocessing.ipynb` - Complete EDA pipeline
- `data/processed/filtered_complaints.csv` - Cleaned dataset
- EDA visualizations and statistical analysis

---

### ✅ **Task 2: Text Chunking, Embedding, and Vector Store Indexing** - **COMPLETED**
**Objective**: Convert complaint narratives into searchable vector representations.

#### 🎯 Sampling Strategy:
- **Sample Size**: 12,500 complaints (stratified)
- **Distribution**: Proportional to original product categories
- **Method**: Random sampling without replacement

#### 📝 Chunking Approach:
- **Chunk Size**: 500 characters (matches pre-built vector store)
- **Chunk Overlap**: 50 characters (preserves context)
- **Algorithm**: Sliding window with sentence awareness
- **Metadata Preservation**: Complaint ID, product category, issue type, dates

#### 🤖 Embedding Model:
- **Selected Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Rationale**: Matches pre-built store, efficient, accurate for financial text
- **Performance**: Optimized for semantic similarity tasks

#### 🗄️ Vector Store:
- **Technology**: ChromaDB with persistent storage
- **Metadata Schema**: Comprehensive complaint context
- **Location**: `vector_store/custom/`

#### 📁 Deliverables:
- `data/sampled/complaints_sample.csv` - Stratified sample
- `data/sampled/sampling_strategy.txt` - Documentation
- `vector_store/custom/` - Vector database structure
- Chunking and embedding implementation code

---

### 🔄 **Task 3: Building the RAG Core Logic and Evaluation** - **IN PROGRESS**
**Objective**: Develop and evaluate the Retrieval-Augmented Generation pipeline.

#### 🧠 RAG Pipeline Components:

##### 1. **Retriever Module** (`src/rag/retriever.py`)
- **Semantic Search**: Cosine similarity using pre-built embeddings
- **Query Processing**: Dynamic embedding generation for user questions
- **Filtering Support**: Product category, date range, issue type filters
- **Fallback Mechanism**: Graceful degradation when pre-built store unavailable

##### 2. **Generator Module** (`src/rag/generator.py`)
- **LLM Integration**: Support for HuggingFace models and OpenAI API
- **Prompt Engineering**: Specialized templates for financial analysis
- **Context Management**: Intelligent context formatting and truncation
- **Answer Generation**: Concise, evidence-based responses

##### 3. **Pipeline Orchestration** (`src/rag/pipeline.py`)
- **End-to-End Flow**: Query → Retrieval → Generation → Response
- **Performance Tracking**: Timing, source quality, answer relevance
- **Batch Processing**: Support for multiple simultaneous queries
- **Trend Analysis**: Automated complaint pattern detection

##### 4. **Evaluation Framework** (`src/evaluation/metrics.py`)
- **Test Suite**: 10 representative financial complaint questions
- **Quality Metrics**: Answer relevance, source accuracy, response time
- **Comparative Analysis**: Performance across product categories
- **Improvement Tracking**: Iterative development feedback

#### 📊 Evaluation Questions & Results:

| Question | Category | Quality Score | Key Insights |
|----------|----------|---------------|--------------|
| What are common credit card complaints? | Credit Cards | 8/10 | Unauthorized charges, billing errors main issues |
| Issues with savings accounts? | Savings Accounts | 7/10 | Fees and withdrawal restrictions common |
| Problems with money transfers? | Money Transfers | 9/10 | Failed transactions and delays frequent |
| Personal loan complaints? | Personal Loans | 8/10 | Application process and rates problematic |
| Compare issues across products | Comparison | 8/10 | Clear differentiation between product issues |

#### 📁 Deliverables:
- Complete RAG pipeline implementation
- Evaluation framework with scoring metrics
- Performance analysis report
- Sample responses and source verification

---

### ⏳ **Task 4: Creating an Interactive Chat Interface** - **UP NEXT**
**Objective**: Build user-friendly interface for non-technical stakeholders.

#### 🎨 Interface Requirements:
- **Natural Language Input**: Plain English questions about complaints
- **Source Transparency**: Display supporting complaint excerpts
- **Response Streaming**: Token-by-token answer generation
- **Filter Controls**: Product category, date range, company filters
- **Session Management**: Query history and result saving

#### 💻 Technology Stack:
- **Frontend Options**: Gradio (lightweight) or Streamlit (feature-rich)
- **Backend Integration**: REST API with RAG pipeline
- **Deployment**: Local server with potential cloud scaling

#### 🎯 User Experience Goals:
- **Asha (Product Manager)**: Identify credit card trends in minutes
- **Support Team**: Find similar complaints for faster resolution
- **Compliance Officer**: Detect fraud patterns proactively
- **Executive**: Dashboard view of emerging issues

#### 📁 Planned Deliverables:
- `app.py` - Complete chat interface application
- UI screenshots and demonstration video
- User documentation and quick-start guide
- Deployment instructions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 5GB disk space

### Installation
```bash
# Clone repository
git clone https://github.com/your-username/rag-complaint-chatbot.git
cd rag-complaint-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Setup
1. Place CFPB dataset in `data/raw/complaints.csv`
2. Place pre-built embeddings in `data/raw/complaint_embeddings.parquet`
3. Run Task 1: `jupyter notebook notebooks/01_eda_preprocessing.ipynb`

### Running the Application
```bash
# Development mode
python app.py

# Or run individual tasks
python -m src.rag.pipeline  # Test RAG pipeline
python -m src.evaluation.metrics  # Run evaluation
```

## 🔧 Technical Implementation Details

### Data Pipeline
```
Raw CFPB Data → Cleaning → Filtering → Chunking → Embedding → Vector Store
```

### RAG Architecture
```
User Query → Query Embedding → Semantic Search → Context Retrieval → 
LLM Prompting → Answer Generation → Source Attribution → Response
```

### Performance Optimization
- **Caching**: Embedding and retrieval result caching
- **Batch Processing**: Parallel embedding generation
- **Index Tuning**: FAISS/ChromaDB parameter optimization
- **Model Selection**: Balance between accuracy and speed

## 📈 Evaluation Metrics

### Retrieval Quality
- **Recall@K**: Relevant documents in top K results
- **Mean Reciprocal Rank**: Ranking of first relevant document
- **Precision**: Proportion of relevant retrieved documents

### Generation Quality
- **ROUGE Scores**: Text similarity with ground truth
- **BERTScore**: Semantic similarity evaluation
- **Human Evaluation**: Expert rating of answer quality

### System Performance
- **Response Time**: End-to-end query processing
- **Throughput**: Queries processed per minute
- **Scalability**: Performance with increasing data volume

## 🏆 Learning Outcomes

Through this project, you will gain expertise in:

1. **RAG Systems**: Combining retrieval and generation for accurate Q&A
2. **Vector Databases**: Implementing semantic search with FAISS/ChromaDB
3. **Text Processing**: Handling noisy, unstructured customer feedback
4. **LLM Integration**: Prompt engineering and model selection
5. **Evaluation Methods**: Quantitative and qualitative system assessment
6. **Production Deployment**: Building scalable, user-friendly interfaces

## 📅 Project Timeline

| Date | Milestone | Status |
|------|-----------|---------|
| Dec 31, 2025 | Project Kickoff | ✅ |
| Jan 4, 2026 | Interim Submission (Tasks 1-2) | ✅ |
| Jan 13, 2026 | Final Submission (Tasks 3-4) | 🔄 |

## 👥 Team & Acknowledgments

**Facilitators**: Kerod, Mahbubah, Filimon, Smegnsh

**Data Source**: Consumer Financial Protection Bureau (CFPB)

**Special Thanks**: The open-source community for tools and libraries

## 📚 References

1. CFPB Complaint Database: https://www.consumerfinance.gov/data-research/consumer-complaints/
2. Sentence Transformers: https://www.sbert.net/
3. ChromaDB Documentation: https://docs.trychroma.com/
4. LangChain RAG Implementation: https://python.langchain.com/

## 📄 License

This project is developed for educational purposes as part of a data engineering challenge.

---

*Last Updated: January 4, 2026*  
*Project Status: Interim Submission Complete - Tasks 1 & 2*  
*Next Milestone: Final Submission (Jan 13, 2026) - Tasks 3 & 4*
>>>>>>> chat-interface
