# Zero-shot LLM-Assisted Expert Finding

A comprehensive system for finding academic experts based on natural language queries using zero-shot framework with Large Language Models (LLMs), embeddings, and RDF knowledge graphs.

## Overview

This project implements an end-to-end academic expert finding pipeline that combines:
- **Query Processing**: Extracting keywords from user queries using LLMs
- **Document Retrieval**: Finding relevant papers using semantic similarity 
- **Document Reranking**: Using reranker model to score document relevance
- **Knowledge Graph Construction**: Building RDF-based knowledge graphs linking papers, authors, and topics
- **Author Ranking**: Re-ranking candidate experts using LLM-based relevance assessment with graph context
- **Evaluation**: Comprehensive metrics (MAP, MRR, NDCG, Precision) for ranking quality

## Key Features

- **Zero-Shot Expert Finding**: No task-specific training required
- **Multi-Model Ensemble**: Supports multiple embedding and LLM backends (Qwen, LLaMA, SPECTER)
- **Knowledge Graph Context**: 1-hop and 2-hop graph traversal for enhanced author context
- **Hybrid Ranking**: Combines citation-based, semantic similarity, and LLM-based relevance signals
- **Comprehensive Evaluation**: Multiple ranking metrics for thorough performance assessment
- **Efficient Caching**: RDF graph caching for improved performance with large knowledge graphs

## Project Structure

```
├── main.py                    # Main pipeline orchestrating the expert finding process
├── evaluation.py              # Evaluation metrics (MAP@K, MRR@K, NDCG@K, Precision@K)
├── exp_filter_data.py         # Document retrieval and reranking 
├── exp_knowledge_graph.py     # RDF knowledge graph construction and traversal
├── llama_config.json          # Configuration for LLaMA model
└── README.md                  # This file
```

## Core Components

### 1. **main.py** - Pipeline Orchestration
The main pipeline that coordinates the entire expert finding process:

**Key Functions:**
- `extract_keywords_from_query(prompt)`: Uses LLM to extract domain keywords from queries
- `get_embedding_from_specter()`: Encodes papers and queries using Qwen embeddings
- `get_avg_sentence_embedding_from_qwen()`: Computes average sentence embeddings for papers
- `validate_authors_binary()`: Binary relevance validation of candidate experts
- `main()`: Orchestrates the full pipeline for multiple queries

**Process Flow:**
1. Extract keywords from query using LLM
2. Compute query embedding with Qwen model
3. Retrieve similar documents via cosine similarity
4. Rerank documents using Qwen reranker
5. Aggregate papers by author to identify experts
6. Validate and rank authors using LLM with knowledge graph context

### 2. **evaluation.py** - Ranking Quality Assessment
Implements standard IR evaluation metrics:

**Metrics:**
- **MAP@K** (Mean Average Precision): Average precision at cutoff K
- **MRR@K** (Mean Reciprocal Rank): Position of first relevant item
- **MP@K** (Mean Precision@K): Precision at cutoff K
- **NDCG@K** (Normalized Discounted Cumulative Gain): Ranking quality considering position
- **Recall**: Global recall across all ground truth experts

**Ground Truth Generation:**
- `get_ground_truth_experts()`: Filters authors by topic tags 

### 3. **exp_filter_data.py** - Document Retrieval & Reranking
Handles document filtering, retrieval, and relevance scoring:

**Key Functions:**
- `rerank_documents_qwen()`: Uses Qwen3-Reranker to score document relevance
- `score_docs_with_llm()`: LLM-based document scoring with configurable thresholds
- `filter_cosine_and_select_top_k_docs()`: Cosine similarity-based initial filtering
- `retrieve_similar_documents_qwen()`: Semantic retrieval using embeddings

**Reranking Strategy:**
- Uses Qwen3-Reranker-0.6B model
- Formats prompt with query and document content
- Extracts probability scores for relevance prediction
- Supports GPU acceleration with batch processing

### 4. **exp_knowledge_graph.py** - RDF Knowledge Graph
Constructs and queries RDF-based knowledge graphs:

**Key Functions:**
- `construct_knowledge_graph()`: Builds RDF graph from papers and authors
- `_load_graph()`: Cached graph loading for performance
- `get_author_context()`: Retrieves 1-hop or 2-hop author context

**Graph Structure:**
```
Paper --hasAuthor--> Author --wrotePaper--> Paper
Paper --hasTopic--> Topic
Author --hasName--> Name (literal)
```

**Ablation Study Support:**
- `CONTEXT_MODE`: Switch between "1hop" and "2hop" graph traversal
- Enables controlled experiments on context impact

## Configuration

### Model Selection
Configure in `main.py`:
```python
CONTEXT_MODE = "1hop"  # or "2hop" for ablation study
ENGINE = "qwen"        # "llama" or "qwen"
```

### LLaMA Configuration
Create `llama_config.json`:
```json
{
  "model_name": "model_path",
  "token": "huggingface_token"
}
```

## Input Data Requirements

The system expects CSV files:
- **papers_df_with_topics_qwen.csv**: Paper data with columns: `id`, `title`, `abstract`, `topics`, `embeddings`, `authors`
- **authors.csv**: Author data with columns: `id`, `name`, `n_citation`, `tags`

## Usage

### Basic Pipeline
```python
python main.py
```

The script processes a predefined list of queries and:
1. Retrieves relevant papers
2. Ranks papers by relevance
3. Identifies candidate authors
4. Validates and ranks authors
5. Evaluates against ground truth
6. Reports metrics (MAP@10, MRR@10, NDCG@5, etc.)

### Evaluation
```python
from evaluation import evaluate_expert_ranking
results = evaluate_expert_ranking(predicted_expert_ids, ground_truth_ids)
print(results)
```

### Knowledge Graph Construction
```python
from exp_knowledge_graph import construct_knowledge_graph
construct_knowledge_graph(papers_df, authors_df)
```

## Dependencies

- **Deep Learning**: `torch`, `transformers`
- **Embeddings**: `sentence-transformers` (Qwen, SPECTER)
- **Text Processing**: `nltk`, `regex`, `scikit-learn`
- **Knowledge Graphs**: `rdflib`
- **Data**: `pandas`, `numpy`
- **Utilities**: `tqdm`

## Model Zoo

| Model | Purpose | Source |
|-------|---------|--------|
| Qwen3-Embedding-0.6B | Dense embeddings | Qwen |
| Qwen3-Reranker-0.6B | Document reranking | Qwen |
| allenai/specter | Paper embeddings (fallback) | AllenAI |
| LLaMA | Keyword extraction & author validation | Meta |
| Qwen | Keyword extraction & author validation | Qwen |

## Ablation Studies

### Context Modes
Compare expert ranking quality with different graph context:
1. Run with `CONTEXT_MODE = "1hop"` - Direct author context
2. Run with `CONTEXT_MODE = "2hop"` - Extended author context

### LLM Selection
Test impact of different LLMs:
1. Set `ENGINE = "llama"` for LLaMA-based ranking
2. Set `ENGINE = "qwen"` for Qwen-based ranking

## Performance Optimization

- **Graph Caching**: RDF graphs cached in-memory to avoid repeated disk I/O
- **Batch Processing**: Vectorized embeddings and reranking operations
- **GPU Support**: CUDA acceleration for transformer models
- **Token Limit**: Truncation at 256 tokens for reranker input

## Output

The system generates:
- Ranked lists of expert author IDs for each query
- Evaluation metrics comparing against ground truth
- RDF knowledge graph files (`.rdf` format)
- CSV files with rankings and scores

## Future Enhancements

- Integration with additional data sources (conferences, journals)
- Multi-language support
- Real-time expert finding via API
- Interactive visualization of author-paper networks
- Temporal analysis of expertise evolution

## References

The system builds on:
- Semantic Scholar's SPECTER embeddings
- Qwen3 family of models
- RDF/XML knowledge graph standards
- Standard IR evaluation metrics

## License

N/A

## Contact

Anonymous N/A
