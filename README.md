# Graph-RAG for Research Citation Discovery

**Live Application:** (https://paper-citation-using-graphrag.streamlit.app/)

## Project Overview

Graph-RAG is a structured citation recommendation system that integrates knowledge graph techniques with retrieval-augmented generation (RAG) to assist researchers in discovering relevant academic references. By modeling research papers, methods, and concepts as graph entities, the system enables deeper semantic understanding and context-aware citation retrieval.

Unlike traditional RAG models or keyword-based search engines, Graph-RAG leverages concept-level graph embeddings and hierarchical clustering to provide more accurate and explainable citation suggestions.

## Key Contributions

- Neo4j-based knowledge graph built from research papers and extracted concepts using LLaMA 3.1 8B.
- Concept-level clustering using the Leiden algorithm for improved topic segmentation.
- Semantic similarity-based retrieval pipeline using BAAI/bge-en-icl embeddings and cosine similarity.
- Streamlit-based web interface providing formatted citation recommendations.
- Evaluated performance improvements over traditional RAG using standard retrieval metrics.

## System Architecture

1. **Data Ingestion and Preprocessing**
2. **Concept Extraction via LLaMA**
3. **Graph Construction in Neo4j**
4. **Leiden Clustering**
5. **Semantic Retrieval and Ranking**
6. **Formatted Output via Streamlit UI**

## Evaluation

| Metric           | Graph-RAG | Traditional RAG |
|------------------|-----------|-----------------|
| Accuracy@Top-K   | 81.26%    | 73.42%          |
| Precision@10     | 74.61%    | 62.49%          |
| NDCG@10          | 71.44%    | 65.85%          |
| Retrieval Time   | 1.36 sec  | 1.51 sec        |

## Technologies Used

- LLMs: LLaMA 3.1 8B, GPT-4o (for benchmarking)
- Embeddings: BAAI/bge-en-icl
- Graph DB: Neo4j
- Clustering: Leiden Algorithm
- UI: Streamlit
- Backend: Python

## Future Work

- Expand to large-scale academic corpora (e.g., Semantic Scholar)
- Dynamic graph updates and continuous ingestion
- Additional export formats (BibTeX, APA)
- Incorporate user feedback for citation reranking
