from util import load_pdf, preprocess_text, split_documents_into_chunks
from add_paper import extract_entities
from community import detect_communities, summarize_communities
from retriever import generate_answers_from_communities


text="Transformer models have been widely used in natural language processing for various tasks such as text classification, question answering, and language translation. These models have achieved state-of-the-art performance on several benchmarks and have been shown to learn rich representations of text data. However, the interpretability of these models remains a challenge, as they are often treated as black boxes. In this work, we propose a novel approach to interpret transformer models by extracting and summarizing the key information from the attention weights of the model. We demonstrate the effectiveness of our approach on a range of tasks, including text classification, question answering, and language translation. Our results show that our approach can provide valuable insights into the inner workings of transformer models and help improve their interpretability."
query = "Which papers cite the work of the authors?"
answer = generate_answers_from_communities(text, query)
print('Answer:', answer)
