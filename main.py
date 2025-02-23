import networkx as nx 
import os
from networkx.algorithms import community  
from cdlib.algorithms import leiden  
import requests
import fitz 
import matplotlib.pyplot as plt
import networkx as nx
import os
import requests
import json
import os
import openai
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from openai import OpenAI

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)


# TODO: Function to load pdfs and divide them into documents. Can start with 5-10 related papers lets say transformer paper and then papers which cited it and so on.


# Function to load and clean PDF text
def load_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract clean text
        cleaned_text = preprocess_text(text)
        return cleaned_text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

# Preprocess extracted text (removes unwanted characters, special tokens, etc.)
def preprocess_text(text):
    # Remove special tokens like <EOS>, <pad>, etc.
    text = re.sub(r'<\s*EOS\s*>', '', text)  # Remove <EOS> tokens
    text = re.sub(r'<\s*pad\s*>', '', text)  # Remove <pad> tokens
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = text.strip()  # Strip leading and trailing spaces
    return text

# 1. Source Documents → Text Chunks
def split_documents_into_chunks(document, chunk_size=3500, overlap_size=300):
    chunks = []
    # Split the document into chunks based on chunk size and overlap
    for i in range(0, len(document), chunk_size - overlap_size):
        chunk = document[i:i + chunk_size]
        # Check if chunk is too short, if so, join with the previous chunk
        if len(chunk) < chunk_size:
            if chunks:
                chunks[-1] += " " + chunk
            else:
                chunks.append(chunk)
        else:
            chunks.append(chunk)
    return chunks


# TODO: Add metadata to documents = {title, authors, publication date, citations, etc.}

# TODO: Merge documents based on concepts and metadata to form nodes.

def cluster_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    clustering = DBSCAN(eps=0.5, min_samples=2).fit(embeddings)
    clusters = {}
    for idx, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(chunks[idx])
    return list(clusters.values())

# Step 3: Extract entities and relationships using LLM
import json

import json
import os

def extract_entities(chunk, schema, output_file="extracted_entities.json"):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": f"""Extract entities and relationships from the given research paper text and return them as a valid JSON object in the following format: {schema}.

Instructions:
1. Identify key entities in the research text and classify them as:
   - **Concepts**: Fundamental ideas, theories, principles, or abstract entities in the research domain.
     - Examples: 'Gradient Descent', 'Transformers', 'Attention', 'Optimization'.
   - **Methods**: Specific techniques, algorithms, models, or procedural approaches used to achieve a goal.
     - Examples: 'Backpropagation', 'Adam Optimizer', 'Self-Attention Mechanism'.
     
2. Create nodes for these identified entities:
   - Each node must have a unique ID, a type (either 'Concept' or 'Method'), and a short summary (1-2 sentences explaining its role in the research context).
   
3. Identify relationships between these entities and create edges:
   - Define meaningful relationships between concepts and methods using descriptive labels (e.g., 'Optimizes', 'Extends', 'Depends on', 'Implemented using', etc.).
   - Ensure each edge links a valid source and target node.
   
4. Strictly return only a valid JSON object in the format: {{'nodes': [], 'edges': []}}, with no extra text or explanations.
"""  

            },
            {
                "role": "user",
                "content": chunk
            }
        ]
    )

    raw_output = response.choices[0].message.content.strip()
    
    # Try to extract JSON from between code blocks if present
    if "```" in raw_output:
        # Split by ``` and take the middle part
        parts = raw_output.split("```")
        if len(parts) >= 3:
            # Take the content between first and second ```
            # Remove any language identifier (like 'json') if present
            json_string = parts[1].strip().split('\n', 1)[-1]
        else:
            json_string = raw_output
    else:
        json_string = raw_output
    
    # Replace single quotes with double quotes
    json_string = json_string.replace("'", '"')
    print("🔍 Raw json_string:", json_string)
    
    extracted_data = {"nodes": [], "edges": []}
    
    try:
        parsed_data = json.loads(json_string)
        if isinstance(parsed_data, dict):
            extracted_data["nodes"] = parsed_data.get("nodes", [])
            extracted_data["edges"] = parsed_data.get("edges", [])
            
        # Validate types
        if not isinstance(extracted_data["nodes"], list):
            extracted_data["nodes"] = []
        if not isinstance(extracted_data["edges"], list):
            extracted_data["edges"] = []
            
    except json.JSONDecodeError as e:
        print(f"❌ Error: LLM did not return valid JSON. Details: {str(e)}")
        
        # Generate a summary for the given chunk
        

        # Load existing JSON file if it exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {"nodes": [], "edges": [], "summary": "", "chunk_count": 0}
    else:
        existing_data = {"nodes": [], "edges": [], "summary": "", "chunk_count": 0}

    # Increment chunk number
    chunk_number = existing_data.get("chunk_count", 0) + 1
    print("\nChunk Number:\n",chunk_number)
    chunk_prefix = f"C{chunk_number}_"

    # Generate unique IDs with chunk prefix
    new_nodes = []
    id_counter = 1
    node_id_map = {}

    for node in extracted_data.get("nodes", []):
        new_id = f"{chunk_prefix}{id_counter}"
        node_id_map[node["id"]] = new_id  # Map old ID to new ID
        node["id"] = new_id
        new_nodes.append(node)
        id_counter += 1

    # Update edges with new IDs
    new_edges = []
    for edge in extracted_data.get("edges", []):
        if edge["source"] in node_id_map and edge["target"] in node_id_map:
            edge["source"] = node_id_map[edge["source"]]
            edge["target"] = node_id_map[edge["target"]]
            new_edges.append(edge)

    # Append new data to existing data
    existing_data["nodes"].extend(new_nodes)
    existing_data["edges"].extend(new_edges)

    # Update chunk count
    existing_data["chunk_count"] = chunk_number

    # Write the updated JSON back to the file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)

    return existing_data  # Return updated JSON


# Step 4: Merge duplicate nodes using LLM
def merge_nodes(nodes):
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "Merge duplicate nodes. Return unified nodes as JSON."
            },
            {
                "role": "user",
                "content": str(nodes)
            }
        ]
    )
    return eval(response.choices[0].message.content)

# Step 5: Visualize the knowledge graph
def visualize_knowledge_graph(nodes, edges):
    G = nx.DiGraph()

    # Add nodes with labels
    for node in nodes:
        G.add_node(node['id'], label=node['label'])

    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'])

    # Draw the graph
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=labels,
        node_size=2000,
        node_color='lightblue',
        font_size=10,
        font_color='black',
        font_weight='bold',
        arrows=True,
    )
    plt.title('Knowledge Graph Visualization')
    plt.show()
if __name__ == "__main__":
    # Example input text (replace with your research paper content)
    # text = load_pdf('data/AttentionIsAllYouNeed.pdf')  
    # if text:
    #     chunks = split_documents_into_chunks(text)
    #     print(f"Number of chunks: {len(chunks)}")
    #     # print(f"First chunk: {chunks[0]}")
    # else:
    #     print("Failed to load or clean the text.")
    #     exit()

    # schema = {
    #     "nodes": [
    #         {
    #         "id": "1|2|3 etc",
    #         "label":"intutive label",
    #         "type": "Concept | Method",
    #         "summary": "Short summary of the node"
    #         }
    #     ],
    #     "edges": [
    #         {
    #         "source": "source_node_id",
    #         "target": "target_node_id",
    #         "label": "Relationship description"
    #         }
    #     ]
        # }


    
    # Cluster similar chunks (optional step for large documents)
    # clusters = cluster_chunks(chunks)

    # all_nodes, all_edges = [], []

    # # # Extract entities and relationships from each cluster or chunk
    # # for cluster in chunks:
    # for chunk in chunks:
    #     print("\nChunk:\n",chunk)
    #     result = extract_entities(chunk, schema)
    #     all_nodes.extend(result["nodes"])
    #     all_edges.extend(result["edges"])

    # # Merge duplicate nodes
    # merged_nodes = merge_nodes(all_nodes)

    # # Visualize the knowledge graph
    #load json extraced_entities.json
    with open('extracted_entities.json', 'r') as f:
        data = json.load(f)
    merged_nodes=data['nodes']
    all_edges=data['edges']
    visualize_knowledge_graph(merged_nodes, all_edges)
# usage of client
# response = client.chat.completions.create(
#     model="meta-llama/Llama-3.3-70B-Instruct",
#     temperature=0,
#     messages=[{"role": "system", "content": "Extract entities and relationships from the following text."}, 
#               {"role": "user", "content": "John is a software engineer at TechCorp."}]
# )

# print(response.choices[0].message.content)
# # TODO: Identify what elements we want to extract. Experiment with the default prompt
# # 2. Text Chunks → Element Instances
# def extract_elements_from_chunks(chunks):
#     elements = []
#     for index, chunk in enumerate(chunks):
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Extract entities and relationships from the following text."},
#                 {"role": "user", "content": chunk}
#             ]
#         )
#         entities_and_relations = response.choices[0].message.content
#         elements.append(entities_and_relations)
#     return elements
# # TODO: Identify how we want our graph to look like. Experiment with the default prompt
# # 3. Element Instances → Element Summaries
# def summarize_elements(elements):
#     summaries = []
#     for index, element in enumerate(elements):
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Summarize the following entities and relationships in a structured format. Use \"->\" to represent relationships, after the \"Relationships:\" word."},
#                 {"role": "user", "content": element}
#             ]
#         )
#         summary = response.choices[0].message.content
#         summaries.append(summary)
#     return summaries

# # 4. Element Summaries → Graph Communities
# def build_graph_from_summaries(summaries):
#     G = nx.Graph()
#     for summary in summaries:
#         lines = summary.split("\n")
#         entities_section = False
#         relationships_section = False
#         entities = []
#         for line in lines:
#             if line.startswith("### Entities:") or line.startswith("**Entities:**"):
#                 entities_section = True
#                 relationships_section = False
#                 continue
#             elif line.startswith("### Relationships:") or line.startswith("**Relationships:**"):
#                 entities_section = False
#                 relationships_section = True
#                 continue
#             if entities_section and line.strip():
#                 entity = line.split(".", 1)[1].strip() if line[0].isdigit() and line[1] == "." else line.strip()
#                 entity = entity.replace("**", "")
#                 entities.append(entity)
#                 G.add_node(entity)
#             elif relationships_section and line.strip():
#                 parts = line.split("->")
#                 if len(parts) >= 2:
#                     source = parts[0].strip()
#                     target = parts[-1].strip()
#                     relation = " -> ".join(parts[1:-1]).strip()
#                     G.add_edge(source, target, label=relation)
#     return G

# def detect_communities(graph):
#     communities = []
#     for component in nx.connected_components(graph):
#         subgraph = graph.subgraph(component)
#         if len(subgraph.nodes) > 1:
#             try:
#                 sub_communities = algorithms.leiden(subgraph)
#                 for community in sub_communities.communities:
#                     communities.append(list(community))
#             except Exception as e:
#                 print(f"Error processing community: {e}")
#         else:
#             communities.append(list(subgraph.nodes))
#     return communities
# # 5. Graph Communities → Community Summaries
# def summarize_communities(communities, graph):
#     community_summaries = []
#     for index, community in enumerate(communities):
#         subgraph = graph.subgraph(community)
#         nodes = list(subgraph.nodes)
#         edges = list(subgraph.edges(data=True))
#         description = "Entities: " + ", ".join(nodes) + "\nRelationships: "
#         relationships = []
#         for edge in edges:
#             relationships.append(
#                 f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
#         description += ", ".join(relationships)

#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Summarize the following community of entities and relationships."},
#                 {"role": "user", "content": description}
#             ]
#         )
#         summary = response.choices[0].message.content.strip()
#         community_summaries.append(summary)
#     return community_summaries

# # 6. Community Summaries → Community Answers → Global Answer
# def generate_answers_from_communities(community_summaries, query):
#     intermediate_answers = []
#     for summary in community_summaries:
#         response = client.chat.completions.create(
#             model="gpt-4",
#             messages=[
#                 {"role": "system", "content": "Answer the following query based on the provided summary."},
#                 {"role": "user", "content": f"Query: {query} Summary: {summary}"}
#             ]
#         )
#         intermediate_answers.append(response.choices[0].message.content)

#     final_response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "Combine these answers into a final, concise response."},
#             {"role": "user", "content": f"Intermediate answers: {intermediate_answers}"}
#         ]
#     )
#     final_answer = final_response.choices[0].message.content
#     return final_answer
# # Main
# def graph_rag_pipeline(documents, query, chunk_size=600, overlap_size=100):
#     chunks = split_documents_into_chunks(documents, chunk_size, overlap_size)
#     elements = extract_elements_from_chunks(chunks)
#     summaries = summarize_elements(elements)
#     graph = build_graph_from_summaries(summaries)
#     communities = detect_communities(graph)
#     community_summaries = summarize_communities(communities)
#     final_answer = generate_answers_from_communities(community_summaries, query)
#     return final_answer

# # Example usage
# query = "What are the main themes in these documents?"
# answer = graph_rag_pipeline(DOCUMENTS, query)
# print('Answer:', answer)
