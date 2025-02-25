import networkx as nx 
import fitz 
import matplotlib.pyplot as plt
import networkx as nx
import json
import re
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from clients import client
from prompts import concepts
# Function to load and clean PDF text
from util import load_pdf, preprocess_text


# TODO: Add metadata to documents = {title, authors, publication date, citations, etc.}

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

def extract_entities(chunk):
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": f"{concepts.er_prompt()}"
            },
            {
                "role": "user",
                "content": chunk
            }
        ]
    )

    raw_output = response.choices[0].message.content.strip()
    return raw_output
    

def extract_json_from_response(raw_response: str) -> dict:
    """
    Extracts and formats JSON from an LLM response that may contain Markdown formatting.
    
    Args:
        raw_response (str): Raw string response from the LLM.
        
    Returns:
        dict: Parsed JSON data.
        
    Raises:
        ValueError: If no valid JSON is found in the response.
        json.JSONDecodeError: If the extracted text isn't valid JSON.
    """
    # First try to match JSON inside code blocks
    code_block_match = re.search(r'```(?:json)?\s*({\s*".*?}\s*)```', raw_response, re.DOTALL)
    
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # If no code block, try to find JSON-like structure directly
        json_match = re.search(r'({[\s\S]*?"novel_concepts"[\s\S]*?})', raw_response)
        if json_match:
            json_str = json_match.group(1)
        else:
            raise ValueError("No JSON structure found in response")

    # Clean up the extracted JSON string
    json_str = json_str.strip()
    
    # Remove any trailing commas before closing braces or brackets
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    try:
        # Parse the JSON
        data = json.loads(json_str)
        
        # Validate expected structure
        required_keys = {'novel_concepts', 'referenced_concepts', 'relationships'}
        if not all(key in data for key in required_keys):
            raise ValueError(f"JSON missing required keys: {required_keys - set(data.keys())}")
            
        return data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error at position {e.pos}: {e.msg}")
        print("Problematic JSON string:")
        print(json_str[max(0, e.pos-50):e.pos+50])  # Show context around error
        raise


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
    file_name='llama'
    file_path = 'data/'+file_name+'.pdf'
    text = load_pdf(file_path)  
    if text:
        chunks = split_documents_into_chunks(text)
        print(f"Number of chunks: {len(chunks)}")
        # print(f"First chunk: {chunks[0]}")
    else:
        print("Failed to load or clean the text.")
        exit()



    result = extract_entities(chunks[0])
    with open(f"Result/txt/{file_name}.txt", "a") as file:
        file.write(result)
    # with open(f"Result/txt/{file_name}.txt", "r") as file:
    #     result=file.read()
    data=extract_json_from_response(result)
    with open(f"Result/json/{file_name}.json", "w") as file:
        json.dump(data,file,indent=4)








