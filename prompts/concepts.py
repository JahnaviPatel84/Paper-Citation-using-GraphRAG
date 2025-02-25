from neo4j import GraphDatabase
from typing import List, Dict, Any
from datetime import datetime
from graph_util import create_neo4j_connection, get_papers_from_db

def er_prompt():
    """
    Generate a prompt for the Entity Relationships task.
    
    Returns:
        str: Prompt for the Entity Relationships task
    """
    driver = None
    try:
        # Create connection
        driver, database = create_neo4j_connection()
        
        # Get papers from the database
        papers = get_papers_from_db(driver, database)
        
        # Print the results
        print("\nPaper details:")
        for paper in papers:
            print("\n" + "="*50)
            print(f"Title: {paper['title']}")
            print(f"DOI: {paper['doi']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Publication Date: {paper['publication_date']}")
            print(f"Novel Concepts: {', '.join(paper['novel_concepts'])}")
            print(f"Created At: {paper['created_at']}")
            if paper.get('updated_at'):
                print(f"Last Updated: {paper['updated_at']}")
            
    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        if driver:
            driver.close()
            print("\nDatabase connection closed")
    
    papers_info = f"Some existing ideas are: {papers}" if papers else ""
    prompt = f"""
    -Goal-
    Extract and classify concepts from a research paper, distinguishing between novel concepts introduced by the paper and existing concepts it utilizes. Generate a structured JSON output that captures the relationships and dependencies between these concepts.

    -Steps-

    1. Identify concepts in the research paper and classify them into two categories:
    - Novel Concepts: New ideas, methods, architectures, or frameworks introduced by this paper. Must include: Name, description, key characteristics, and claimed improvements/benefits. Example: A new neural architecture, optimization technique, or theoretical framework
    - Referenced Concepts: Existing ideas, methods, or techniques that the paper builds upon. Must include: Name, description, and how it's utilized in the current work. Example: Existing architectures, standard optimization methods, or established theoretical foundations. {papers_info}
    
    2. For each novel concept, extract: 
    - Formal definition or description
    - Key innovations or distinctions from existing work
    - Claimed advantages or improvements
    - Technical components or sub-parts
    - Limitations or constraints
    - Experimental validation methods

    3. For each concept relationship, identify:
    - How novel concepts build upon or modify referenced concepts
    - Dependencies between concepts
    - Comparative advantages or trade-offs
    - Implementation details or requirements

    #############
    Output Format
    #############
    {{
        "novel_concepts": [
            {{
                "name": "string",
                "type": "string (e.g., architecture, method, framework)",
                "description": "string",
                "key_innovations": ["string"],
                "advantages": ["string"],
                "components": ["string"],
                "limitations": ["string"],
                "validation_methods": ["string"]
            }}
        ],
        "referenced_concepts": [
            {{
                "name": "string",
                "type": "string",
                "description": "string",
                "original_source": "string",
                "usage_in_paper": "string"
            }}
        ],
        "relationships": [
            {{
                "source_concept": "string",
                "target_concept": "string",
                "relationship_type": "string (e.g., extends, depends_on, improves)",
                "description": "string",
                "technical_details": "string"
            }}
        ]
    }}

    ########
    Example
    ########
    Input text:
    We introduce TransformerX, a novel attention mechanism that extends the original Transformer architecture. Unlike standard self-attention which has O(n²) complexity, TransformerX achieves O(n log n) complexity by introducing hierarchical token clustering. Our method builds upon the original attention mechanism but adds a preliminary clustering step that groups similar tokens before computing attention scores.

    Expected output:
    {{
        "novel_concepts": [
            {{
                "name": "TransformerX",
                "type": "attention_mechanism",
                "description": "Hierarchical attention mechanism using token clustering",
                "key_innovations": [
                    "Hierarchical token clustering before attention computation",
                    "Reduced computational complexity from O(n²) to O(n log n)"
                ],
                "advantages": [
                    "Lower computational complexity",
                    "Maintains attention mechanism benefits"
                ],
                "components": [
                    "Token clustering module",
                    "Modified attention computation"
                ],
                "limitations": [],
                "validation_methods": []
            }}
        ],
        "referenced_concepts": [
            {{
                "name": "Transformer",
                "type": "architecture",
                "description": "Original Transformer architecture with self-attention mechanism",
                "original_source": "Attention Is All You Need (Vaswani et al.)",
                "usage_in_paper": "Base architecture that is extended"
            }},
            {{
                "name": "Self-attention",
                "type": "mechanism",
                "description": "Attention mechanism for relating different positions in a sequence",
                "original_source": "Attention Is All You Need (Vaswani et al.)",
                "usage_in_paper": "Referenced as baseline with O(n²) complexity"
            }}
        ],
        "relationships": [
            {{
                "source_concept": "TransformerX",
                "target_concept": "Transformer",
                "relationship_type": "extends",
                "description": "Extends original Transformer by adding hierarchical token clustering",
                "technical_details": "Modifies attention computation to reduce complexity"
            }}
        ]
    }}

    ########
    Notes:
    ########
    1. Focus on clearly distinguishing between truly novel contributions and existing concepts
    2. Extract specific technical details and implementation requirements
    3. Maintain clear traceability between novel concepts and their building blocks
    4. Include quantitative improvements or metrics when available
    5. Note any limitations or constraints mentioned for novel concepts
    """
    
    return prompt