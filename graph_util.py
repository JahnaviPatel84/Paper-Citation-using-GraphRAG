from neo4j import GraphDatabase
from typing import List, Dict, Any
from datetime import datetime

def create_neo4j_connection(uri: str = "neo4j://localhost:7687", 
                          user: str = "neo4j", 
                          password: str = "mynewpassword", 
                          database: str = "neo4j"):
    """
    Create a connection to Neo4j database.
    
    Args:
        uri (str): Neo4j connection URI
        user (str): Neo4j username
        password (str): Neo4j password
        database (str): Neo4j database name
    
    Returns:
        tuple: (driver, database_name) - Neo4j driver instance and database name
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # Test the connection
        driver.verify_connectivity()
        print("Successfully connected to Neo4j database")
        return driver, database
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        raise

def get_papers_from_db(driver, database: str) -> List[Dict[str, Any]]:
    """
    Retrieve all papers and their metadata from Neo4j database.
    
    Args:
        driver: Neo4j driver instance
        database (str): Neo4j database name
    
    Returns:
        List[Dict[str, Any]]: List of papers with their metadata
    """
    try:
        with driver.session(database=database) as session:
            query = """
            MATCH (p:Paper)
            RETURN {
                title: p.title,
                doi: p.doi,
                authors: p.authors,
                publication_date: p.publication_date,
                created_at: p.created_at,
                updated_at: p.updated_at,
                novel_concepts: [(p)<-[:DESCRIBED_IN]-(c:Concept:Novel) | c.name]
            } as paper
            ORDER BY p.publication_date DESC
            """
            
            result = session.run(query)
            papers = [dict(record["paper"]) for record in result]
            print(f"Successfully retrieved {len(papers)} papers")
            return papers
            
    except Exception as e:
        print(f"Error executing query: {e}")
        raise