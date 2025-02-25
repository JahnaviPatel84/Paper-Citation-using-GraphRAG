import json
from neo4j import GraphDatabase
import logging
from typing import Dict, List, Any
from datetime import datetime

class ResearchConceptsGraphCreator:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize the graph creator with Neo4j connection details."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def close(self):
        """Close the Neo4j driver connection."""
        self.driver.close()

    def create_constraints(self):
        """Create necessary constraints in Neo4j."""
        with self.driver.session(database=self.database) as session:
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Component) REQUIRE c.name IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    self.logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    self.logger.error(f"Error creating constraint: {e}")
                    raise

    def create_paper_node(self, paper_info: Dict[str, str]):
        """Create a node for the research paper."""
        query = """
        MERGE (p:Paper {doi: $doi})
        ON CREATE SET 
            p.title = $title,
            p.authors = $authors,
            p.publication_date = $publication_date,
            p.created_at = datetime()
        ON MATCH SET 
            p.title = $title,
            p.authors = $authors,
            p.publication_date = $publication_date,
            p.updated_at = datetime()
        RETURN p
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, 
                    title=paper_info['title'],
                    doi=paper_info['doi'],
                    authors=paper_info['authors'],
                    publication_date=paper_info['publication_date']
                )
                self.logger.info(f"Created/Updated paper node: {paper_info['title']}")
                return result
            except Exception as e:
                self.logger.error(f"Error creating paper node: {e}")
                raise

    def connect_novel_concepts_to_paper(self):
        """Connect all Novel concept nodes to the Paper node."""
        query = """
        MATCH (c:Concept:Novel), (p:Paper)
        WHERE NOT (c)-[:DESCRIBED_IN]->(p)
        CREATE (c)-[:DESCRIBED_IN {created_at: datetime()}]->(p)
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                self.logger.info("Connected all novel concepts to paper node")
                return result
            except Exception as e:
                self.logger.error(f"Error connecting concepts to paper: {e}")
                raise

    def create_novel_concepts(self, concepts: List[Dict[str, Any]]):
        """Create nodes for novel concepts."""
        query = """
        UNWIND $concepts AS concept
        MERGE (c:Concept:Novel {name: concept.name})
        ON CREATE SET 
            c.type = concept.type,
            c.description = concept.description,
            c.key_innovations = concept.key_innovations,
            c.advantages = concept.advantages,
            c.validation_methods = concept.validation_methods,
            c.limitations = concept.limitations,
            c.created_at = datetime()
        ON MATCH SET 
            c.type = concept.type,
            c.description = concept.description,
            c.key_innovations = concept.key_innovations,
            c.advantages = concept.advantages,
            c.validation_methods = concept.validation_methods,
            c.limitations = concept.limitations,
            c.updated_at = datetime()
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, concepts=concepts)
                self.logger.info(f"Created/Updated {len(concepts)} novel concept nodes")
                return result
            except Exception as e:
                self.logger.error(f"Error creating novel concepts: {e}")
                raise

    def create_referenced_concepts(self, concepts: List[Dict[str, Any]]):
        """Create nodes for referenced concepts."""
        query = """
        UNWIND $concepts AS concept
        MERGE (c:Concept:Referenced {name: concept.name})
        ON CREATE SET 
            c.type = concept.type,
            c.description = concept.description,
            c.original_source = concept.original_source,
            c.usage_in_paper = concept.usage_in_paper,
            c.created_at = datetime()
        ON MATCH SET 
            c.type = concept.type,
            c.description = concept.description,
            c.original_source = concept.original_source,
            c.usage_in_paper = concept.usage_in_paper,
            c.updated_at = datetime()
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, concepts=concepts)
                self.logger.info(f"Created/Updated {len(concepts)} referenced concept nodes")
                return result
            except Exception as e:
                self.logger.error(f"Error creating referenced concepts: {e}")
                raise

    def create_relationships(self, relationships: List[Dict[str, Any]]):
        """Create relationships between concepts."""
        query = """
        UNWIND $relationships AS rel
        MATCH (source:Concept {name: rel.source_concept})
        MATCH (target:Concept {name: rel.target_concept})
        MERGE (source)-[r:RELATES_TO {type: rel.relationship_type}]->(target)
        ON CREATE SET 
            r.description = rel.description,
            r.technical_details = rel.technical_details,
            r.created_at = datetime()
        ON MATCH SET 
            r.description = rel.description,
            r.technical_details = rel.technical_details,
            r.updated_at = datetime()
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, relationships=relationships)
                self.logger.info(f"Created/Updated {len(relationships)} relationships")
                return result
            except Exception as e:
                self.logger.error(f"Error creating relationships: {e}")
                raise

    def create_components(self, concepts: List[Dict[str, Any]]):
        """Create component nodes and relationships."""
        query = """
        UNWIND $concepts AS concept
        MATCH (c:Concept {name: concept.name})
        UNWIND concept.components AS component_name
        MERGE (comp:Component {name: component_name})
        MERGE (c)-[r:HAS_COMPONENT]->(comp)
        ON CREATE SET r.created_at = datetime()
        ON MATCH SET r.updated_at = datetime()
        """
        
        concepts_with_components = [c for c in concepts if 'components' in c and c['components']]
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, concepts=concepts_with_components)
                self.logger.info(f"Created/Updated components for {len(concepts_with_components)} concepts")
                return result
            except Exception as e:
                self.logger.error(f"Error creating components: {e}")
                raise

    def create_graph_from_json(self, json_data: Dict[str, Any],paper_info: Dict[str, Any]):
        """Create the complete graph from JSON data."""
        try:
            # Create constraints first
            self.create_constraints()
            
            # Create paper node first
            self.create_paper_node(paper_info)
            
            # Create all nodes and relationships
            self.create_novel_concepts(json_data['novel_concepts'])
            self.create_referenced_concepts(json_data['referenced_concepts'])
            self.create_relationships(json_data['relationships'])
            self.create_components(json_data['novel_concepts'])
            
            # Connect all novel concepts to the paper
            self.connect_novel_concepts_to_paper()
            
            self.logger.info("Successfully created complete graph")
            
        except Exception as e:
            self.logger.error(f"Error creating graph: {e}")
            raise
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Retrieve all Paper nodes and their metadata from the graph.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing paper metadata
        """
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
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
                papers = [dict(record["paper"]) for record in result]
                self.logger.info(f"Retrieved {len(papers)} papers from the database")
                return papers
            except Exception as e:
                self.logger.error(f"Error retrieving papers: {e}")
                raise

def main():
    # Configuration
    NEO4J_URI = "neo4j://localhost:7687"  # Update with your Neo4j URI
    NEO4J_USER = "neo4j"                   # Update with your username
    NEO4J_PASSWORD = "mynewpassword"            # Update with your password
    NEO4J_DATABASE = "neo4j"               # Update with your database name

    # Read JSON data
    try:
        with open('Result/json/llama.json', 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Create graph
    creator = ResearchConceptsGraphCreator(
        NEO4J_URI, 
        NEO4J_USER, 
        NEO4J_PASSWORD, 
        NEO4J_DATABASE
    )
    paper_info = {
        "title": 'LLaMA: Open and Efficient Foundation Language Models',
        "doi": "10.18653/v1/2021.acl-long.1",
        "authors": ["Yang Liu", "Myle Ott", "Naman Goyal", "Jingfei Du", "Mandar Joshi", "Danqi Chen", "Omer Levy", "Mike Lewis", "Luke Zettlemoyer", "Veselin Stoyanov"],
        "publication_date": datetime(2021, 8, 1)
    }

    try:
        creator.create_graph_from_json(json_data,paper_info)
        print("Graph created successfully!")
    except Exception as e:
        print(f"Error creating graph: {e}")
    finally:
        creator.close()

if __name__ == "__main__":
    main()