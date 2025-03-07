import pandas as pd
import numpy as np
from datetime import datetime

# Load dataset
file_path = "dblp_v12_50000_papers.csv"
df = pd.read_csv(file_path)

def get_paper_info(paper_id, json_format=True):
    # Find the paper by ID
    paper_row = df[df['id'] == paper_id]
    
    if paper_row.empty:
        return f"No paper found with ID {paper_id}"

    # Extract paper details
    paper = paper_row.iloc[0]
    title = paper['title']
    doi = paper['doi'] if pd.notna(paper['doi']) else "N/A"
    author_names = paper['author_names'].split(',') if pd.notna(paper['author_names']) else []
    author_ids = paper['author_ids'].split(',') if pd.notna(paper['author_ids']) else []
    references = paper['references'].split(',') if pd.notna(paper['references']) else []
    year = int(paper['year']) if pd.notna(paper['year']) else None
    publication_date = datetime(year, 1, 1) if year else None  # Assuming only year is available
    conference = paper['venue_name'] if pd.notna(paper['venue_name']) else "N/A"
    fos = paper['fos'].split(',') if pd.notna(paper['fos']) else []

    # Can edit these to include relevant metadata
    result = {
        "id": int(paper['id']),
        "title": title,
        "doi": doi,
        "references": references,
        "authors": author_names,
        "author_ids": author_ids,
        "publication_date": publication_date,
        "conference": conference,
        "fos": fos,
    }
    
    if json_format:
        return result
    else:
        # Return as a DataFrame row if json_format False
        return paper_row
    


# Usage:
# paper_id = 45128760
# output = get_paper_info(paper_id, json_format=True)