import os
import argparse
from dotenv import load_dotenv
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.ingestion_pipeline import IngestionPipeline

def get_valid_cfa_topics():
    core_topics = [
        "asset_allocation",
        "portfolio_construction",
        "performance_measurement",
        "derivatives_risk_management",
        "ethical_professional_standards",
        "alternative_investments",
    ]
    
    specialized_pathways = [
        "portfolio_management",
        "private_wealth",
        "private_markets",
    ]
    
    return core_topics, specialized_pathways

def main():
    
    load_dotenv()

    
    core_topics, specialized_pathways = get_valid_cfa_topics()
    all_topics = core_topics + specialized_pathways

    parser = argparse.ArgumentParser(description="Ingest documents into the vector database.")
    parser.add_argument("input_path", type=str, help="Path to the document file or directory to ingest.")
    parser.add_argument("--topic", type=str, choices=all_topics, 
                        help="CFA topic to associate with the document(s). One of: " + ", ".join(all_topics))
    parser.add_argument("--pathway", type=str, choices=specialized_pathways,
                        help="CFA specialized pathway (if applicable). One of: " + ", ".join(specialized_pathways))
    parser.add_argument("--auto-detect", action="store_true", 
                        help="Attempt to auto-detect CFA topic from filename or path.")
    args = parser.parse_args()

    input_path = args.input_path
    cfa_topic = args.topic
    cfa_pathway = args.pathway
    auto_detect = args.auto_detect

    
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    
    if auto_detect and not cfa_topic:
        print("Attempting to auto-detect CFA topic from path...")

        path_lower = input_path.lower()
        detected_topic = None
        for topic in all_topics:
            
            search_term = topic.replace('_', ' ')
            if search_term in path_lower:
                detected_topic = topic
                break       
        if detected_topic:
            cfa_topic = detected_topic
            print(f"Auto-detected topic: {cfa_topic}")
        else:
            print("Could not auto-detect topic. Please specify with --topic.")
            if not cfa_topic:  
                print("Available topics:")
                print("Core topics: " + ", ".join(core_topics))
                print("Specialized pathways: " + ", ".join(specialized_pathways))
                sys.exit(1)  
        if detected_topic in specialized_pathways:
            cfa_pathway = detected_topic
            print(f"Auto-detected pathway: {cfa_pathway}")
    if cfa_topic:
        print(f"Using CFA topic: {cfa_topic}")
    else:
        print("No CFA topic specified. Documents will be ingested without topic classification.")    
    if cfa_pathway:
        if cfa_pathway not in specialized_pathways:
            print(f"Warning: {cfa_pathway} is not a valid CFA pathway. Continuing anyway.")
        print(f"Using CFA pathway: {cfa_pathway}")
    try:
        pipeline = IngestionPipeline()
    except Exception as e:
        print(f"Error initializing Ingestion Pipeline: {e}")
        sys.exit(1)

    
    if os.path.isfile(input_path):
        print(f"Processing single file: {input_path}")
        
        metadata = {"cfa_topic": cfa_topic, "cfa_pathway": cfa_pathway}
        pipeline.run(input_path, metadata=metadata)
    elif os.path.isdir(input_path):
        print(f"Processing directory: {input_path}")
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path):
                print(f"\nProcessing file: {filename}")
                
                metadata = {"cfa_topic": cfa_topic, "cfa_pathway": cfa_pathway}
                pipeline.run(file_path, metadata=metadata)
            else:
                print(f"Skipping non-file item: {filename}")
    else:
        print(f"Error: Input path is neither a file nor a directory: {input_path}")
        sys.exit(1)

    print("\nIngestion process finished.")

if __name__ == "__main__":
    main() 