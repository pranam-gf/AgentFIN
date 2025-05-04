import os
import argparse
from dotenv import load_dotenv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

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

def process_single_file(pipeline: IngestionPipeline, file_path: str, metadata: dict, strategy: Optional[str]):
    """Helper function to process one file, suitable for parallel execution."""
    start_time = time.time()
    print(f"Starting processing for: {os.path.basename(file_path)}")
    try:
        pipeline.run(file_path, metadata=metadata, strategy=strategy)
        duration = time.time() - start_time
        print(f"Finished processing {os.path.basename(file_path)} in {duration:.2f} seconds")
        return True, file_path
    except Exception as e:
        duration = time.time() - start_time
        print(f"!!! Error processing {os.path.basename(file_path)} after {duration:.2f} seconds: {e}")
        return False, file_path

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
    parser.add_argument("--strategy", type=str, default=None, choices=['fast', 'hi_res', 'ocr_only'],
                        help="Unstructured parsing strategy to use.")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Override chunk size from .env file.")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="Override chunk overlap from .env file.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for processing files.")
    parser.add_argument("--chunking-strategy", type=str, default="basic", 
                        choices=["basic", "by_title", "by_page", "by_similarity"],
                        help="Unstructured chunking strategy to use (default: basic).")
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
        pipeline = IngestionPipeline(
            chunk_size_override=args.chunk_size,
            chunk_overlap_override=args.chunk_overlap,
            chunking_strategy_override=args.chunking_strategy
        )
    except Exception as e:
        print(f"Error initializing Ingestion Pipeline: {e}")
        sys.exit(1)

    files_to_process = []
    if os.path.isfile(input_path):
        files_to_process.append(input_path)
    elif os.path.isdir(input_path):
        print(f"Scanning directory: {input_path}")
        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path):
                 if filename.lower().endswith(('.pdf', '.txt', '.md', '.docx', '.pptx')):
                     files_to_process.append(file_path)
                 else:
                     print(f"Skipping non-document file: {filename}")
            else:
                print(f"Skipping non-file item: {filename}")
    else:
        print(f"Error: Input path is neither a file nor a directory: {input_path}")
        sys.exit(1)

    if not files_to_process:
        print("No files found to process.")
        sys.exit(0)

    print(f"Found {len(files_to_process)} files to process with {args.workers} workers.")

    success_count = 0
    error_count = 0
    total_start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        for file_path in files_to_process:
            metadata = {"cfa_topic": cfa_topic, "cfa_pathway": cfa_pathway}
            futures.append(executor.submit(process_single_file, pipeline, file_path, metadata, args.strategy))

        for future in as_completed(futures):
            try:
                success, processed_file_path = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
            except Exception as exc:
                print(f'!!! Generated an exception processing a file: {exc}')
                error_count += 1

    total_duration = time.time() - total_start_time
    print("\n--- Ingestion Summary ---")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {error_count} files")
    print("Ingestion process finished.")

if __name__ == "__main__":
    main() 