import os
import uuid
import sys # Added for temporary exit
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured_client.models.errors import SDKError
from openai import OpenAI, APIError

class VectorDBProvider:
    def upsert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        print(f"WARN: Using Placeholder VectorDBProvider.upsert for {len(vectors)} vectors.")
        pass

class PineconeProvider(VectorDBProvider):
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int = None, metric: str = "cosine", create_if_not_exists: bool = True, cloud: str = "aws", region: str = "us-east-1"):
        print(f"Initializing PineconeProvider for index '{index_name}' in env '{environment}'...")
        if not api_key or not environment or not index_name:
            raise ValueError("Pinecone API key, environment, and index name are required.")
        try:
            print("DEBUG: Attempting Pinecone(api_key=...)")
            self.pc = Pinecone(api_key=api_key)
            print(f"DEBUG: Pinecone object created: {type(self.pc)}")
            print("DEBUG: Attempting self.pc.list_indexes()")
            index_list_obj = self.pc.list_indexes()
            print(f"DEBUG: list_indexes() called, type: {type(index_list_obj)}")
            index_list = index_list_obj.names()
            print(f"DEBUG: Index names: {index_list}")
            if index_name not in index_list:
                print(f"DEBUG: Index '{index_name}' not found, attempting creation...")
                if create_if_not_exists and dimension:
                    print(f"Pinecone index '{index_name}' not found. Creating with dimension {dimension}...")
                    print(f"DEBUG: Attempting self.pc.create_index(...) for {index_name}")
                    self.pc.create_index(
                        name=index_name,
                        dimension=dimension,
                        metric=metric,
                        spec=ServerlessSpec(
                            cloud=cloud,
                            region=region
                        )
                    )
                    print(f"DEBUG: Pinecone index '{index_name}' created successfully.")
                else:
                    error_msg = f"Pinecone index '{index_name}' does not exist."
                    if create_if_not_exists:
                        error_msg += " Cannot create index without dimension specified in provider config."
                    print(f"Warning: {error_msg}")
                    raise ValueError(error_msg)
            else:
                 print(f"DEBUG: Pinecone index '{index_name}' already exists.")
            print(f"DEBUG: Attempting self.pc.Index('{index_name}')")
            self.index = self.pc.Index(index_name)
            print(f"DEBUG: Index object obtained: {type(self.index)}")
            print("DEBUG: Attempting self.index.describe_index_stats()")
            stats = self.index.describe_index_stats()
            print(f"DEBUG: describe_index_stats() called, type: {type(stats)}")
            print(f"Index stats: Dimension: {stats.dimension}, Vector count: {stats.total_vector_count}")
        except Exception as e:
            print(f"ERROR CAUGHT in PineconeProvider.__init__: {e}")
            import traceback
            traceback.print_exc()
            raise

    def upsert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        if not self.index:
             print("Error: Pinecone index is not initialized.")
             return
        if not vectors or len(vectors) != len(metadata) or len(vectors) != len(ids):
            print("Error: Vectors, metadata, and IDs must be non-empty and have the same length.")
            return
        vectors_to_upsert = []
        for i, vec in enumerate(vectors):
            clean_meta = {}
            for k, v in metadata[i].items():
                if isinstance(v, (str, bool, float, int)):
                    clean_meta[k] = v
                elif isinstance(v, list):
                    if all(isinstance(item, str) for item in v):
                        clean_meta[k] = v
                    else:
                        print(f"Warning: Metadata key '{k}' contains a list with non-string items for ID {ids[i]}. Skipping this key.")
            vectors_to_upsert.append((ids[i], vec, clean_meta))
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone index...")
        try:
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                 batch = vectors_to_upsert[i:i + batch_size]
                 upsert_response = self.index.upsert(vectors=batch)
                 print(f"  Upserted batch {i//batch_size + 1}, response: {upsert_response}")
            print(f"Pinecone upsert completed for {len(vectors_to_upsert)} vectors.")
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")

load_dotenv()

class IngestionPipeline:
    def __init__(self, 
                 chunk_size_override: Optional[int] = None, 
                 chunk_overlap_override: Optional[int] = None,
                 chunking_strategy_override: Optional[str] = None,
                 embedding_provider: str = "openai",
                 embedding_model_name: Optional[str] = None):
        print("Initializing Ingestion Pipeline...")
        self.embedding_provider_name = embedding_provider.lower()
        self.embedding_model = embedding_model_name or self._get_default_embedding_model(self.embedding_provider_name)
        
        self.unstructured_api_key = os.getenv("UNSTRUCTURED_API_KEY")
        if not self.unstructured_api_key:
            raise ValueError("UNSTRUCTURED_API_KEY environment variable not set.")
        self.unstructured_client = UnstructuredClient(api_key_auth=self.unstructured_api_key)

        self.openai_client = None
        if self.embedding_provider_name == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set for 'openai' embedding provider.")
            self.openai_client = OpenAI(api_key=openai_api_key)
            print("OpenAI client initialized.")
        # TODO : Add similar blocks for other providers in future for perfomacne
        
        self.embedding_dimension = self._get_embedding_dimension(self.embedding_model)
        print(f"Using Embedding Provider (client-side): {self.embedding_provider_name}, Model: {self.embedding_model}, Dimension: {self.embedding_dimension}")
        
        self.vector_db_provider_name = "pinecone"
        self.chunk_size = chunk_size_override if chunk_size_override is not None else int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = chunk_overlap_override if chunk_overlap_override is not None else int(os.getenv("CHUNK_OVERLAP", 100))
        self.chunking_strategy = chunking_strategy_override if chunking_strategy_override is not None else os.getenv("CHUNKING_STRATEGY", "basic")
        print(f"Chunk Size: {self.chunk_size}, Overlap: {self.chunk_overlap}, Chunking Strategy: {self.chunking_strategy}")
        
        self.vector_db_provider: VectorDBProvider = self._load_vector_db_provider()

    def _get_default_embedding_model(self, provider_name: str) -> str:
        if provider_name == "openai":
            return "text-embedding-3-large"
        else:
            raise ValueError(f"No default embedding model specified for provider: {provider_name}")

    def _get_embedding_dimension(self, model_name: str) -> int:
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        dimension = model_dimensions.get(model_name)
        if dimension is None:
            print(f"Warning: Unknown embedding model '{model_name}'. Attempting to get dimension from environment variable EMBEDDING_DIMENSION or falling back to 1536.")
            return int(os.getenv("EMBEDDING_DIMENSION", 1536))
        return dimension

    def _load_vector_db_provider(self) -> VectorDBProvider:
        print(f"Loading Vector DB Provider: {self.vector_db_provider_name}")
        if self.vector_db_provider_name == "pinecone":
            api_key = os.getenv("PINECONE_API_KEY")
            environment = os.getenv("PINECONE_ENVIRONMENT")
            index_name = os.getenv("PINECONE_INDEX_NAME")
            create_index = os.getenv("PINECONE_CREATE_INDEX", "True").lower() == "true"
            pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
            pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
            if not api_key or not environment or not index_name:
                raise ValueError("PINECONE_API_KEY, PINECONE_ENVIRONMENT, and PINECONE_INDEX_NAME are required.")
            try:
                if not self.embedding_dimension:
                     raise ValueError("Embedding dimension must be determined before initializing PineconeProvider.")
                return PineconeProvider(
                    api_key=api_key,
                    environment=environment,
                    index_name=index_name,
                    dimension=self.embedding_dimension,
                    create_if_not_exists=create_index,
                    cloud=pinecone_cloud,
                    region=pinecone_region
                )
            except Exception as e:
                print(f"Error loading PineconeProvider: {e}")
                raise
        else:
            raise ValueError(f"Unsupported vector DB provider: {self.vector_db_provider_name}")

    def parse_file(self, file_path: str, additional_metadata: Dict[str, Any] = None, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Parses a local file using the Unstructured API partition endpoint,
        then generates embeddings client-side.

        Args:
            file_path: Path to the local file.
            additional_metadata: Optional metadata to add to each element.
            strategy: Optional partitioning strategy (e.g., 'hi_res', 'fast').

        Returns:
            A list of dictionaries, each representing a structured element
            with 'text', 'metadata', and 'embeddings' keys.
        """
        print(f"Parsing file with Unstructured API: {file_path}" + (f" using strategy '{strategy}'" if strategy else ""))
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return []
        if additional_metadata is None:
            additional_metadata = {}
        base_metadata = {
            "source": os.path.basename(file_path),
            "file_path": file_path,
            **additional_metadata
        }
        
        raw_elements = [] 

        try:
            with open(file_path, "rb") as f:
                files_arg = shared.Files(
                    content=f.read(),
                    file_name=os.path.basename(file_path)
                )
                partition_args = {
                    "files": files_arg,
                    "chunking_strategy": self.chunking_strategy,
                    # "embedding_provider": self.embedding_provider_name,
                    # "embedding_model_name": self.embedding_model,      
                    # "embedding_api_key": self.embedding_api_key       
                }
                if self.chunk_size: 
                    partition_args["max_characters"] = self.chunk_size
                    partition_args["overlap"] = self.chunk_overlap

                partition_args = {k: v for k, v in partition_args.items() if v is not None}
                
                if strategy:
                    partition_args["strategy"] = strategy

                log_args = partition_args.copy()
                if 'files' in log_args and hasattr(log_args['files'], 'file_name'):
                     log_args['files'] = f"<file content for {log_args['files'].file_name}>"
                else:
                     log_args['files'] = "<binary file content>"

                print(f"Sending {os.path.basename(file_path)} to Unstructured partition endpoint with config: {log_args}...")

                req = operations.PartitionRequest(
                    partition_parameters=shared.PartitionParameters(**partition_args)
                )
                response = self.unstructured_client.general.partition(request=req)

            if response.elements:
                print(f"Received {len(response.elements)} elements from Unstructured.")
                for element_dict in response.elements:
                    element_text = element_dict.get('text', '')
                    element_type = element_dict.get('type', 'Unknown')
                    element_metadata = element_dict.get('metadata', {})
                    combined_metadata = {
                        **base_metadata,
                        **element_metadata,
                        "element_type": element_type,
                    }
                    cleaned_metadata = {k: v for k, v in combined_metadata.items() if v is not None}
                    raw_elements.append({ 
                        "text": element_text,
                        "metadata": cleaned_metadata,
                    })
            else:
                print(f"Warning: Unstructured API returned no elements for {file_path}")
                return []

        except SDKError as e:
            print(f"Unstructured SDK Error processing {file_path}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error processing {file_path} with Unstructured: {e}")
            import traceback
            traceback.print_exc()
            return []

        if not raw_elements:
            return []

        final_elements_with_embeddings = []
        texts_to_embed = [el["text"] for el in raw_elements if el.get("text")]

        if not texts_to_embed:
            print(f"No text content found in elements from {file_path} to embed.")
            for el in raw_elements:
                el["embeddings"] = None
                final_elements_with_embeddings.append(el)
            return final_elements_with_embeddings

        generated_embeddings = []
        if self.embedding_provider_name == "openai" and self.openai_client:
            print(f"Generating embeddings for {len(texts_to_embed)} text chunks using OpenAI model {self.embedding_model}...")
            openai_batch_size = 100 
            for i in range(0, len(texts_to_embed), openai_batch_size):
                batch_texts = texts_to_embed[i:i + openai_batch_size]
                try:
                    res = self.openai_client.embeddings.create(input=batch_texts, model=self.embedding_model)
                    generated_embeddings.extend([item.embedding for item in res.data])
                    print(f"  Generated embeddings for batch {i//openai_batch_size + 1}/{ (len(texts_to_embed) -1)//openai_batch_size + 1 }")
                except APIError as e:
                    print(f"OpenAI API Error during embedding batch {i//openai_batch_size + 1}: {e}")
                    generated_embeddings.extend([None] * len(batch_texts))
                except Exception as e:
                    print(f"Unexpected error during OpenAI embedding batch {i//openai_batch_size + 1}: {e}")
                    generated_embeddings.extend([None] * len(batch_texts))
        else:
            print(f"Warning: Embedding provider '{self.embedding_provider_name}' is not configured for client-side embeddings or client not initialized. Skipping embedding.")
            generated_embeddings = [None] * len(texts_to_embed)
        
        embedding_idx = 0
        for raw_el in raw_elements:
            final_element = raw_el.copy()
            if raw_el.get("text") and embedding_idx < len(generated_embeddings):
                final_element["embeddings"] = generated_embeddings[embedding_idx]
                embedding_idx += 1
            else:
                final_element["embeddings"] = None 
            final_elements_with_embeddings.append(final_element)
            
        if embedding_idx != len(generated_embeddings) and texts_to_embed: 
             print(f"Warning: Mismatch in number of embeddings generated ({len(generated_embeddings)}) and text chunks that required embedding ({embedding_idx}).")

        return final_elements_with_embeddings

    def store_chunks(self, chunks_with_embeddings: List[Dict[str, Any]]):
        """Stores chunks and their embeddings (extracted from the chunk dict) in the vector database."""
        if not chunks_with_embeddings:
            print("Error: No chunks provided for storage. Skipping storage.")
            return
        embeddings = []
        metadata_list = []
        ids = []
        successful_extractions = 0

        print(f"Preparing {len(chunks_with_embeddings)} chunks (with embeddings) for storage...")
        for chunk in chunks_with_embeddings:
            embedding_vector = chunk.get("embeddings")
            if embedding_vector is None:
                print(f"Warning: Skipping chunk because it lacks embeddings. Metadata: {chunk.get('metadata', {})}")
                continue 
            chunk_meta = chunk.get("metadata", {}).copy()
            chunk_meta["text"] = chunk.get("text", "") 
            embeddings.append(embedding_vector)
            metadata_list.append(chunk_meta)
            ids.append(str(uuid.uuid4())) 
            successful_extractions += 1

        if not successful_extractions:
            print("Error: No chunks with embeddings were found to store.")
            return
            
        print(f"Extracted embeddings and metadata for {successful_extractions} chunks.")

        try:
            self.vector_db_provider.upsert(vectors=embeddings, metadata=metadata_list, ids=ids)
            print(f"Successfully stored {len(ids)} chunks.")

            # ---- DEBUG CODE  ----
            # if ids and hasattr(self.vector_db_provider, 'index') and self.vector_db_provider.index is not None:
            #     print(f"\n--- DEBUG: Fetching vector with ID: {ids[0]} to inspect its metadata ---")
            #     try:
            #         fetch_response = self.vector_db_provider.index.fetch(ids=[ids[0]])
            #         print(f"DEBUG: Fetch response: {fetch_response}")
            #         if fetch_response and fetch_response.vectors:
            #             for vector_id, vector_data in fetch_response.vectors.items():
            #                 print(f"  Vector ID: {vector_id}")
            #                 print(f"  Metadata: {vector_data.metadata}")
            #                 # print(f"  Values: {vector_data.values}") # Optional: to see the vector values
            #         else:
            #             print("DEBUG: Vector not found or empty response.")
            #     except Exception as e:
            #         print(f"DEBUG: Error fetching vector by ID: {e}")
            #     print("--- DEBUG: Halting after first successful store_chunks for inspection. REMOVE THIS FOR NORMAL OPERATION. ---")
            #     sys.exit(0) # Or raise Exception("Debug halt")
            # ---- DEBUG CODE END ----

        except Exception as e:
            print(f"Error storing chunks in vector DB: {e}")

    def run(self, file_path: str, metadata: Dict[str, Any] = None, strategy: Optional[str] = None):
        """Runs the full ingestion pipeline for a single file."""
        print(f"\n----- Starting pipeline for: {file_path} -----")
        if metadata:
            print(f"Using provided metadata: {metadata}")

        
        elements_with_embeddings = self.parse_file(file_path, additional_metadata=metadata, strategy=strategy)
        if not elements_with_embeddings:
            print(f"Pipeline halted for {file_path}: No elements parsed/returned by API.")
            return
        
        if elements_with_embeddings[0].get("embeddings") is None:
            print(f"Warning: Unstructured API did not return embeddings for the first element of {file_path}. Check API configuration and subscription. Skipping storage.")
            return
        
        chunks_with_embeddings = elements_with_embeddings
        self.store_chunks(chunks_with_embeddings)
        print(f"----- Finished pipeline for: {file_path} -----")