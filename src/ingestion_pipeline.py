import os
import uuid
from typing import List, Dict, Any
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from openai import OpenAI

class EmbeddingProvider:
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        print(f"WARN: Using Placeholder EmbeddingProvider.get_embeddings for {len(texts)} texts.")
        return [[0.0] * 1536] * len(texts)

class VectorDBProvider:
    def upsert(self, vectors: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        print(f"WARN: Using Placeholder VectorDBProvider.upsert for {len(vectors)} vectors.")
        pass

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str, dimension: int):
        print(f"Initializing OpenAIEmbeddingProvider with model: {model}")
        if not api_key:
            raise ValueError("OpenAI API key is required.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = dimension

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            processed_texts = [text.replace("\n", " ") for text in texts]
            response = self.client.embeddings.create(
                input=processed_texts,
                model=self.model
            )
            embeddings = [item.embedding for item in response.data]
            if embeddings and len(embeddings[0]) != self.dimension:
                print(f"Warning: OpenAI returned dimension {len(embeddings[0])}, expected {self.dimension}. Check model name and config.")
            return embeddings
        except openai.APIError as e:
            print(f"OpenAI API Error: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during OpenAI embedding generation: {e}")
            raise

class PineconeProvider(VectorDBProvider):
     def __init__(self, config):
        print("WARN: Using Placeholder PineconeProvider")
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
    def __init__(self):
        print("Initializing Ingestion Pipeline...")
        self.embedding_provider_name = os.getenv("EMBEDDING_MODEL_PROVIDER", "openai").lower()
        self.embedding_model = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002")
        self.model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", self.model_dimensions.get(self.embedding_model, 1536)))
        print(f"Using embedding dimension: {self.embedding_dimension}")
        self.vector_db_provider_name = "pinecone"
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        print(f"Chunk Size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        self.embedding_provider: EmbeddingProvider = self._load_embedding_provider()
        self.vector_db_provider: VectorDBProvider = self._load_vector_db_provider()

    def _load_embedding_provider(self) -> EmbeddingProvider:
        print(f"Loading Embedding Provider: {self.embedding_provider_name} (Model: {self.embedding_model})")
        if self.embedding_provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingProvider.")
            embedding_config = {
                "provider": "openai",
                "base_model": self.embedding_model,
                "base_dimension": self.embedding_dimension,
            }
            from types import SimpleNamespace
            config_obj = SimpleNamespace(**embedding_config)
            try:
                return OpenAIEmbeddingProvider(api_key=api_key, model=self.embedding_model, dimension=self.embedding_dimension)
            except Exception as e:
                 print(f"Error loading OpenAIEmbeddingProvider: {e}")
                 raise
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider_name}")

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

    def parse_file(self, file_path: str, additional_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        print(f"Parsing file with docling: {file_path}...")
        try:
            converter = DocumentConverter()
            result = converter.convert(file_path)
            extracted_text = result.document.export_to_markdown()
            if not extracted_text or not extracted_text.strip():
                 print(f"Warning: Docling did not extract text from {file_path}")
                 return []
            file_name = os.path.basename(file_path)
            base_metadata = {"source_document": file_name}
            if additional_metadata:
                base_metadata.update({k: v for k, v in additional_metadata.items() if v is not None})
            parsed_data = [{"text": extracted_text, "metadata": base_metadata}]
            print(f"Parsed {len(parsed_data)} element(s) with text using docling.")
            if additional_metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in additional_metadata.items() if v])
                if metadata_str:
                    print(f"Added base metadata: {metadata_str}")
            return parsed_data
        except ImportError:
             print("Error: 'docling' library not found. Please install it: pip install docling")
             raise
        except Exception as e:
            print(f"Error parsing file {file_path} with docling: {e}")
            import traceback
            traceback.print_exc()
            return []

    def chunk_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"Chunking {len(elements)} parsed elements...")
        texts_to_chunk = [element['text'] for element in elements]
        metadata_to_copy = [element['metadata'] for element in elements]
        if not texts_to_chunk:
            print("No text found in elements to chunk.")
            return []
        try:
            langchain_docs = self.text_splitter.create_documents(
                texts=texts_to_chunk,
                metadatas=metadata_to_copy
            )
            final_chunks = [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in langchain_docs
            ]
            print(f"Created {len(final_chunks)} chunks from {len(elements)} elements.")
            return final_chunks
        except Exception as e:
            print(f"Error during chunking: {e}")
            return []

    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        print(f"Generating embeddings for {len(chunks)} chunks...")
        texts_to_embed = [chunk['text'] for chunk in chunks]
        if not texts_to_embed:
            return []
        try:
            embeddings = self.embedding_provider.get_embeddings(texts_to_embed)
            print(f"Generated {len(embeddings)} embeddings.")
            if embeddings and len(embeddings[0]) != self.embedding_dimension:
                actual_dim = len(embeddings[0])
                print(f"Note: Actual embedding dimension ({actual_dim}) differs from configured dimension ({self.embedding_dimension}). Updating internal dimension.")
                self.embedding_dimension = actual_dim
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

    def store_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        print(f"Storing {len(chunks)} chunks with embeddings...")
        if not chunks or len(chunks) != len(embeddings):
            print("Warning: Mismatch between chunks and embeddings count or empty lists. Skipping storage.")
            return
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata_list = [chunk['metadata'] for chunk in chunks]
        try:
            self.vector_db_provider.upsert(vectors=embeddings, metadata=metadata_list, ids=ids)
        except Exception as e:
            print(f"Error storing chunks: {e}")

    def run(self, file_path: str, metadata: Dict[str, Any] = None):
        print(f"--- Starting Ingestion for {os.path.basename(file_path)} ---")
        parsed_elements = self.parse_file(file_path, additional_metadata=metadata)
        if not parsed_elements:
            print("No elements parsed. Exiting.")
            return
        chunks = self.chunk_elements(parsed_elements)
        if not chunks:
            print("No chunks created. Exiting.")
            return
        embeddings = self.generate_embeddings(chunks)
        if not embeddings:
            if chunks:
                 print("Embedding generation failed. Exiting.")
            else:
                 print("No embeddings generated (likely no text found). Exiting.")
            return
        self.store_chunks(chunks, embeddings)
        print(f"--- Finished Ingestion for {os.path.basename(file_path)} ---")