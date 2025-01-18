import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class ProteinEmbeddingDatabase:
    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        
        # Initialize FAISS index
        # ESM-2 t6 8M model has embedding dimension of 320
        self.dimension = self.model.embeddings.word_embeddings.embedding_dim
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Dictionary to store mapping between FAISS indices and amino acid IDs
        self.id_mapping: Dict[int, str] = {}
        # Counter for generating unique IDs
        self.current_index = 0
        
    def generate_amino_acid_id(self, uniprot_id: str, position: int) -> str:
        """Generate a unique ID for each amino acid."""
        return f"{uniprot_id}_{position}"
    
    def get_embeddings(self, sequence: str) -> torch.Tensor:
        """Get embeddings for a protein sequence."""
        inputs = self.tokenizer(sequence, return_tensors="pt")
        # inputs = self.tokenizer(sequence, return_tensors="pt", truncation=True, padding=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get per-token embeddings (excluding special tokens)
            embeddings = outputs.last_hidden_state[:, 1:-1, :]  # Remove <cls> and <eos>
        
        return embeddings.cpu()
    
    def add_protein(self, uniprot_id: str, sequence: str):
        """Add a protein sequence to the database."""
        # Get embeddings
        embeddings = self.get_embeddings(sequence)
        embeddings_np = embeddings.numpy().reshape(-1, self.dimension)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings_np)
        
        # Store mapping between FAISS indices and amino acid IDs
        for i in range(len(sequence)):
            amino_acid_id = self.generate_amino_acid_id(uniprot_id, i)
            self.id_mapping[self.current_index] = amino_acid_id
            self.current_index += 1
    
    def search_similar_amino_acids(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar amino acids given a query embedding."""
        # Ensure query embedding has correct shape
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get corresponding amino acid IDs
        results = [(self.id_mapping[int(idx)], float(dist)) 
                  for idx, dist in zip(indices[0], distances[0])]
        
        return results
    
    def get_amino_acid_embedding(self, amino_acid_id: str) -> np.ndarray:
        """Retrieve embedding for a specific amino acid ID."""
        # Find the FAISS index for the amino acid ID
        index = None
        for idx, aid in self.id_mapping.items():
            if aid == amino_acid_id:
                index = idx
                break
        
        if index is None:
            raise ValueError(f"Amino acid ID {amino_acid_id} not found in database")
        
        # Reconstruct embedding from FAISS index
        embedding = self.index.reconstruct(index)
        return embedding
    
    def save_database(self, faiss_path: str, mapping_path: str):
        """Save the FAISS index and ID mapping."""
        faiss.write_index(self.index, faiss_path)
        
        # Save ID mapping as CSV
        mapping_df = pd.DataFrame(list(self.id_mapping.items()), 
                                columns=['faiss_index', 'amino_acid_id'])
        mapping_df.to_csv(mapping_path, index=False)
    
    @classmethod
    def load_database(cls, faiss_path: str, mapping_path: str, 
                     model_name: str = "facebook/esm2_t6_8M_UR50D") -> 'ProteinEmbeddingDatabase':
        """Load a saved database."""
        instance = cls(model_name=model_name)
        
        # Load FAISS index
        instance.index = faiss.read_index(faiss_path)
        
        # Load ID mapping
        mapping_df = pd.read_csv(mapping_path)
        instance.id_mapping = dict(zip(mapping_df.faiss_index, mapping_df.amino_acid_id))
        instance.current_index = max(instance.id_mapping.keys()) + 1
        
        return instance