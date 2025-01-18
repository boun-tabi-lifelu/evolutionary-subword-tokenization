import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/cta/share/users/esm"

from time import time
import sqlite3
import pandas as pd
import torch
from tqdm import tqdm
from protein_embedding_database import ProteinEmbeddingDatabase

print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.current_device(), torch.cuda.get_device_name(0))


# Connect to DB
db_file = "/cta/share/users/uniprot/human/human.db"
conn = sqlite3.connect(db_file)

df_uniref50 = pd.read_sql(f"""SELECT Entry as uniprot_id, Sequence as sequence
                          FROM proteins
                          WHERE Entry IN (SELECT uniprot_accession FROM uniref50_distilled)""", conn)
df_uniref50 = df_uniref50[df_uniref50['sequence'].str.len() < 3000].reset_index(drop=True)

df_uniref90 = pd.read_sql(f"""SELECT Entry as uniprot_id, Sequence as sequence
                          FROM proteins
                          WHERE Entry IN (SELECT uniprot_accession FROM uniref90_distilled)""", conn)
df_uniref90 = df_uniref90[df_uniref90['sequence'].str.len() < 3000].reset_index(drop=True)

conn.close()

proteins = {}
for idx, row in pd.concat([df_uniref50, df_uniref90]).drop_duplicates().iterrows():
    proteins[row['uniprot_id']] = row['sequence']

print(len(proteins))

# proteins = {
#     "P12345": "MKWVTFISLLLLFSSAYS",
#     "P67890": "MLCAGRRRFAUT",
#     "P678909": "MLCAGRRRFAUT",
#     "P6782290": "MLCAGACACACRRRFAUT"
# }

t0 = time()
model_name = "facebook/esm2_t33_650M_UR50D"

print(model_name)

# Initialize the database
db = ProteinEmbeddingDatabase(model_name = model_name)

# Add proteins to the database
for uniprot_id, sequence in tqdm(proteins.items()):
    db.add_protein(uniprot_id, sequence)

# Save the database
root_path = "/cta/share/users/uniprot/human/faiss"
faiss_path = f"{root_path}/{model_name.replace('/', '_')}_protein_embeddings.faiss"
id_map_path = f"{root_path}/{model_name.replace('/', '_')}_id_mapping.csv"
db.save_database(faiss_path, id_map_path)
print(time()-t0)