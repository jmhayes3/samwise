from pinecone_datasets import load_dataset


dataset = load_dataset("squad-text-embedding-ada-002")
dataset.head()
dataset.documents.drop(['sparse_values', 'blob'], axis=1, inplace=True)
print(dataset.head())

index_name = 'chatbot-index'

import pinecone
import os

# Load Pinecone API key
api_key = os.getenv('PINECONE_API_KEY') or '0443305a-a580-4277-9ae7-224f4afe3918'
# Set Pinecone environment. Find next to API key in console
env = os.getenv('PINECONE_ENVIRONMENT') or 'us-west1-gcp-free'

pinecone.init(api_key=api_key, environment=env)

import time

# if index_name in pinecone.list_indexes():
#     pinecone.delete_index(index_name)

# # we create a new index
# pinecone.create_index(
#     name=index_name,
#     metric='dotproduct',
#     dimension=1536  # 1536 dim of text-embedding-ada-002
# )

# # wait for index to be initialized
# while not pinecone.describe_index(index_name).status['ready']:
#     time.sleep(1)

index = pinecone.Index(index_name)
index.describe_index_stats()

# index.upsert_from_dataframe(dataset.documents, batch_size=100)

# index.describe_index_stats()

# from langchain.embeddings.openai import OpenAIEmbeddings

# openai_api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'
# model_name = 'text-embedding-ada-002'

# embed = OpenAIEmbeddings(
#     model=model_name,
#     openai_api_key=openai_api_key
# )
