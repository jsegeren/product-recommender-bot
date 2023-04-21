import os
import csv
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai
import pinecone

# Load Pinecone API key
PINECONE_API_KEY = "ddad3c74-f42d-4dfb-a88e-dd199b611d28"
PINECONE_API_ENVIRONMENT = "northamerica-northeast1-gcp"
OPENAI_API_KEY = "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"

def preprocess_and_build_index(embeddingModel, dataframe):
    index_name = "gift-recommender"

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)
    # Calculate embeddings
    dataframe["embedding"] = dataframe["product_description"].apply(lambda x: embeddingModel.encode(x))
    embeddings = dataframe["embedding"].tolist()

    # Check if index exists
    if index_name not in pinecone.list_indexes():
        # Create Pinecone index
        pinecone.create_index(name=index_name, dimension=len(embeddings[0]), metric='cosine', shards=1)

        # Upsert embeddings
        pinecone_index = pinecone.Index(index_name=index_name)
        pinecone_index.upsert(vectors={str(idx): emb for idx, emb in enumerate(embeddings)})
    else:
        # Initialize the Pinecone index object
        pinecone_index = pinecone.Index(index_name=index_name)

    return pinecone_index

def retrieve_top_products(user_description, pinecone_index, model, dataframe, top_n):
    # Calculate the user description embedding
    query_embedding = model.encode(user_description).tolist()  # Convert numpy array to list

    # Fetch top-n nearest neighbors from Pinecone index
    nearest_neighbors = pinecone_index.query([query_embedding], top_k=top_n, include_metadata=True)
    
    # Extract product information from nearest_neighbors
    product_info = []
    for match in nearest_neighbors['matches']:
        product_id = int(match['id'])
        product = dataframe.loc[product_id]
        product_info.append((product_id, product['product_name'], product['product_description']))
    
    return product_info


def recommend_gifts(user_description, pinecone_index, model, dataframe, top_n):
    top_products = retrieve_top_products(user_description, pinecone_index, model, dataframe, top_n)
    
    prompt = f"Given the shopper's request/description '{user_description}', list the top 3 gifts from our catalogue of available products:\n\n"

    for index, (product_id, product_name, product_description) in enumerate(top_products):
        prompt += f"{index + 1}. {product_name} - {product_description}\n"

    prompt += "\Suggested gifts:"

    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.3,
    )

    return response.choices[0].message['content'].strip()

# Generate sample data
def generate_sample_data(filename, num_samples):
    openai.api_key = OPENAI_API_KEY
    prompt = f"Generate {num_samples} realistic gift product names and their descriptions:\n"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.8,
    )

    product_data = response.choices[0].message.content.strip().split('\n')

    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['product_name', 'product_description']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in product_data:
            if ':' not in item:
                continue

            product_name, product_description = item.split(':', 1)
            writer.writerow({'product_name': product_name.strip(), 'product_description': product_description.strip()})


# Load data
def load_data(filename):
    data = pd.read_csv(filename)
    data['combined_info'] = data.apply(
        lambda x: f"{x['product_name']} {x['product_description']}",
        axis=1
    )
    return data

def main():
    data_filename = 'sample_products.csv'
    
    if not os.path.exists(data_filename) or os.path.getsize(data_filename) == 0:
        print("Generating sample data...")
        generate_sample_data(data_filename, 5)
    else:
        print("Sample data file already exists, skipping data generation.")

    data = load_data(data_filename)

    # Initialize the sentence embedding model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Build Pinecone index
    pinecone_index = preprocess_and_build_index(model, data)

    # Test Pinecone-based recommendation
    user_description = input("Hi there! Welcome to Giftem.Shop. What kind of gift are you looking for today?\n")
    recommendations = recommend_gifts(user_description, pinecone_index, model, data, 3)
    print(recommendations)


if __name__ == "__main__":
    main()
