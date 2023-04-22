import os
import csv
import pandas as pd
from dotenv import load_dotenv
import openai
import pinecone

# Load Pinecone API key
PINECONE_API_KEY = "ddad3c74-f42d-4dfb-a88e-dd199b611d28"
PINECONE_API_ENVIRONMENT = "northamerica-northeast1-gcp"
OPENAI_API_KEY = "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"

NUMBER_TOTAL_PRODUCTS = 2000


def clear_index_data(pinecone_index):
    # Delete all vectors from the index
    pinecone_index.delete(deleteAll=True)


def generate_embeddings(texts, model="text-embedding-ada-002"):
    openai.api_key = OPENAI_API_KEY

    embeddings = []
    for text in texts:
        response = openai.Embedding.create(input=text, engine=model)
        embedding = response["data"][0][
            "embedding"
        ]  # Access the 'embedding' attribute from the 'data' attribute
        # print("Embedding: ")
        # print(embedding)
        embeddings.append(embedding)

    return embeddings


def preprocess_and_build_index(dataframe):
    index_name = "gift-recommender"

    print("Initializing Pinecone...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

    # Process product descriptions and obtain embeddings
    print("Generating embeddings...")
    product_descriptions = dataframe["product_description"].tolist()
    embeddings = generate_embeddings(product_descriptions)

    # Get the dimension of the first embedding
    embedding_dimension = len(embeddings[0])

    # Delete existing index if it exists
    # if index_name in pinecone.list_indexes():
    #     pinecone.delete_index(index_name)

    # Check if index exists
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name, dimension=embedding_dimension, metric="cosine", shards=1
        )

    # Initialize the Pinecone index object
    pinecone_index = pinecone.Index(index_name=index_name)

    # Clear the existing data in the Pinecone index
    print("Clearing index data...")
    clear_index_data(pinecone_index)

    # Upsert embeddings
    print("Upserting embeddings...")
    ids = [str(idx) for idx in range(len(product_descriptions))]
    pinecone_index.upsert(vectors=zip(ids, embeddings))

    return pinecone_index


def retrieve_top_products(user_description, pinecone_index, dataframe, top_n):
    # Convert user_description to an embedding
    user_embedding = generate_embeddings([user_description])[0]

    # Fetch top-n nearest neighbors from Pinecone index
    print("Retrieving top products...")
    nearest_neighbors = pinecone_index.query(
        vector=user_embedding, top_k=top_n, include_values=True
    )

    # Extract product information from nearest_neighbors
    product_info = []
    for match in nearest_neighbors["matches"]:
        product_id = int(match["id"])
        product = dataframe.loc[product_id]
        product_info.append(
            (product_id, product["product_name"], product["product_description"])
        )

    return product_info


def recommend_gifts(user_description, pinecone_index, dataframe, top_n):
    top_products = retrieve_top_products(
        user_description, pinecone_index, dataframe, top_n
    )

    prompt = f"Given the shopper's request/description '{user_description}', choose the best gifts (min 2, max 3) from the following {top_n} options in our catalog:\n\n"

    for index, (product_id, product_name, product_description) in enumerate(
        top_products
    ):
        prompt += f"{index + 1}. {product_name} - {product_description}\n"

    prompt += "\nThe best gifts are:"

    print("Prompt: ", prompt)

    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.3,
    )

    return response.choices[0].message["content"].strip()


# Generate sample data
def generate_sample_data(filename, num_samples):
    openai.api_key = OPENAI_API_KEY
    prompt = (
        f"Generate {num_samples} realistic gift product names and their descriptions:\n"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.8,
    )

    product_data = response.choices[0].message.content.strip().split("\n")

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["product_name", "product_description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in product_data:
            if ":" not in item:
                continue

            product_name, product_description = item.split(":", 1)
            writer.writerow(
                {
                    "product_name": product_name.strip(),
                    "product_description": product_description.strip(),
                }
            )


# Load data
def load_data(filename):
    data = pd.read_csv(filename)
    data["combined_info"] = data.apply(
        lambda x: f"{x['product_name']} {x['product_description']}", axis=1
    )
    return data


def main():
    data_filename = "sample_products.csv"

    if not os.path.exists(data_filename) or os.path.getsize(data_filename) == 0:
        print("Generating sample data...")
        generate_sample_data(data_filename, NUMBER_TOTAL_PRODUCTS)
    else:
        print("Sample data file already exists, skipping data generation.")

    data = load_data(data_filename)

    # Build Pinecone index
    pinecone_index = preprocess_and_build_index(data)

    # Test Pinecone-based recommendation
    user_description = input(
        "Hi there! Welcome to Giftem.Shop. What kind of gift are you looking for today?\n"
    )
    recommendations = recommend_gifts(
        user_description, pinecone_index, data, NUMBER_TOTAL_PRODUCTS
    )

    print(recommendations)


if __name__ == "__main__":
    main()
