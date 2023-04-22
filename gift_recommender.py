import os
import csv
import pandas as pd
from dotenv import load_dotenv
import openai
import pinecone
import re
import json

# Load Pinecone API key
PINECONE_API_KEY = "ddad3c74-f42d-4dfb-a88e-dd199b611d28"
PINECONE_API_ENVIRONMENT = "northamerica-northeast1-gcp"
OPENAI_API_KEY = "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"

NUMBER_TOTAL_PRODUCTS = 1000
NUMBER_FILTERED_PRODUCTS = 5
BATCH_SIZE = 25


def clear_index_data(pinecone_index):
    # Delete all vectors from the index
    pinecone_index.delete(deleteAll=True)


def generate_single_embedding(text, model="text-embedding-ada-002"):
    openai.api_key = OPENAI_API_KEY

    escaped_text = json.dumps(text)  # Escape special characters
    try:
        response = openai.Embedding.create(input=escaped_text, engine=model)
        embedding = response["data"][0]["embedding"]
    except Exception as e:
        print(f"Error generating embedding for text: {text}")
        print(e)
        embedding = None

    return embedding


def generate_embeddings(texts, model="text-embedding-ada-002", parallel_calls=10):
    embeddings = []

    with ThreadPoolExecutor(max_workers=parallel_calls) as executor:
        future_to_text = {
            executor.submit(generate_single_embedding, text, model): text
            for text in texts
        }
        for future in as_completed(future_to_text):
            text = future_to_text[future]
            try:
                embedding = future.result()
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as exc:
                print(f"Error generating embedding for text: {text}")
                print(exc)

    return embeddings


def preprocess_and_build_index(dataframe, batch_size=BATCH_SIZE):
    index_name = "gift-recommender"

    print("Initializing Pinecone...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

    if index_name not in pinecone.list_indexes():
        # Process product descriptions and obtain embeddings
        print("Generating embeddings...")
        product_descriptions = dataframe["product_description"].tolist()
        embeddings = generate_embeddings(product_descriptions)

        print("Completed generating embeddings!")

        embedding_dimension = len(embeddings[0])
        pinecone.create_index(
            name=index_name, dimension=embedding_dimension, metric="cosine", shards=1
        )
        pinecone_index = pinecone.Index(index_name=index_name)
        for i in range(0, len(dataframe), batch_size):
            batch_data = dataframe.iloc[i : i + batch_size]
            embeddings = generate_embeddings(batch_data["product_description"].tolist())
            ids = [str(id) for id in batch_data["id"].tolist()]

            print(f"Upserting embeddings for batch {i // batch_size + 1}...")
            pinecone_index.upsert(vectors=zip(ids, embeddings))

    pinecone_index = pinecone.Index(index_name=index_name)
    # print("Clearing index data...")
    # clear_index_data(pinecone_index)

    return pinecone_index


def retrieve_top_products(user_description, pinecone_index, dataframe, top_n):
    # Convert user_description to an embedding
    search_string = get_search_string(user_description)
    filtered_dataframe = dataframe[
        dataframe["combined_info"].str.contains(search_string, case=False)
    ]
    if len(filtered_dataframe) == 0:
        filtered_dataframe = dataframe
    filtered_descriptions = filtered_dataframe["product_description"].tolist()
    user_embedding = generate_embeddings([search_string])[0]

    # Fetch top-n nearest neighbors from Pinecone index
    print("Retrieving top products...")
    nearest_neighbors = pinecone_index.query(
        vector=user_embedding, top_k=top_n, include_values=True
    )

    # Extract product information from nearest_neighbors
    product_info = []
    for match in nearest_neighbors["matches"]:
        product_id = int(match["id"])
        product = filtered_dataframe.loc[product_id]
        product_info.append(
            (product_id, product["product_name"], product["product_description"])
        )

    return product_info


def get_search_string(user_description):
    prompt = f"Given the user's description of what they are looking for:\n'{user_description}',\nplease generate a search string for finding the most relevant products."

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

    # print("Prompt: ", prompt)

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


from concurrent.futures import ThreadPoolExecutor, as_completed


## TODO Retry if request failed
def generate_single_product():
    openai.api_key = OPENAI_API_KEY
    prompt = "You help create random, realistic names and descriptions for products sold by an online gift retailer. Please generate a gift product name and its description. Please use the format ```Name:{product_name}\nDescription:{product_description}\n```"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=2,
            timeout=10,  # Add a timeout to the API request
        )
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

    item = response.choices[0].message.content.strip()

    pattern = r"Name:(.+)\nDescription:(.+)"
    match = re.search(pattern, item)

    if not match:
        return None

    product_name, product_description = match.groups()

    return {
        "product_name": product_name.strip(),
        "product_description": product_description.strip(),
    }


# Generate sample data
def generate_sample_data(filename, num_samples):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["product_name", "product_description"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        sampleNumber = 1

        with ThreadPoolExecutor() as executor:
            product_futures = [
                executor.submit(generate_single_product) for _ in range(num_samples)
            ]
            for future in as_completed(product_futures):
                product = future.result()
                print(f"Generated product {sampleNumber}: ")
                print(product)
                if product is not None:
                    writer.writerow(product)
                    sampleNumber += 1


# Load data
def load_data(filename):
    data = pd.read_csv(filename)
    data["id"] = data.index  # Add/generate an  'id' column
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

    # Initialize Pinecone
    print("Initializing Pinecone...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

    # Build Pinecone index
    index_name = "gift-recommender"
    # if index_name not in pinecone.list_indexes():
    pinecone_index = preprocess_and_build_index(data)
    # else:
    # pinecone_index = pinecone.Index(index_name=index_name)

    # Test Pinecone-based recommendation
    user_description = input(
        "Hi there! Welcome to Giftem.Shop. What kind of gift are you looking for today?\n"
    )
    recommendations = recommend_gifts(
        user_description, pinecone_index, data, NUMBER_FILTERED_PRODUCTS
    )

    print(recommendations)


if __name__ == "__main__":
    main()
