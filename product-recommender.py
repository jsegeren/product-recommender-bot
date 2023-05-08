import os
import csv
import pandas as pd
from dotenv import load_dotenv
import openai
import pinecone
import re
import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import blake3
from dotenv import load_dotenv
from tkinter import ttk
import tkinter as tk
import customtkinter as ctk
import threading
import math
import textwrap

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

load_dotenv()

# Load API keys (secret)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_API_ENVIRONMENT = os.getenv("PINECONE_API_ENVIRONMENT", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# print(f"Pinecone API Key: {PINECONE_API_KEY}")
# print(f"Pinecone API Environment: {PINECONE_API_ENVIRONMENT}")
# print(f"OpenAI API Key: {OPENAI_API_KEY}")

NUMBER_TOTAL_PRODUCTS = 3000
NUMBER_FILTERED_PRODUCTS = 20
NUMBER_FINAL_RECOMMENDATIONS = 3

REQUEST_BATCH_SIZE = 25
REQUEST_RETRIES = 3
REQUEST_RETRY_BACKOFF = 2
REQUEST_PARALLEL_CALLS = 10

HASH_FILENAME = "hash.txt"
DATA_FILENAME = "sample_products.csv"

UI_APP_TITLE = "Gift Recommendation Assistant"
UI_SCALING = 1.3
UI_WINDOW_WIDTH = 1100
UI_WINDOW_HEIGHT = 580

USE_CUSTOM_PARSER = True  # Set to True to use the custom parser
CUSTOM_DATA_FILENAME = "./giftem/giftem-products.json"  # Change this to your custom data file when using the custom parser


def compute_file_hash(filename):
    hash_blake3 = blake3.blake3()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_blake3.update(chunk)
    return hash_blake3.hexdigest()


def clear_index_data(pinecone_index):
    # Delete all vectors from the index
    pinecone_index.delete(deleteAll=True)


def generate_single_embedding(
    text,
    model="text-embedding-ada-002",
    retries=REQUEST_RETRIES,
    backoff=REQUEST_RETRY_BACKOFF,
):
    openai.api_key = OPENAI_API_KEY

    escaped_text = json.dumps(text)  # Escape special characters
    attempt = 0
    while attempt <= retries:
        try:
            response = openai.Embedding.create(input=escaped_text, engine=model)
            embedding = response["data"][0]["embedding"]  # type: ignore
            return embedding
        except Exception as e:
            print(f"Error generating embedding for text: {text}")
            print(e)
            if attempt == retries:
                return None
            attempt += 1
            time.sleep(backoff**attempt)


def generate_embeddings(
    texts, model="text-embedding-ada-002", parallel_calls=REQUEST_PARALLEL_CALLS
):
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


def preprocess_and_build_index(dataframe, rebuild=False, batch_size=REQUEST_BATCH_SIZE):
    index_name = "gift-recommender"
    is_index_empty = False

    print("Initializing Pinecone...")
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

    if rebuild and index_name in pinecone.list_indexes():
        print("Clearing existing index...")
        pinecone_index = pinecone.Index(index_name=index_name)
        clear_index_data(pinecone_index)
        is_index_empty = True

    if index_name not in pinecone.list_indexes():
        # Process product descriptions and obtain embeddings
        print("Generating embeddings...")
        combined_info = dataframe["combined_info"].tolist()
        embeddings = generate_embeddings(combined_info)

        print("Completed generating embeddings!")

        embedding_dimension = len(embeddings[0])
        pinecone.create_index(
            name=index_name, dimension=embedding_dimension, metric="cosine", shards=1
        )
        print("Created new index (Pinecone)!")
        is_index_empty = True

    if is_index_empty:
        pinecone_index = pinecone.Index(index_name=index_name)
        number_total_items = len(dataframe)
        number_total_batches = number_total_items // batch_size + 1
        for i in range(0, number_total_items, batch_size):
            batch_data = dataframe.iloc[i : i + batch_size]
            embeddings = generate_embeddings(batch_data["combined_info"].tolist())
            ids = [str(id) for id in batch_data["id"].tolist()]

            pinecone_index.upsert(vectors=zip(ids, embeddings))  # type: ignore
            print(
                f"Finished upserting embeddings for batch {i // batch_size + 1} of {number_total_batches}..."
            )

    pinecone_index = pinecone.Index(index_name=index_name)
    # print("Clearing index data...")
    # clear_index_data(pinecone_index)

    return pinecone_index


def retrieve_top_products(user_description, pinecone_index, dataframe, top_n):
    # Convert user_description to an embedding
    search_string = get_search_string(user_description)
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

        if product_id not in dataframe.index:
            print(f"Product ID {product_id} not found in dataframe, skipping")
            continue

        product = dataframe.loc[product_id]
        print(f"Matched product {product_id} - {product['name']}")
        product_info.append(
            (
                product_id,
                product["name"],
                product["price"],
                product["description"],
            )
        )

    return product_info


def get_search_string(user_description):
    prompt = f"Given the user's description of what they are looking for:\n'{user_description}',\nplease think of a good gift idea, and create a simple search query for finding that product via an online store product catalogue search. E.g. If user is looking for a 'gift for a doctor' please give a search query like 'relaxation', or 'home office'. If the user is looking for a 'coffee maker', then you can just directly give a search query of 'coffee maker'. Please *ONLY* output the verbatim query string."

    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.3,
    )

    search_string = response.choices[0].message["content"].strip()  # type: ignore
    print("Search string: " + search_string)

    return search_string


def recommend_gifts(
    search_query,
    pinecone_index,
    dataframe,
    number_filtered_products,
    number_final_recommendations,
    callback=None,
):
    top_products = retrieve_top_products(
        search_query, pinecone_index, dataframe, number_filtered_products
    )

    prompt = f'You are an expert personal shopping assistant. Given the product search query, "{search_query}", please choose the best {number_final_recommendations} gifts from the following relevant {number_filtered_products} options in our product catalogue:\n\n'

    for index, (
        product_id,
        product_name,
        product_price,
        product_description,
    ) in enumerate(top_products):
        prompt += (
            f"{product_id}. {product_name} ({product_price}) - {product_description}\n"
        )

    prompt += "\nPlease give your recommendations in this format: {rank}. *{product name}* (${product price}) - {reason / explanation why this is a good gift suggestion}."

    # print("Prompt: ", prompt)

    openai.api_key = OPENAI_API_KEY
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
        stream=True,
    )

    # Handle the streaming response and call the callback
    message_parts = []
    first_chunk = True
    for chunk in response:
        chunk_message = chunk["choices"][0]["delta"]
        content = chunk_message.get("content", "")
        if content:
            message_parts.append(content)
            if callback:
                callback(
                    "".join(message_parts),
                    update_existing=not first_chunk,
                    streaming=True,
                    first_chunk=first_chunk,
                )  # Add first_chunk parameter
                first_chunk = False


def generate_single_product(retries=REQUEST_RETRIES, backoff=REQUEST_RETRY_BACKOFF):
    product_categories = [
        "Electronics",
        "Home & Kitchen",
        "Clothing & Accessories",
        "Beauty & Personal Care",
        "Books",
        "Sports & Outdoors",
        "Toys & Games",
        "Automotive",
        "Health & Household",
        "Grocery & Gourmet Food",
        "Office Products",
        "Patio, Lawn & Garden",
        "Arts, Crafts & Sewing",
        "Tools & Home Improvement",
        "Pet Supplies",
        "Baby",
        "Appliances",
        "Cell Phones & Accessories",
        "Industrial & Scientific",
        "Musical Instruments",
        "Movies & TV",
        "Software",
        "Video Games",
        "Jewelry",
        "Watches",
        "Handmade Products",
        "Collectibles & Fine Art",
        "Camera & Photo",
        "Computers & Accessories",
        "Furniture",
        # Add more categories as needed
    ]
    openai.api_key = OPENAI_API_KEY

    # Choose a random category
    category = random.choice(product_categories)

    prompt = (
        f"You are the VP of Product Marketing for a major online retailer. "
        f"Please provide the NAME, PRICE, and SHORT DESCRIPTION for a unique and completely random product in the '{category}' category sold by your store. "
        "The description must be a single, clear, concise sentence (less than 10 words) that is helpful to a potential customer. "
        "Avoid using excessive keywords and focus on providing a meaningful, human-readable, English-language description. "
        'Please use the format "Name:{product name}\nPrice:{product price}\nDescription:{product description}\n"'
    )

    attempt = 0
    while attempt <= retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=1.0,  # Increase the temperature to 1.0 for more randomness
                timeout=10,  # Add a timeout to the API request
            )

            item = response.choices[0].message.content.strip()  # type: ignore

            pattern = r"Name:(.+)\nPrice:(.+)\nDescription:(.+)"
            match = re.search(pattern, item)

            if not match:
                attempt += 1
                continue

            product_name, product_price, product_description = match.groups()

            # Ensure the description is less than 50 words
            description_words = product_description.strip().split()
            if len(description_words) > 50:
                attempt += 1
                continue

            return {
                "product_name": product_name.strip(),
                "product_price": product_price.strip(),
                "product_description": product_description.strip(),
            }
        except Exception as e:
            print(f"Error during API call: {e}")
            if attempt == retries:
                return None
            attempt += 1
            time.sleep(backoff**attempt)


# Generate sample data
def generate_sample_data(filename, num_samples, batch_size=REQUEST_BATCH_SIZE):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["product_name", "product_price", "product_description"]
        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames,
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        writer.writeheader()

        sampleNumber = 1

        # Split the data generation into smaller batches
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch in range(num_batches):
            samples_in_batch = (
                batch_size
                if batch < num_batches - 1
                else num_samples - batch * batch_size
            )
            with ThreadPoolExecutor(max_workers=samples_in_batch) as executor:
                product_futures = [
                    executor.submit(generate_single_product)
                    for _ in range(samples_in_batch)
                ]
                for future in as_completed(product_futures):
                    product = future.result()
                    print(f"Finished generating product {sampleNumber}: ")
                    print(product)
                    if product is not None:
                        writer.writerow(product)
                        sampleNumber += 1


def standardize_column_names(data):
    column_mapping = {
        "product_name": "name",
        "product_price": "price",
        "product_description": "description",
    }
    data = data.rename(columns=column_mapping)
    return data


# Load and parse data
# Support configurability to load using custom parser (for custom product datasets)
def load_and_parse_data(filename, use_custom_parser=False):
    if use_custom_parser:
        data = parse_giftem_data(filename)
    else:
        data = load_data(filename)

    data = standardize_column_names(data)  # Standardize column names

    data["id"] = data.index  # Add/generate an 'id' column
    data["combined_info"] = data.apply(
        lambda x: f"{x['name']} {x['price']} {x['description']}",
        axis=1,
    )
    return data


# Load data
def load_data(filename):
    data = pd.read_csv(filename)
    data["id"] = data.index  # Add/generate an  'id' column
    data["combined_info"] = data.apply(
        lambda x: f"{x['product_name']} {x['product_price']} {x['product_description']}",
        axis=1,
    )
    return data


def parse_giftem_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        giftem_data = json.load(f)

    data_list = []
    for product in giftem_data:
        product_id = product.get("id")
        product_SKU = product.get("SKU")
        product_name = product.get("name", "")
        product_description = product.get("description", "")
        unit_cost = product.get("unitCost")
        weight = product["inventory"].get("shippingValue")
        weight_unit = product["inventory"].get("weightUnit")
        requires_shipping = product["inventory"].get("requiresShipping", False)
        categories = [
            category["name"]
            for category in product["organization"].get("categories", [])
        ]
        tags = product["organization"].get("tags", [])
        brands = product["organization"].get("brands", [])
        vendors = [
            vendor["name"] for vendor in product["organization"].get("vendors", [])
        ]
        images = [image["url"] for image in product.get("images", [])]

        # Set product_price to unitPrice if channelSpecific is not empty, otherwise set it to None
        # Note that this dataset seems to represent prices in units of cents; we want to convert to dollars
        product_price = (
            product["channelSpecific"][0]["configs"]["unitPrice"] / 100
            if product.get("channelSpecific")
            else None
        )

        combined_info = f"{product_name} {product_price} {product_description}"

        data_list.append(
            {
                "id": product_id,
                "SKU": product_SKU,
                "name": product_name,
                "price": product_price,
                "description": product_description,
                "unitCost": unit_cost,
                "weight": weight,
                "weightUnit": weight_unit,
                "requiresShipping": requires_shipping,
                "categories": categories,
                "tags": tags,
                "brands": brands,
                "vendors": vendors,
                "images": images,
                "combined_info": combined_info,
            }
        )

    dataframe = pd.DataFrame(data_list)
    return dataframe


class ChatBotUI(ctk.CTk):
    def __init__(self, pinecone_index, data):
        super().__init__()
        self.pinecone_index = pinecone_index
        self.data = data
        self.chat_labels = []
        self.create_chat_area()
        self.title(UI_APP_TITLE)
        self.geometry(f"{UI_WINDOW_WIDTH}x{UI_WINDOW_HEIGHT}")

        self.create_text_input_area()

        self.bot_message_in_progress = None

        ctk.set_widget_scaling(UI_SCALING)  # Custom DPI scaling

    def create_chat_area(self):
        self.chat_area_frame = ctk.CTkFrame(self)
        self.chat_area_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        self.chat_area = ctk.CTkScrollableFrame(self.chat_area_frame)
        self.chat_area.pack(expand=True, fill=tk.BOTH)

    def create_text_input_area(self):
        self.text_input_frame = ctk.CTkFrame(self)
        self.text_input_frame.pack(fill=tk.X, padx=5, pady=5)

        self.text_input = ctk.CTkEntry(
            self.text_input_frame, font=ctk.CTkFont(size=12), width=30
        )
        self.text_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.text_input.bind("<Return>", self.on_submit)

        self.submit_button = ctk.CTkButton(
            self.text_input_frame,
            text="Submit",
            command=self.on_submit,
        )
        self.submit_button.pack(side=tk.RIGHT)

    def count_textbox_lines(self, textbox):
        text = textbox.get(1.0, tk.END).strip()
        font = ctk.CTkFont(size=12)
        linespace = font.metrics()["linespace"]
        widget_width = round(UI_WINDOW_WIDTH * 0.8)
        wrap_width = widget_width // font.measure("a")
        wrapped_lines = textwrap.wrap(text, wrap_width)
        num_lines = len(wrapped_lines) + sum(
            [line.count("\n") for line in wrapped_lines]
        )
        return num_lines

    def calculate_textbox_height(self, old_chat_text):
        num_lines = self.count_textbox_lines(old_chat_text)
        font = ctk.CTkFont(size=12)
        linespace = font.metrics()["linespace"]
        return num_lines * linespace * 2.1

    def create_new_textbox(self, old_chat_text, role):
        new_height = self.calculate_textbox_height(old_chat_text)

        new_chat_text = ctk.CTkTextbox(
            self.chat_area,
            wrap=tk.WORD,
            font=ctk.CTkFont(size=12),
            padx=10,
            pady=10,
            width=(round(UI_WINDOW_WIDTH * 0.8)),
            height=math.ceil(new_height),
            activate_scrollbars=False,
        )
        new_chat_text.insert(tk.END, old_chat_text.get(1.0, tk.END))
        new_chat_text.configure(state="disabled")

        if role == "bot":
            self.bot_message_in_progress = new_chat_text
        else:
            self.chat_labels.append(new_chat_text)

        return new_chat_text

    def update_chat(
        self, message, update_existing=False, streaming=False, first_chunk=False
    ):
        if not update_existing or first_chunk:
            self.append_chat("bot", "", update_existing=update_existing)
            self.new_bot_bubble_required = False

        if self.bot_message_in_progress:
            chat_text = self.bot_message_in_progress
            chat_text.configure(state="normal")

            # if streaming:
            chat_text.delete(1.0, tk.END)
            chat_text.insert(tk.END, f"Bot: {message}")
            # else:
            # chat_text.delete(1.0, tk.END)
            # chat_text.insert(tk.END, f"Bot: {message}")

            chat_text.configure(state="disabled")

            # Create a new textbox with the appropriate height and replace the old one
            new_chat_text = self.create_new_textbox(chat_text, role="bot")
            chat_text.pack_forget()
            new_chat_text.pack(anchor=tk.W, padx=10, pady=5)
            self.bot_message_in_progress = new_chat_text

        self.chat_area.update_idletasks()
        self.remove_loading_spinner()

    def append_chat(self, role, message, update_existing=False):
        font = ctk.CTkFont(size=12)
        linespace = font.metrics()["linespace"]
        height = 1 * linespace * 2

        if update_existing and role == "bot" and self.bot_message_in_progress:
            chat_text = self.bot_message_in_progress
            chat_text.configure(state="normal")
            new_text = f"Bot: {message}"
            chat_text.delete(1.0, tk.END)
            chat_text.insert(tk.END, new_text)
            chat_text.configure(state="disabled")

            chat_text.pack(anchor=tk.W, padx=10, pady=5)

            self.chat_area.update()
        else:
            if role == "user":
                message = f"You: {message}"
                self.bot_message_in_progress = None
            elif role == "bot":
                message = f"Bot: {message}"
                if not self.bot_message_in_progress:
                    self.bot_message_in_progress = ctk.CTkTextbox(
                        self.chat_area,
                        wrap=tk.WORD,
                        font=ctk.CTkFont(size=12),
                        padx=10,
                        pady=10,
                        width=(round(UI_WINDOW_WIDTH * 0.8)),
                        height=height,
                        activate_scrollbars=False,
                    )
                    self.bot_message_in_progress.pack(anchor=tk.W, padx=10, pady=5)
                    self.chat_labels.append(self.bot_message_in_progress)

            chat_text = (
                self.bot_message_in_progress
                if role == "bot" and self.bot_message_in_progress
                else ctk.CTkTextbox(
                    self.chat_area,
                    wrap=tk.WORD,
                    font=ctk.CTkFont(size=12),
                    padx=10,
                    pady=10,
                    width=(round(UI_WINDOW_WIDTH * 0.8)),
                    height=height,
                    activate_scrollbars=False,
                )
            )

            chat_text.insert(tk.END, message)
            chat_text.configure(state="disabled")

            chat_text.pack(anchor=tk.W, padx=(10 if role == "bot" else 50), pady=5)

            if not self.bot_message_in_progress or role == "user":
                self.chat_labels.append(chat_text)
            self.chat_area.update()

    def append_loading_spinner(self):
        self.loading_spinner = ctk.CTkLabel(
            self.chat_area,
            text="Great! Working on it...",
            font=ctk.CTkFont(size=12),
            padx=10,
            pady=10,
        )
        self.loading_spinner.pack(anchor=tk.W, padx=50, pady=5)

    def remove_loading_spinner(self):
        self.loading_spinner.pack_forget()
        self.chat_area.update()

    def on_submit(self, event=None):
        user_input = self.text_input.get().strip()
        if user_input:
            self.text_input.delete(0, tk.END)
            self.append_chat("user", user_input)
            self.new_bot_bubble_required = True

            # Disable text input and submit button
            self.text_input.configure(state="disabled")
            self.submit_button.configure(state="disabled")

            recommendation_thread = threading.Thread(
                target=self.get_recommendations_and_append, args=(user_input,)
            )
            recommendation_thread.start()

    def get_recommendations_and_append(self, user_input):
        self.append_loading_spinner()

        recommendations = recommend_gifts(
            user_input,
            self.pinecone_index,
            self.data,
            NUMBER_FILTERED_PRODUCTS,
            NUMBER_FINAL_RECOMMENDATIONS,
            callback=lambda message, update_existing, streaming=True, first_chunk=False: self.update_chat(
                message, update_existing, streaming, first_chunk
            ),
        )

        self.remove_loading_spinner()
        self.append_chat("bot", recommendations)

        # Re-enable text input and submit button
        self.text_input.configure(state="normal")
        self.submit_button.configure(state="normal")


def main():
    if USE_CUSTOM_PARSER:
        data_filename = CUSTOM_DATA_FILENAME
    elif not os.path.exists(DATA_FILENAME) or os.path.getsize(DATA_FILENAME) == 0:
        print("Generating sample data...")
        generate_sample_data(DATA_FILENAME, NUMBER_TOTAL_PRODUCTS)
        data_filename = DATA_FILENAME
    else:
        print("Sample data file already exists, skipping data generation.")
        data_filename = DATA_FILENAME

    data = load_and_parse_data(data_filename, USE_CUSTOM_PARSER)

    # Compute the hash of the current CSV file
    current_hash = compute_file_hash(data_filename)

    # Check if the stored hash file exists and read the hash
    if os.path.exists(HASH_FILENAME):
        with open(HASH_FILENAME, "r") as f:
            stored_hash = f.read().strip()
    else:
        stored_hash = None

    # Update the index if the hash has changed
    rebuild_index = stored_hash != current_hash

    # Initialize Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENVIRONMENT)

    # Build Pinecone index
    index_name = "gift-recommender"
    pinecone_index = preprocess_and_build_index(data, rebuild=rebuild_index)

    # If the index was rebuilt, update the stored hash
    if rebuild_index:
        with open(HASH_FILENAME, "w") as f:
            f.write(current_hash)

    # Display UI
    # display_ui(lambda user_description: recommend_gifts(user_description, pinecone_index, data, NUMBER_FILTERED_PRODUCTS, NUMBER_FINAL_RECOMMENDATIONS))

    # Display UI
    # Initialize Tkinter
    ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
    ctk.set_default_color_theme(
        "blue"
    )  # Themes: "blue" (standard), "green", "dark-blue"

    chatbot_ui = ChatBotUI(pinecone_index, data)

    # Show welcome message
    chatbot_ui.append_chat(
        "bot",
        "Hi there! Welcome to Joshua's Gifts. What kind of gift are you looking for today?",
    )

    # Run the Tkinter event loop
    chatbot_ui.mainloop()

    # Test Pinecone-based recommendation
    # user_description = input(
    #     "Hi there! Welcome to Joshua's Gifts. What kind of gift are you looking for today?\n\n"
    # )
    # recommendations = recommend_gifts(
    #     user_description, pinecone_index, data, NUMBER_FILTERED_PRODUCTS, NUMBER_FINAL_RECOMMENDATIONS
    # )

    # print(recommendations)


if __name__ == "__main__":
    main()
