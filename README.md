# Product Recommender

<!-- ![Product Recommender Cover Image](https://example.com/path/to/your/image.jpg) -->

Product Recommender is a prototype semantic search and recommendation system for product suggestions based on user descriptions. This program leverages Pinecone's vector search capabilities and OpenAI's GPT-4 to generate a list of product recommendations that closely match the user's requirements. This program can be used for eCommerce applications or similar scenarios.

## Requirements

To run this program, you'll need the following:

1. Python 3.6 or later
2. Pinecone API key
3. OpenAI API key
4. The following Python libraries:
   - pandas
   - dotenv
   - openai
   - pinecone
   - concurrent.futures
   - hashlib

## Installation

1. Clone the repository.
2. Install the required Python packages:

```
pip install -r requirements.txt
```

3. Set up a `.env` file in the same directory as the script with your Pinecone and OpenAI API keys:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_API_ENVIRONMENT=your_pinecone_api_environment
OPENAI_API_KEY=your_openai_api_key
```

## Usage

1. Run the script:

```
python product_recommender.py
```

2. Follow the prompts to input a description of the gift you are looking for.

3. The script will generate product recommendations based on the user's description.

## How it works

The script will perform the following steps:

1. If a sample product data file (`sample_products.csv`) doesn't exist or is empty, the script will generate sample product data using OpenAI's GPT-3.5-turbo model.
2. Load the product data and preprocess it by combining the product information into a single string.
3. If the product data has changed since the last run, rebuild the Pinecone index by generating embeddings for the product data and storing them in Pinecone. Embeddings are generated using OpenAI's [text-embedding-ada-002](https://platform.openai.com/docs/models/embeddings) model.
4. Prompt the user to describe the gift they're looking for.
5. Generate a search string based on the user's gift description using OpenAI's GPT-4.
6. Retrieve the top products from the Pinecone index that match the search string.
7. Choose the best gift recommendations from the top products using GPT-4.
8. Share and explain the gift recommendations to the user.

## License

This project is licensed under the MIT License. Please refer to the LICENSE file for more information.

## Contributing

- Contributions, issues, and feature requests are welcome.
- Feel free to check the issues page to see if you can help with anything.
- If you encounter a bug, or have a feature request, please upvote an issue, or create a new one.

## Acknowledgments

- OpenAI for making the GPT-4 and GPT-3.5-turbo models available.
- Pinecone for its efficient vector search service.
