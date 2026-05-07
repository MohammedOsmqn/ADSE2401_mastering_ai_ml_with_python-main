# Python script to demonstrate collaborative filtering for item-based recommendations

# Import required modules
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------------------------
# STEP 1. Create a sample dataset
# -----------------------------------------------------------------------------------------------

data = {
    "user_id": [
        1, 1, 1,
        2, 2,
        3, 3, 3,
        4, 4,
        5, 5, 5,
    ],
    "product": [
        "Laptop", "Mouse", "Keyboard",
        "Laptop", "Mouse",
        "Laptop", "Keyboard", "Monitor",
        "Monitor", "Keyboard",
        "Laptop", "Mouse", "Monitor",
    ],
    "interaction": [1] * 13,
}

# Convert the dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# -----------------------------------------------------------------------------------------------
# STEP 2. Create a user-item / product matrix
# -----------------------------------------------------------------------------------------------
user_item_matrix = df.pivot_table(
    index="user_id",
    columns="product",
    values="interaction",
    fill_value=0
)

print("User-item Matrix:\n", user_item_matrix)

# -----------------------------------------------------------------------------------------------
# STEP 3. Compute item-item similarity matrix using cosine similarity
# -----------------------------------------------------------------------------------------------

# Transpose so that items are rows and users are columns
item_user_matrix = user_item_matrix.T

# Compute cosine similarity between items
item_similarity = cosine_similarity(item_user_matrix)

# Convert to a DataFrame for easy lookup
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=item_user_matrix.index,
    columns=item_user_matrix.index
)

print("\nItem-Item Similarity Matrix:\n", item_similarity_df)

# -----------------------------------------------------------------------------------------------
# STEP 4. Recommendation Function
# -----------------------------------------------------------------------------------------------

def recommend_similar_products(product_name, top_n=3):
    """
    Recommend products similar to the given product.

    args:
        product_name (str): The name of the product.
        top_n (int): The number of similar products to return.

    return:
        List of recommended products.
    """
    if product_name not in item_similarity_df.columns:
        return f"Product {product_name} not found in item similarity matrix."

    # Get similarity scores for the product
    similarity_scores = item_similarity_df[product_name]

    # Remove the product itself
    similarity_scores = similarity_scores.drop(product_name)

    # Get top N most similar products
    similar_products = similarity_scores.sort_values(ascending=False).head(top_n)

    return list(similar_products.index)

# -----------------------------------------------------------------------------------------------
# STEP 5. Test the Collaborative Product Recommendation Engine
# -----------------------------------------------------------------------------------------------

product_2_search = "Laptop"
recommended = recommend_similar_products(product_2_search)

# Disply the recommendations
print(f"\nProducts similar to '{product_2_search}': {recommended}")