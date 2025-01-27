# ecommerce_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from datetime import datetime

# 1. Load Data
customers = pd.read_csv('Customers.csv')
products = pd.read_csv('Products.csv')
transactions = pd.read_csv('Transactions.csv')

# 2. Data Cleaning
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Check for missing values
print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())

# 3. EDA - Visualizations
# Distribution of Product Prices
plt.figure(figsize=(10, 6))
sns.histplot(products['Price'], kde=True)
plt.title('Distribution of Product Prices')
plt.show()

# Monthly Total Transaction Value
transaction_monthly = transactions.groupby(transactions['TransactionDate'].dt.to_period("M")).agg({'TotalValue': 'sum'})
transaction_monthly.plot(kind='line', figsize=(12, 6), title="Monthly Transaction Total")
plt.show()

# Regional Distribution of Customers
plt.figure(figsize=(10, 6))
sns.countplot(x='Region', data=customers)
plt.title('Distribution of Customers by Region')
plt.show()

# 4. Lookalike Model (Removed saving Lookalike.csv)

# Merging Customer Data with Transactions
merged_data = pd.merge(transactions, customers, on='CustomerID', how='left')

# Aggregate Transaction Data
customer_transactions = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': 'nunique',
    'TransactionDate': 'nunique'
}).reset_index()

# Feature Engineering
customer_transactions['AvgTransactionValue'] = customer_transactions['TotalValue'] / customer_transactions['TransactionDate']

# Normalize the features
scaler = StandardScaler()
customer_transactions_scaled = scaler.fit_transform(customer_transactions[['TotalValue', 'Quantity', 'ProductID', 'AvgTransactionValue']])

# Calculate Cosine Similarity
similarity_matrix = cosine_similarity(customer_transactions_scaled)

# Convert Similarity Matrix to DataFrame
similarity_df = pd.DataFrame(similarity_matrix, index=customer_transactions['CustomerID'], columns=customer_transactions['CustomerID'])

# Get Top 3 Similar Customers for First 20 Customers
lookalike_dict = {}
for customer_id in customer_transactions['CustomerID'].head(20):
    similar_customers = similarity_df[customer_id].sort_values(ascending=False).iloc[1:4]
    lookalike_dict[customer_id] = [(similar_customer, score) for similar_customer, score in zip(similar_customers.index, similar_customers.values)]

# 5. Customer Segmentation (Clustering)

# Aggregate Transaction Information for Clustering
customer_profile = merged_data.groupby('CustomerID').agg({
    'TotalValue': 'sum',
    'Quantity': 'sum',
    'ProductID': 'nunique',
    'TransactionDate': 'nunique'
}).reset_index()

# Normalize Features for Clustering
customer_profile_scaled = scaler.fit_transform(customer_profile[['TotalValue', 'Quantity', 'ProductID', 'TransactionDate']])

# Apply KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_profile['Cluster'] = kmeans.fit_predict(customer_profile_scaled)

# Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(customer_profile_scaled, customer_profile['Cluster'])
print(f'Davies-Bouldin Index: {db_index:.2f}')

# PCA for 2D Visualization of Clusters
pca = PCA(n_components=2)
principal_components = pca.fit_transform(customer_profile_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=principal_components[:, 0], y=principal_components[:, 1], hue=customer_profile['Cluster'], palette='viridis')
plt.title(f'Customer Segmentation (DB Index: {db_index:.2f})')
plt.show()
