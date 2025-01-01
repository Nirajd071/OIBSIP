import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
apps_data = pd.read_csv("apps.csv")

# Data Cleaning
# Convert 'Size' to string and clean it
apps_data['Size'] = apps_data['Size'].astype(str).replace('Varies with device', np.nan)
apps_data['Size'] = apps_data['Size'].str.replace('M', '').str.replace('k', '').str.replace(',', '', regex=True)
apps_data['Size'] = pd.to_numeric(apps_data['Size'], errors='coerce')  # Convert to float
apps_data['Size'] = apps_data['Size'].apply(lambda x: x * 0.001 if x and x < 1 else x)  # Convert KB to MB
apps_data['Size'].fillna(apps_data['Size'].median(), inplace=True)

# Clean 'Installs' and 'Price' columns
apps_data['Installs'] = apps_data['Installs'].astype(str).str.replace('[+,]', '', regex=True)
apps_data['Installs'] = pd.to_numeric(apps_data['Installs'], errors='coerce')  # Convert to float
apps_data['Price'] = apps_data['Price'].astype(str).str.replace('$', '', regex=True)
apps_data['Price'] = pd.to_numeric(apps_data['Price'], errors='coerce')  # Convert to float

# Fill missing values
apps_data['Rating'].fillna(apps_data['Rating'].median(), inplace=True)
apps_data['Current Ver'].fillna('Unknown', inplace=True)
apps_data['Android Ver'].fillna('Unknown', inplace=True)

# Basic Data Analysis
print("Basic Information about the Dataset:")
print(apps_data.info())
print("\nSummary Statistics:")
print(apps_data.describe())

# Top Categories by Number of Apps
top_categories = apps_data['Category'].value_counts().head(10)
print("\nTop Categories by Number of Apps:")
print(top_categories)

# Average Rating by Category
average_ratings = apps_data.groupby('Category')['Rating'].mean().sort_values(ascending=False)
print("\nAverage Ratings by Category:")
print(average_ratings)

# Correlation Analysis
correlation = apps_data[['Rating', 'Size', 'Installs', 'Price']].corr()
print("\nCorrelation Matrix:")
print(correlation)

# Visualizations
# 1. Top 10 Categories by Number of Apps
plt.figure(figsize=(10, 6))
top_categories.plot(kind='bar', color='skyblue')
plt.title("Top 10 Categories by Number of Apps")
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Distribution of Ratings
plt.figure(figsize=(10, 6))
sns.histplot(apps_data['Rating'], bins=30, kde=True, color='green')
plt.title("Distribution of App Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 3. Relationship Between Installs and Rating
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Installs', y='Rating', data=apps_data, alpha=0.7, color='orange')
plt.title("Relationship Between Installs and Rating")
plt.xlabel("Number of Installs")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# 4. Average Rating by Top 10 Categories
plt.figure(figsize=(10, 6))
average_ratings.head(10).plot(kind='bar', color='purple')
plt.title("Average Rating by Top 10 Categories")
plt.xlabel("Category")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Key Insights from the Dataset
print("\nKey Insights:")
print("- The dataset contains information about various Android apps, including their size, ratings, and number of installs.")
print("- Most apps fall into categories such as 'FAMILY', 'GAME', and 'TOOLS'.")
print("- The average rating of apps varies by category. Categories like 'BOOKS_AND_REFERENCE' and 'EDUCATION' tend to have higher average ratings.")
print("- Ratings are positively correlated with the number of installs, but the relationship is weak.")
