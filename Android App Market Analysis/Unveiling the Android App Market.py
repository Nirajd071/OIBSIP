# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Load datasets (replace 'path_to_apps_data.csv' and 'path_to_reviews_data.csv' with actual file paths)
apps_data = pd.read_csv("C:/Users/niraj/Desktop/Internship/Android App Market Analysis/apps.csv")
reviews_data = pd.read_csv("C:/Users/niraj/Desktop/Internship/Android App Market Analysis/user_reviews.csv")

# Step 1: Data Cleaning (Apps Data)
# Remove duplicates
apps_data.drop_duplicates(inplace=True)

# Handle missing values
missing_apps = apps_data.isnull().sum()
print("\nMissing Values in Apps Dataset:\n", missing_apps)

# Dropping columns with excessive missing data
threshold_apps = 0.5 * len(apps_data)
apps_data.dropna(thresh=threshold_apps, axis=1, inplace=True)
apps_data.dropna(inplace=True)

# Correcting data types (e.g., 'Size', 'Installs', 'Price')
if 'Installs' in apps_data.columns:
    apps_data['Installs'] = apps_data['Installs'].str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    apps_data['Installs'] = pd.to_numeric(apps_data['Installs'], errors='coerce')

if 'Price' in apps_data.columns:
    apps_data['Price'] = apps_data['Price'].str.replace('$', '', regex=False)
    apps_data['Price'] = pd.to_numeric(apps_data['Price'], errors='coerce')

if 'Size' in apps_data.columns:
    def size_to_kb(size):
        if isinstance(size, str):
            if 'M' in size:
                return float(size.replace('M', '')) * 1024
            elif 'k' in size:
                return float(size.replace('k', ''))
        return np.nan

    apps_data['Size'] = apps_data['Size'].apply(size_to_kb)

# Step 2: Data Cleaning (Reviews Data)
# Remove duplicates
reviews_data.drop_duplicates(inplace=True)

# Handle missing values
missing_reviews = reviews_data.isnull().sum()
print("\nMissing Values in Reviews Dataset:\n", missing_reviews)

reviews_data.dropna(inplace=True)

# Step 3: Category Exploration
if 'Category' in apps_data.columns:
    category_distribution = apps_data['Category'].value_counts().sort_values(ascending=False).head(10)
    print("\nCategory Distribution:\n", category_distribution)

    plt.figure(figsize=(12, 6))
    category_distribution.plot(kind='bar', color='skyblue')
    plt.title('Top 10 App Distribution Across Categories')
    plt.xlabel('Category')
    plt.ylabel('Number of Apps')
    plt.xticks(rotation=45)
    plt.show()

# Step 4: Metrics Analysis
# Ratings distribution
if 'Rating' in apps_data.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(apps_data['Rating'], kde=True, bins=20, color='purple')
    plt.title('App Ratings Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

# App size analysis
if 'Size' in apps_data.columns:
    clean_size_data = apps_data['Size'].dropna()

    if not clean_size_data.empty:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=clean_size_data, color='green')
        plt.title('App Size Distribution')
        plt.xlabel('Size (KB)')
        plt.show()
    else:
        print("No valid size data available for plotting.")

# Pricing analysis
if 'Price' in apps_data.columns:
    clean_price_data = apps_data[apps_data['Price'] > 0]['Price']

    if not clean_price_data.empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(clean_price_data, bins=20, color='orange')
        plt.title('Price Distribution of Paid Apps')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')
        plt.show()
    else:
        print("No paid apps data available for plotting.")

# Step 5: Sentiment Analysis
if 'Translated_Review' in reviews_data.columns:
    def analyze_sentiment(review):
        analysis = TextBlob(review)
        return analysis.sentiment.polarity

    reviews_data['Sentiment_Polarity'] = reviews_data['Translated_Review'].apply(str).apply(analyze_sentiment)

    if not reviews_data['Sentiment_Polarity'].dropna().empty:
        plt.figure(figsize=(8, 6))
        sns.histplot(reviews_data['Sentiment_Polarity'], kde=True, color='red')
        plt.title('Sentiment Polarity Distribution')
        plt.xlabel('Sentiment Polarity')
        plt.ylabel('Frequency')
        plt.show()

# Step 6: Correlation Analysis
if 'Rating' in apps_data.columns and 'Installs' in apps_data.columns:
    correlation = apps_data[['Rating', 'Installs']].corr()
    print("\nCorrelation Matrix:\n", correlation)

    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Between Rating and Installs')
    plt.show()

print("\nFull Data Science Analysis and Visualizations Completed.")
