# EDA ( exploratory data analysis )

import pandas as pd

df = pd.read_csv("mushrooms.csv")
df.head(5)

df.isnull().sum()

df.info()

df.duplicated().sum()

# Feature Association & Selection

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# 1. Load data
df = pd.read_csv('mushrooms.csv')

# 2. Encode categorical data to numbers for the math engine
df_encoded = df.apply(LabelEncoder().fit_transform)

X = df_encoded.drop('class', axis=1) # Features
y = df_encoded['class']              # Target

# 3. Calculate Mutual Information (Relationship Strength)
# discrete_features=True because our data is categorical
mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
mi_results = pd.Series(mi_scores, name="MI Scores", index=X.columns).sort_values(ascending=False)

# 4. Visualize the results
plt.figure(figsize=(10, 8))
sns.barplot(x=mi_results.values, y=mi_results.index, palette='viridis')
plt.title("Relationship Strength (Mutual Information) with Mushroom Class")
plt.xlabel("Strength of Relationship (0 to 1)")
plt.ylabel("Mushroom Features")
plt.savefig('feature_importance.png')

# 5. Print top features
print("Top 5 Most Related Features:")
print(mi_results.head(5))



# Advanced Data Visualization (Correlation Analysis)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('mushrooms.csv')

# 1. Transform categorical text into numbers so we can do math
# This is necessary because correlation formulas require numerical input
df_encoded = df.apply(LabelEncoder().fit_transform)

# 2. Calculate the correlation matrix
# This compares every column against every other column
corr_matrix = df_encoded.corr()

# 3. Create the Heatmap
plt.figure(figsize=(16, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Mushroom Feature Correlation Heatmap")
plt.savefig('correlation_heatmap.png')