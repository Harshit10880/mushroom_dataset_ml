# EDA ( exploratory data analysis )
import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split



df = pd.read_csv("mushrooms.csv")
df.head(5)

df.isnull().sum()

df.info()

df.duplicated().sum()

# First phase

# Feature Association & Selection
# 2. Encode categorical data to numbers for the math engine
df_encoded = df.apply(LabelEncoder().fit_transform)

X = df_encoded.drop('class', axis=1) # Features
y = df_encoded['class']              # Target

# 3. Calculate Mutual Information (Relationship Strength)
# discrete_features=True because our data is categorical
# see the corelation as a score from 0 to 1, where 1 means a perfect relationship and 0 means no relationship

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


# Second things

# Advanced Data Visualization (Correlation Analysis)
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




# Data preprocessing
selected_features = ['odor', 'spore-print-color', 'gill-color', 'ring-type', 'stalk-surface-above-ring']
target = 'class'

# Create a new dataframe with only your chosen columns
df_subset = df[selected_features + [target]].copy()

le = LabelEncoder()
df_subset['class'] = le.fit_transform(df_subset['class'])
df_subset['class']

# 3. One-Hot Encode the Features
# This creates separate 0/1 columns for each smell, color, etc.
df_final = pd.get_dummies(df_subset, columns=selected_features, dtype=int)

print("New Dataset Shape:", df_final.shape)
print("\nFirst 5 rows of the transformed data:")
print(df_final.head())




# Train split data
X = df_final.drop('class', axis=1)
y = df_final['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total mushrooms: {len(df_final)}")
print(f"Training set size: {len(X_train)} (Questions for the model)")
print(f"Testing set size: {len(X_test)} (The Surprise Quiz)")






# Logistic regression problem

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



model = LogisticRegression(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# confusion matrix phase

cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Edible', 'Poisonous'],
            yticklabels=['Edible', 'Poisonous'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Mushroom Classification Confusion Matrix')
plt.show()


print("\n--- Detailed Performance Report ---")
print(classification_report(y_test, y_pred, target_names=['Edible', 'Poisonous']))