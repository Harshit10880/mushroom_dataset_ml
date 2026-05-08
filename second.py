new_features = ['gill-size', 'population', 'bruises', 'habitat', 'stalk-root']
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

X_new = pd.get_dummies(df[new_features], dtype=int)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
new_model = LogisticRegression(random_state=42)
new_model.fit(X_train, y_train)
y_pred = new_model.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred)

print(f"New Feature Set Accuracy: {new_accuracy * 100:.2f}%")

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