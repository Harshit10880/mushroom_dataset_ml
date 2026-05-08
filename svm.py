best_features_svm = ['odor', 'spore-print-color', 'gill-color', 'ring-type', 'stalk-surface-above-ring']

# Encode Target with a unique variable name
le_svm = LabelEncoder()
y_svm = le_svm.fit_transform(df['class'])

# One-Hot Encode Features with a unique variable name
X_encoded_svm = pd.get_dummies(df[best_features_svm], dtype=int)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_encoded_svm, y_svm, test_size=0.2, random_state=42
)

# Initialize SVM with a unique model name
svm_model_final = SVC(kernel='linear', random_state=42)

svm_model_final.fit(X_train_svm, y_train_svm)
y_pred_svm_final = svm_model_final.predict(X_test_svm)

# Calculate Accuracy
final_svm_acc = accuracy_score(y_test_svm, y_pred_svm_final)
print(f"SVM Accuracy with Top Features: {final_svm_acc * 100:.2f}%")

# compare actual value with predicted value
results_df_svm = pd.DataFrame({'Actual': y_test_svm, 'Predicted': y_pred_svm_final})
display(results_df_svm.head(20))

# confusion matrix for svm

cm_svm = confusion_matrix(y_test_svm, y_pred_svm_final)

plt.figure(figsize=(7,5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Edible', 'Poisonous'], 
            yticklabels=['Edible', 'Poisonous'])
plt.xlabel('Predicted by SVM')
plt.ylabel('Actual Truth')
plt.title('SVM Classification Confusion Matrix')
plt.show()

print(classification_report(y_test_svm, y_pred_svm_final, 
                            target_names=['Edible', 'Poisonous']))