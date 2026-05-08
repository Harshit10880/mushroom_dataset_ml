# trying it with other features

new_features_svm = ['gill-size', 'population', 'bruises', 'habitat', 'stalk-root']

le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

X_feature_new_svm = pd.get_dummies(df[new_features_svm], dtype=int)
y_new_svm = df['class']

X_train_new_svm, X_test_new_svm, y_train_new_svm, y_test_new_svm = train_test_split(
    X_feature_new_svm, y_svm, test_size=0.2, random_state=42
)

svm_new_model_final = SVC(kernel='linear', random_state=42)

svm_new_model_final.fit(X_train_new_svm, y_train_new_svm)
y_pred_new_svm_final = svm_new_model_final.predict(X_test_new_svm)

# Calculate Accuracy
final_svm_acc_new = accuracy_score(y_test_new_svm, y_pred_new_svm_final)
print(f"SVM Accuracy with Top Features: {final_svm_acc_new * 100:.2f}%")

# confusion matrix

cm_svm_new = confusion_matrix(y_test_new_svm, y_pred_new_svm_final)

plt.figure(figsize=(7,5))
sns.heatmap(cm_svm_new, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Edible', 'Poisonous'], 
            yticklabels=['Edible', 'Poisonous'])
plt.xlabel('Predicted by SVM')
plt.ylabel('Actual Truth')
plt.title('SVM Classification Confusion Matrix')
plt.show()

print(classification_report(y_test_new_svm, y_pred_new_svm_final, 
                            target_names=['Edible', 'Poisonous']))