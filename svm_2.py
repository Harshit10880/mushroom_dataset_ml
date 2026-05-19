# trying it with other features
from common_imports import *  
from python import * 

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




# CHECK FOR SUPPOER VECTORS

features_viz = ['odor', 'spore-print-color', 'gill-color', 'ring-type', 'stalk-surface-above-ring']

y_viz = LabelEncoder().fit_transform(df['class'])
X_encoded_viz = pd.get_dummies(df[features_viz], dtype=int)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_encoded_viz)

svm_viz = SVC(kernel='linear', random_state=42)
svm_viz.fit(X_pca, y_viz)

# Create a Mesh Grid to draw the decision boundary
h = .02  # step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict across the entire mesh grid
Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(10, 7))
# Draw the decision boundary (the "Hyperplane" area)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)

# Plot the individual mushrooms (Red=Poisonous, Blue=Edible)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_viz, cmap=plt.cm.coolwarm, edgecolors='k', s=30)

# CIRCLE THE SUPPORT VECTORS
plt.scatter(svm_viz.support_vectors_[:, 0], svm_viz.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='black', label='Support Vectors')

plt.title('SVM Decision Boundary & Support Vectors (PCA Reduced 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

print(f"Number of Support Vectors found: {len(svm_viz.support_vectors_)}")