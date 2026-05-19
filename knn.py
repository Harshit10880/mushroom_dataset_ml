from common_imports import *
from python import *

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train, y_train)

pridt_knn = model_knn.predict(X_test)
accuracy = accuracy_score(y_test, pridt_knn)
print(f"Model Accuracy: {accuracy * 100:.2f}%")