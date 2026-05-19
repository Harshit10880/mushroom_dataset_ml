from common_imports import *
from python import *

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)

predt_dt = model_dt.predict(X_test)
accuracy = accuracy_score(y_test, predt_dt)
print(f"Model Accuracy: {accuracy * 100:.2f}%")