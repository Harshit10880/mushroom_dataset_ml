from common_imports import *
from python import *

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

predt_nv = model_nb.predict(X_test)
accuracy = accuracy_score(y_test, predt_nv)
print(f"Model Accuracy: {accuracy * 100:.2f}%")