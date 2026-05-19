import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
# logisticRegression problem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# implement confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import joblib
# implement support vector algo
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier