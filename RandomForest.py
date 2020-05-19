import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Importing dataset
data = pd.read_csv("~/Desktop/bankdataCD.csv", encoding = 'utf-8')

#CLEAN DATA
#converts marital status descriptions - married = 1 single = 0
data["marital_cleaned"] = np.where(data["marital"] == "single", 1, np.where(data["marital"] == "married", 2, 3))

#converts loan, default, housing status descriptions - yes =1 no = 0

data["default_cleaned"] = np.where(data["default"] == "no", 0, 1)
data["housing_cleaned"] = np.where(data["housing"] == "no", 0, 1)
data["loan_cleaned"]    = np.where(data["loan"] == "no", 0, 1)

#converts education description | unknown =0, | primary=1 | secondary =2 | teritiary =4 | anything else =5
data["education_cleaned"] = np.where(data["education"] == "unknown", 0,
                                np.where(data["education"] == "primary", 1,
                                    np.where(data["education"] == "secondary", 2,
                                        np.where(data["education"] == "tertiary", 3, 4)
                                    )
                                )
                            )

#converts job description into numerical values
data["job_cleaned"] = np.where(data["job"] == "unknown", 0,
                        np.where(data["job"] == "technician", 1,
                            np.where(data["job"] == "entrepreneur", 2,
                                np.where(data["job"] == "blue-collar", 3,
                                    np.where(data["job"] == "management", 4,
                                        np.where(data["job"] == "retired", 5,
                                            np.where(data["job"] == "admin.", 6,
                                                np.where(data["job"] == "services", 7,
                                                    np.where(data["job"] == "self-employed", 8,
                                                        np.where(data["job"] == "unemployed", 9,
                                                            np.where(data["job"] == "housemaid", 10,
                                                                np.where(data["job"] == "student", 11, 12)
                        )))))))))))

#converts poutcome description | failure =0, | success=1 | unknown=2 | anything else =3
data["poutcome_cleaned"] = np.where(data["poutcome"] == "unknown", 0,
                                np.where(data["poutcome"] == "success", 1,
                                    np.where(data["poutcome"] == "unknown", 2, 3)
                                )
                           )

data["y_cleaned"] = np.where(data["y"] == "no", 0, 1)

# Cleansed dataset of bankCD
data = data[
    [
        "age",
        "job_cleaned",
        "marital_cleaned",
        "education_cleaned",
        "default_cleaned",
        "balance",
        "housing_cleaned",
        "loan_cleaned",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome_cleaned",
        "y_cleaned"
    ]
].dropna(axis = 0, how = 'any')


# Import train_test_split function
X = data[['loan_cleaned', 'education_cleaned', 'marital_cleaned', 'default_cleaned', 'housing_cleaned']]
y = data['y_cleaned']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

#Import Random Forest Model
clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy of :",metrics.accuracy_score(y_test, y_pred))