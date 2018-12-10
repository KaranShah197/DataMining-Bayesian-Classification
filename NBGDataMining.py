import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

# Importing dataset
data = pd.read_csv("bankdataCD.csv", encoding = 'utf-8')

#CLEAN DATA
#converts marital status descriptions - married =1 single = 0
data["marital_cleaned"] = np.where(data["marital"] == "single", 0, 1)

#converts loan, default, housing status descriptions - yes =1 no = 0

data["default_cleaned"]=np.where(data["default"] == "no", 0, 1)
data["housing_cleaned"]=np.where(data["housing"] == "no", 0, 1)
data["loan_cleaned"]=np.where(data["loan"]=="no", 0, 1)

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
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                      )

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
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome_cleaned",
        "y_cleaned"
    ]
].dropna(axis = 0, how = 'any')

#data.to_csv(r'~/Desktop/NBGMining.csv', index=None, sep=',', mode='a')
# Split dataset in training and test datasets
X_train, X_test = train_test_split(data, test_size = 0.3, random_state = int(time.time()))

# Up sample true cases
print(X_train['y_cleaned'].value_counts())
data_y = X_train[X_train['y_cleaned']==1]
data_n = X_train[X_train['y_cleaned']==0]
data_y_upsampled = resample(data_y, replace=True,n_samples=27948,random_state=123)
populatedData = pd.concat([data_n,data_y_upsampled])
print(populatedData['y_cleaned'].value_counts())

# Instantiate the classifier
gnb = GaussianNB()
used_features = [
    "age",
    "job_cleaned",
    "marital_cleaned",
    "education_cleaned",
    "default_cleaned",
    "balance"
]

# Train classifier
gnb.fit(X_train[used_features].values, X_train["y_cleaned"] )
y_pred = gnb.predict(X_test[used_features] )

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0], # Total number of test_data
          (X_test["y_cleaned"] != y_pred).sum(), # Total number of misslabled data
          100*(1-(X_test["y_cleaned"] != y_pred).sum()/X_test.shape[0]) # Accuracy
))

tn, fp, fn, tp = confusion_matrix(X_test['y_cleaned'], y_pred).ravel()
print('tn: {}, fp: {}\nfn: {}, tp: {}'.format(tn,fp,fn,tp))

print('Populated result')
# Train classifier
gnb.fit(
    populatedData[used_features].values,
    populatedData["y_cleaned"]
)
y_pred = gnb.predict(X_test[used_features])

print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0], # Total number of test_data
          (X_test["y_cleaned"] != y_pred).sum(), # Total number of misslabled data
          100*(1-(X_test["y_cleaned"] != y_pred).sum()/X_test.shape[0]) # Accuracy
))

tn, fp, fn, tp = confusion_matrix(X_test['y_cleaned'], y_pred).ravel()
print('tn: {}, fp: {}\nfn: {}, tp: {}'.format(tn,fp,fn,tp))