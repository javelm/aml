# Importing the libraries
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PowerTransformer, StandardScaler
import utility
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pickle

# Importing the dataset
df = pd.read_csv(r'C:\Users\zabbix_automation\Desktop\Crop_recommendation.csv')

""" Data Preprocessing """
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Crop' column
df['Crop_encoded'] = label_encoder.fit_transform(df['label'])

encoded_labels = pd.DataFrame({
    'Original Label': label_encoder.classes_,
    'Encoded Value': label_encoder.transform(label_encoder.classes_)
})

df = df.drop(columns=['label'])

# Rename the encoded column if desired
df = df.rename(columns={'Crop_encoded': 'label'})

# Remove outliers
clean_df = utility.remove_outliers_iqr(df)

# Apply Power Transformation
pt = PowerTransformer()
clean_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']] = pt.fit_transform(
    clean_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
)

# Assuming df is the original DataFrame with all necessary columns
scaler = StandardScaler()
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Check if the columns in `df` match those in `features`
assert all(col in clean_df.columns for col in features), "Feature columns are missing in the DataFrame"

# Apply scaling
clean_df[features] = scaler.fit_transform(clean_df[features])

""" Split the Dataset to Test and Train Set """

# Case 1: 80% Train, 20% Test
X = clean_df.drop('label', axis=1)
y = clean_df['label']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

# Case 2: 10% Train, 90% Test
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.9, random_state=42)


""" Build Model """
model_lr = LogisticRegression()
model_lr.fit(X_train1, y_train1)

# Example MLE model (using a probabilistic model)

model_mle = GaussianNB()
model_mle.fit(X_train1, y_train1)

# Any other model (e.g., Random Forest)

model_rf = RandomForestClassifier()
model_rf.fit(X_train1, y_train1)

y_pred_lr = model_lr.predict(X_test1)


# Saving model to disk
try:
    pickle.dump(model_lr, open('model.pkl','wb'))
except Exception as e:
    print(str(e))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
# print(model.predict([[88,40,41,25.09865, 85.12345, 7.01010, 200.09854323]]))