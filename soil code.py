# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")
crops
# Check for crop types
crops['crop'].unique()
# Split data
X = crops.drop('crop', axis=1)
y = crops['crop']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

features_dict = {}

for feature in ['N', 'P', 'K', 'ph']:
    log_reg = LogisticRegression(multi_class='multinomial')
    # Corrected indexing for pandas DataFrame
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    # Updated to include the 'average' parameter with a value suitable for multiclass classification
    f1_score = metrics.f1_score(y_test, y_pred, average='weighted')
    features_dict[feature] = f1_score
    print(f'F1 Score for {feature}: {f1_score}')
    
    best_predictive_feature = {'F1 Score': features_dict}
best_predictive_feature = pd.DataFrame(best_predictive_feature).reset_index()
best_predictive_feature.columns = ['Variables', 'F1 Score']
print(f"F1 score of the best predictive feature is {round(best_predictive_feature['F1 Score'].max(), 3)}.")
