from sklearn.preprocessing import StandardScaler
import pandas as pd 

df=pd.read_csv('merged_final.csv')
print(df)
cols_to_scale = df.columns[1:-1]
scaler=StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

#scaled_data=scaler.fit_transform(df)
#scaled_df= pd.DataFrame(scaled_data, columns=df.columns)

print(df)
#df.to_csv('z_final.csv',index=False)


from imblearn.over_sampling import SMOTE
import pandas as pd

# Assuming you have a DataFrame 'df' with features and a target variable 'target'
df_features = df.drop(columns=['Condition','Gene_id'])
target = df['Condition']

# Create a SMOTE object
smote = SMOTE()

# Upsample the data
X_resampled, y_resampled = smote.fit_resample(df_features, target)

# Reconstruct the DataFrame with upsampled data
df_upsampled = pd.DataFrame(X_resampled, columns=df_features.columns)
df_upsampled['target'] = y_resampled
print(df_upsampled)
#df_upsampled.to_csv('upsampled.csv',index=False)

from boruta import BorutaPy
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from pymrmr import mRMR
import pandas as pd

# Assuming df contains your DataFrame with features and target variable
df_features1 = df_upsampled.drop(columns=['target'])
target = df_upsampled['target']

# Step 1: Perform Boruta feature selection
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2)
boruta_selector.fit(df_features1.values, target.values.astype(int))  # Use int here
boruta_features = df_features1.columns[boruta_selector.support_].tolist()

# Step 2: Perform RFE feature selection
rfe_selector = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=len(boruta_features))
rfe_selector.fit(df_features1.values, target.values)
rfe_features = df_features1.columns[rfe_selector.support_].tolist()

# Step 3: Perform mRMR feature selection
mrmr_features = mRMR(df_features1, 'MIQ', len(boruta_features))

# Step 4: Extract common features
common_features = set(boruta_features) & set(rfe_features) & set(mrmr_features)

print("Common Features:", common_features)
