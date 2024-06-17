#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px 
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from datetime import date


# In[2]:


transcript=pd.read_csv(r"C:\Users\welcome\Desktop\Data Sets\transcript.csv")
profile=pd.read_csv(r"C:\Users\welcome\Desktop\Data Sets\profile.csv", parse_dates=['became_member_on'])
portfolio=pd.read_csv(r"C:\Users\welcome\Desktop\Data Sets\portfolio.csv")


# In[3]:


transcript=transcript.drop_duplicates()
profile=profile.drop_duplicates()
portfolio=portfolio.drop_duplicates()


# In[4]:


transcript.info()


# In[5]:


transcript.head(10)


# In[6]:


profile.info()


# In[7]:


profile.head(10)


# In[8]:


portfolio.info()


# In[9]:


portfolio.head(10)


# In[10]:


# Display the count of unique values for each column
table = []
for col in transcript.columns:
    row = [col, transcript[col].nunique()]
    table.append(row)
print(tabulate(table, headers=['Column', 'Unique Values']))


# In[11]:


# Drop unnecessary columns
transcript = transcript.drop("Unnamed: 0", axis=1)


# In[12]:


transcript['event'].value_counts()


# In[13]:


transcript.describe()


# In[14]:


# Display the count of unique values for each column
table = []
for col in profile.columns:
    row = [col, profile[col].nunique()]
    table.append(row)
print(tabulate(table, headers=['Column', 'Unique Values']))


# In[15]:


# Drop unnecessary columns
profile = profile.drop("Unnamed: 0", axis=1)


# In[16]:


# Display the count of unique values for each column
profile['gender'].value_counts()


# In[17]:


profile.describe()


# In[18]:


# Display the count of unique values for each column
table = []
for col in portfolio.columns:
    row = [col, portfolio[col].nunique()]
    table.append(row)
print(tabulate(table, headers=['Column', 'Unique Values']))


# In[19]:


# Drop unnecessary columns
portfolio = portfolio.drop("Unnamed: 0", axis=1)


# In[20]:


# Display the count of unique values for each column
table = []
for col in portfolio.columns:
    row = [col, portfolio[col].value_counts()]
    table.append(row)
print(tabulate(table, headers=['Column', 'Unique Values']))


# In[21]:


portfolio.describe()


# In[22]:


profile.isna().sum()


# In[23]:


2175/17000


# In[24]:


# profile['gender']=profile['gender'].fillna('U')


# In[25]:


# profile['income']=profile['income'].fillna(profile['income'].mean())


# In[26]:


profile.dropna()


# In[27]:


# # Plot histograms for numerical columns
# # for col in transcript.select_dtypes(include="number").columns:
#     sns.histplot(x=col, data=transcript, kde=True)
#     plt.show()


# In[28]:


# Plot histograms for numerical columns
for col in profile.select_dtypes(include="number").columns:
    sns.histplot(x=col, data=profile, kde=True)
    plt.show()


# In[29]:


profile[profile['age']>100]


# In[30]:


# Plot histograms for numerical columns
for col in portfolio.select_dtypes(include="number").columns:
    sns.countplot(x=col, data=portfolio)
    plt.show()


# In[31]:


# Plot histograms for numerical columns
for col in portfolio.select_dtypes(include="object").columns:
    sns.countplot(x=col, data=portfolio)
    plt.show()


# In[32]:


for col in [profile['gender']]:
    sns.countplot(x=col, data=profile)
    plt.show()


# In[33]:


# Compute and visualize correlation matrix
profile.corr()
sns.heatmap(profile.corr(), annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)


# In[34]:


# Display percentage of outliers for each numerical column
table = []
Num_Columns=profile.select_dtypes(include="number")
for col in Num_Columns:
    IQR = profile[col].quantile(0.75) - profile[col].quantile(0.25)
    lower = profile[col].quantile(0.25) - 1.5 * IQR
    upper = profile[col].quantile(0.75) + 1.5 * IQR
    outlier = (profile[col] <= lower) | (profile[col] >= upper)
    outlier_percentage = (profile[outlier].shape[0] / profile.shape[0]) * 100
    row = [col, outlier_percentage]
    table.append(row)
print(tabulate(table, headers=['Column', 'Outlier Percentage']))


# In[35]:


table = []
Num_Columns=profile.select_dtypes(include="number")
for col in Num_Columns:
    IQR = profile[col].quantile(0.75) - profile[col].quantile(0.25)
    lower = profile[col].quantile(0.25) - 1.5 * IQR
    upper = profile[col].quantile(0.75) + 1.5 * IQR
    outlier = (profile[col] <= lower) | (profile[col] >= upper)
    profile.drop(profile[outlier].index, inplace=True)


# In[36]:


# # Display percentage of outliers for each numerical column
# table = []
# Num_Columns=portfolio.select_dtypes(include="number")
# for col in Num_Columns:
#     IQR = portfolio[col].quantile(0.75) - portfolio[col].quantile(0.25)
#     lower = portfolio[col].quantile(0.25) - 1.5 * IQR
#     upper = portfolio[col].quantile(0.75) + 1.5 * IQR
#     outlier = (portfolio[col] <= lower) | (portfolio[col] >= upper)
#     outlier_percentage = (portfolio[outlier].shape[0] /portfolio.shape[0]) * 100
#     row = [col, outlier_percentage]
#     table.append(row)
# print(tabulate(table, headers=['Column', 'Outlier Percentage']))


# In[37]:


#profile_portfolio_merged = pd.merge(profile, portfolio, on='id', how='inner')


# In[38]:


profile['year'] = profile['became_member_on'].dt.year
profile['month'] = profile['became_member_on'].dt.month
profile['day']=profile['became_member_on'].dt.day_name()
profile['member_since_how_manydays'] = (pd.to_datetime('today') - profile['became_member_on']).astype('timedelta64[D]').astype(int)


# In[39]:


profile['age_group'] = pd.cut(x=profile['age'], bins=[18, 20, 40, 60, 80, 101],
                    labels=['Teenage(0-19)', 'young(20-39)', 'Middle-age(40-59)',
                            'Old(60-79)', 'Very-Old(80-100)'], include_lowest=True)


# In[40]:


profile['income_group'] = pd.cut(x=profile['income'], bins=[0,50000, 60000, 70000, 80000, 90000, 100000, 200000],
                    labels=['<50k', '50k-60k', '60k-70k',
                            '70k-80k', '80k-90k', '90k-1L', '1L-2L'], include_lowest=True)


# In[41]:


for col in [profile['year'], profile['month'], profile['day'], profile['age_group'], profile['income_group']]:
    sns.countplot(x=col, data=profile)
    plt.show()


# In[42]:


sns.countplot(x=portfolio['channels'], data=profile)
plt.show()


# In[43]:


sns.countplot(x=transcript['event'], data=transcript)
plt.show()


# In[44]:


transcript['value'] = transcript['value'].str.replace('{offer id:', '')
transcript['value'] = transcript['value'].str.replace('{amount:', '')
transcript['value'] = transcript['value'].str.replace('}', '')
transcript['value'] = transcript['value'].str.replace("'", '')


# In[45]:


transcript['value'].value_counts()


# In[46]:


transcript.isna().sum()


# In[47]:


list(portfolio['id'].unique())


# In[48]:


transcript_offer_type=transcript[transcript['event'].isin(['offer received', 'offer viewed', 'offer completed'])]
portfolio_transcript = portfolio.merge(right=transcript_offer_type, how='left', left_on='id', right_on='value')
portfolio_transcript.head(150)


# In[49]:


# transcript_offer_type=transcript[transcript['event'].isin(['offer received', 'offer viewed', 'offer completed'])]
profile_transcript = profile.merge(right=transcript, how='inner', left_on='id', right_on='person')
profile_transcript.head()


# In[50]:


profile_transcript = profile_transcript.drop(["became_member_on", "income", "person"] , axis=1)


# In[51]:


profile_transcript['value'] = profile_transcript['value'].str.replace('{offer id:', '')
profile_transcript['value'] = profile_transcript['value'].str.replace('{amount:', '')
profile_transcript['value'] = profile_transcript['value'].str.replace('}', '')
profile_transcript['value'] = profile_transcript['value'].str.replace("'", '')


# In[52]:


profile_transcript_copy=profile_transcript


# In[53]:


profile_transcript.info()


# In[54]:


profile_transcript.head()


# In[55]:


profile_transcript_portfolio = profile_transcript.merge(right=portfolio, how='outer', left_on='value', right_on='id')


# In[56]:


profile_transcript_portfolio=profile_transcript_portfolio.dropna()
profile_transcript_portfolio.head(100)


# In[57]:


# Encode categorical variables and one-hot encode 
profile_transcript_encoded = profile_transcript
label_encoder = LabelEncoder()
Cat_Columns=profile_transcript.select_dtypes(include="object").columns
for col in Cat_Columns:
    profile_transcript_encoded[col] = label_encoder.fit_transform(profile_transcript[col])
profile_transcript_encoded = pd.get_dummies(profile_transcript_encoded)


# In[58]:


profile_transcript.head()


# In[59]:


profile_transcript_encoded.columns


# In[60]:


# Scale numerical variables using MinMaxScaler
profile_transcript_scaled = profile_transcript_encoded
scaler = MinMaxScaler()
Columns = profile_transcript_encoded.columns
for col in Columns:
    profile_transcript_scaled[col] = scaler.fit_transform(profile_transcript_encoded[[col]])


# In[61]:


profile_transcript_scaled.head()


# In[ ]:


# Visualize the elbow method to find optimal K in KMeans
plt.figure(figsize=(15, 5))
Elbow_M = KElbowVisualizer(KMeans(), k=(2, 15))
Elbow_M.fit(profile_transcript_scaled)
Elbow_M.show()


# In[ ]:


# Apply KMeans clustering with obtained K
model = KMeans(n_clusters=6)
model.fit(profile_transcript_scaled)
# profile_transcript_copy["cluster"] = model.labels_


# In[ ]:


profile_transcript_copy.head()


# In[ ]:


# Reduce dimensionality using PCA
pca = PCA(n_components=2)
embedding = pca.fit_transform(profile_transcript_scaled)
projection = pd.DataFrame(columns=['x', 'y'], data=embedding)
projection['Age_group'] = profile_transcript['age_group']
projection['Income'] = profile_transcript['income_group']
projection['cluster'] = model.labels_


# In[ ]:


# Visualize clusters using scatter plot
fig = px.scatter(projection, x='x', y='y', color=model.labels_, hover_data=['Age_group', 'Income', 'cluster'])
fig.show()


# In[ ]:


# profile_cluster_1=profile_transcript_copy[profile_transcript_copy["Cluster"]==1]


# In[ ]:


# profile_cluster_1.head(50)


# In[ ]:


# profile_cluster_1.groupby(['event'])['value'].count()


# In[ ]:


profile_transcript_copy["cluster"]=model.labels_


# In[ ]:


# Encode categorical variables and one-hot encode 
profile_transcript_portfolio_encoded = profile_transcript
label_encoder = LabelEncoder()
Cat_Columns=profile_transcript_portfolio.select_dtypes(include=["object","category"]).columns
for col in Cat_Columns:
    profile_transcript_portfolio_encoded[col] = label_encoder.fit_transform(profile_transcript_portfolio[col])
profile_transcript_portfolio_encoded = pd.get_dummies(profile_transcript_portfolio_encoded)


# In[ ]:


# Scale numerical variables using MinMaxScaler
profile_transcript_portfolio_scaled = profile_transcript_portfolio_encoded
scaler = MinMaxScaler()
Columns = profile_transcript_portfolio_encoded.columns
for col in Columns:
    profile_transcript_portfolio_scaled[col] = scaler.fit_transform(profile_transcript_portfolio_encoded[[col]])


# In[ ]:


profile_transcript_portfolio.head()


# In[ ]:


# Example: Analyzing clusters
# Assuming profile_transcript_copy already has 'cluster' labels from KMeans

# Group by cluster and analyze event counts
cluster_analysis = profile_transcript_copy.groupby('cluster')['event'].value_counts().unstack().fillna(0)
print(cluster_analysis)

# # Visualize event counts by cluster
# plt.figure(figsize=(10, 6))
# sns.heatmap(cluster_analysis, annot=True, cmap='Blues', fmt='g')
# plt.title('Event Counts by Cluster')
# plt.show()

# Example: Calculate offer completion rate by cluster
offer_completion_rate = profile_transcript_copy[profile_transcript_copy['event'] == 'offer completed'].groupby('cluster')['event'].count() / profile_transcript_copy.groupby('cluster')['event'].count()
print("Offer Completion Rate by Cluster:")
print(offer_completion_rate)

# Example: Analyze demographic distributions within clusters
sns.countplot(x='age_group', hue='cluster', data=profile_transcript_copy)
plt.title('Age Group Distribution by Cluster')
plt.show()

sns.countplot(x='income_group', hue='cluster', data=profile_transcript_copy)
plt.title('Income Group Distribution by Cluster')
plt.show()

# Example: Analyze offer types by cluster
offer_type_counts = profile_transcript_copy[profile_transcript_copy['event'].isin(['offer received', 'offer viewed', 'offer completed'])].groupby(['cluster', 'event'])['value'].count().unstack().fillna(0)
print("Offer Type Counts by Cluster:")
print(offer_type_counts)


# In[ ]:


# # Group by cluster and analyze event counts
# cluster_analysis = profile_transcript_portfolio.groupby('cluster')['event'].value_counts().unstack().fillna(0)
# print(cluster_analysis)

# # Visualize event counts by cluster
# plt.figure(figsize=(10, 6))
# sns.heatmap(cluster_analysis, annot=True, cmap='Blues', fmt='g')
# plt.title('Event Counts by Cluster')
# plt.show()
# # Calculate offer completion rate by cluster
# offer_completion_rate = profile_transcript_portfolio[profile_transcript_portfolio['event'] == 'offer completed'].groupby('cluster')['event'].count() / profile_transcript_portfolio.groupby('cluster')['event'].count()
# print("Offer Completion Rate by Cluster:")
# print(offer_completion_rate)
# # Analyze demographic distributions within clusters
# sns.countplot(x='age_group', hue='cluster', data=profile_transcript_portfolio)
# plt.title('Age Group Distribution by Cluster')
# plt.show()

# sns.countplot(x='income_group', hue='cluster', data=profile_transcript_portfolio)
# plt.title('Income Group Distribution by Cluster')
# plt.show()
# # Analyze offer types by cluster
# offer_type_counts = profile_transcript_portfolio[profile_transcript_portfolio['event'].isin(['offer received', 'offer viewed', 'offer completed'])].groupby(['cluster', 'event'])['value'].count().unstack().fillna(0)
# print("Offer Type Counts by Cluster:")
# print(offer_type_counts)


# In[ ]:


# Assuming profile_transcript_copy already has 'cluster' labels from KMeans

# Example: Define target variable
profile_transcript['completed_offer'] = np.where(profile_transcript_scaled['event'] == 'offer completed', 1, 0)

# Example: Define features for modeling
features = ['age', 'income_group', 'gender', 'value', 'cluster']

# Example: Split data into train and test sets
X = profile_transcript_portfolio_scaled[features]
y = profile_transcript_portfolio_scaled['completed_offer']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example: Train a classification model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Example: Predictions and evaluation
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Example: Feature importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
print(feature_importance.sort_values(by='Importance', ascending=False))

# Example: Optimize hyperparameters (Grid Search)
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Example: Retrain model with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Example: Predictions and evaluation with best model
y_pred_best = best_model.predict(X_test)
print("Accuracy with best model:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))


# In[ ]:





# In[ ]:




