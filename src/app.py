import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the pre-trained model
model = joblib.load('../Outputs/models/kmeans_model.pkl')

# Load the dataset
raw_data= pd.read_csv('../data/Train.csv')
X_train=joblib.load('../data/preprocessed_data/X_train_norm.pkl')

y_kmeans=model.fit_predict(X_train)
X_train['Cluster'] = y_kmeans

st.set_page_config(page_title="Customer Segmentation", page_icon=":bar_chart:", layout="wide")
st.title("Customer Segmentation Dashboard")


#Displaying the raw data
st.subheader("Raw Data")
st.write("This is the raw data used for clustering.")
st.dataframe(raw_data)

# Showing data after clustering
st.subheader("Data After Clustering")
st.write("This is the data after clustering.")
st.dataframe(X_train)
st.write("The data has been clustered into {} clusters.".format(len(X_train['Cluster'].unique())))

# Displaying the silhouette score
st.subheader("Silhouette Score")
st.write("The silhouette score is a measure of how similar an object is to its own cluster compared to other clusters.")
st.write("The silhouette score for this model is: {:.2f}".format(silhouette_score(X_train, y_kmeans)))


# Correlation heatmap
corr="../Outputs/plots/correlation_matrix.png"
st.subheader("Correlation Heatmap")
st.write("This is the correlation heatmap of the data.")
st.image(corr, caption='Correlation Heatmap', use_container_width=True)
st.write("The correlation heatmap shows the correlation between different features in the dataset. The darker the color, the stronger the correlation.")
st.write("Based on the above heatmap, the following features have been selected for clustering:")
st.markdown("- Age")
st.markdown("- Profession")
st.markdown("- Graduated")
st.markdown("- Ever Married")
st.markdown("- Family Size")
st.markdown("- Spending Score")


# Displaying the countplot
cp="../Outputs/plots/countplot_spending_score.png"
st.subheader("Countplot of Spending Score")
st.write("This is the countplot of the spending score in each segment.")
st.image(cp, caption='Countplot of Spending Score', use_container_width=True)
st.write("The countplot shows the distribution of spending score in each segment. The spending score is a measure of how much a customer spends on average.")
st.write("The above countplot gives us an idea of how the spending score is distributed in each segment. The spending score is a measure of how much a customer spends on average. The higher the spending score, the more a customer spends.")
st.markdown("**Segment was encoded as follows:**")
st.markdown("- 0: A")
st.markdown("- 1: B")
st.markdown("- 2: C")
st.markdown("- 3: D")
st.markdown("**Spending Score was encoded as follows:**")
st.markdown("- 0: Low Spending Score")
st.markdown("- 1: Medium Spending Score")
st.markdown("- 2: High Spending Score")



# Side bar
st.sidebar.title("Customer Segmentation Dashboard")
st.sidebar.subheader("Select a Graph to be displayed below after clustering")
graph = st.sidebar.selectbox("Select Graph", ("Age vs Spending Score", "Profession vs Spending Score", "Graduated vs Spending Score", "Ever Married vs Spending Score", "Family_Size vs Spending Score"))


if graph == "Age vs Spending Score":
    st.subheader("Age vs Spending Score")
    st.write("This graph shows the distribution of Age and Spending Score.")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X_train, x='Age', y='Spending_Score', hue='Cluster', palette='Set1')
    plt.title("Age vs Spending Score")
    plt.xlabel("Age")
    plt.ylabel("Spending Score")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

elif graph == "Profession vs Spending Score":
    st.subheader("Profession vs Spending Score")
    st.write("This graph shows the distribution of Profession and Spending Score.")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X_train, x='Profession', y='Spending_Score', hue='Cluster', palette='Set1')
    plt.title("Profession vs Spending Score")
    plt.xlabel("Profession")
    plt.ylabel("Spending Score")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
elif graph == "Graduated vs Spending Score":
    st.subheader("Graduated vs Spending Score")
    st.write("This graph shows the distribution of Graduated and Spending Score.")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X_train, x='Graduated', y='Spending_Score', hue='Cluster', palette='Set1')
    plt.title("Graduated vs Spending Score")
    plt.xlabel("Graduated")
    plt.ylabel("Spending Score")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
elif graph == "Ever Married vs Spending Score":
    st.subheader("Ever Married vs Spending Score")
    st.write("This graph shows the distribution of Ever Married and Spending Score.")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X_train, x='Ever_Married', y='Spending_Score', hue='Cluster', palette='Set1')
    plt.title("Ever Married vs Spending Score")
    plt.xlabel("Ever Married")
    plt.ylabel("Spending Score")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)
elif graph == "Family_Size vs Spending Score":
    st.subheader("Family Size vs Spending Score")
    st.write("This graph shows the distribution of Family Size and Spending Score.")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X_train, x='Family_Size', y='Spending_Score', hue='Cluster', palette='Set1')
    plt.title("Family Size vs Spending Score")
    plt.xlabel("Family Size")
    plt.ylabel("Spending Score")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)



# Finding Cluster Summary
cluster_summary = X_train.groupby("Cluster").mean()
scaler = joblib.load('../data/preprocessed_data/scaler.pkl')
cluster_summary_original = pd.DataFrame(
    scaler.inverse_transform(cluster_summary).round(0).astype(int),
    columns=cluster_summary.columns,
    index=cluster_summary.index
)

# Loading all encoders for reversing the encoding

pr = joblib.load('../data/preprocessed_data/profession_encoder.pkl')
em= joblib.load('../data/preprocessed_data/married_encoder.pkl')
gr = joblib.load('../data/preprocessed_data/graduated_encoder.pkl')
ss = joblib.load('../data/preprocessed_data/spending_score_encoder.pkl')



cluster_data=pd.DataFrame(cluster_summary_original)


# Inverse transforming the encoded columns
cluster_data['Profession'] = pr.inverse_transform(cluster_data['Profession'].astype(int))
cluster_data['Ever_Married'] = em.inverse_transform(cluster_data['Ever_Married'].astype(int))
cluster_data['Graduated'] = gr.inverse_transform(cluster_data['Graduated'].astype(int))
cluster_data['Spending_Score'] = ss.inverse_transform(cluster_data['Spending_Score'].astype(int))


# Displaying the cluster summary in a dataframe
st.subheader("Cluster Summary")
st.write("This is the summary of each cluster.")
st.dataframe(cluster_data)


# Conclusions
st.subheader("Conclusions")
st.markdown("- People who spend the least are married, aged around 50, are academics, and have a family size of 3.")
st.markdown("- People who have an average expenditure are either single, aged around 30-45, and work in healthcare or entertainment.")


st.markdown(" ##### For more details about data analysis and model training, please refer to the exploratory_data_analysis.ipynb and train_model.ipynb in this [GitHub repository](https://github.com/Saber0722/PCA_MNIST)")

