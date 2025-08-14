import streamlit as st
import pandas as pd
import numpy as np
import joblib



# Load the model and preprocessors
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler_model.pkl')

# Streamlit app title
st.title("Customer Clustering App")
st.write("Enter customer spending data to predict their cluster.")

# Input fields for the features
fresh = st.number_input("Fresh Spending", min_value=0.0, value=1000.0)
milk = st.number_input("Milk Spending", min_value=0.0, value=1000.0)
grocery = st.number_input("Grocery Spending", min_value=0.0, value=1000.0)
frozen = st.number_input("Frozen Spending", min_value=0.0, value=1000.0)
detergents_paper = st.number_input("Detergents & Paper Spending", min_value=0.0, value=1000.0)
delicassen = st.number_input("Delicassen Spending", min_value=0.0, value=1000.0)

# Button to predict
if st.button("Predict Cluster"):
    # Prepare input data
    input_data = [[fresh, milk, grocery, frozen, detergents_paper, delicassen]]
    input_df = pd.DataFrame(input_data, columns=['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
    
    # Normalize and apply PCA (if used in training)
    input_scaled = scaler.transform(input_df)
    
    # input_pca = pca.transform(input_scaled)  # Uncomment if PCA was used in prediction
    
    # Predict cluster
    cluster = kmeans.predict(input_scaled)[0]
    
    # Display result
    st.success(f"The customer belongs to Cluster {cluster}")

# Optional: Display a note about clusters
st.write("Note: Clusters are based on spending patterns. Cluster 0 might represent high spenders on Fresh, Cluster 1 on Grocery, etc.")