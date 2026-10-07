#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("knn_dry_bean_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("dry_bean_selected_features.csv")

package = load_model()
df = load_data()

model = package["model"]
scaler = package["scaler"]
le = package["label_encoder"]
features = package["features"]
best_k = package["best_k"]

st.title("🌱 Dry Bean Classification")
st.write("K-Nearest Neighbors (KNN) Classification")

st.info(
    f"Selected Features: {len(features)} | Best K: {best_k}"
)

st.subheader("Select Input Method")

input_method = st.radio(
    "Choose how to provide bean measurements:",
    ["Enter Manually", "Use Real Dataset Example"]
)

if input_method == "Use Real Dataset Example":

    selected_class = st.selectbox(
        "Select Bean Class",
        sorted(df["Class"].unique())
    )

    class_data = df[df["Class"] == selected_class]

    sample_number = st.number_input(
        "Sample Number",
        min_value=1,
        max_value=len(class_data),
        value=1,
        step=1
    )

    sample = class_data.iloc[sample_number - 1]

    perimeter = float(sample["Perimeter"])
    aspect_ratio = float(sample["AspectRation"])
    roundness = float(sample["roundness"])
    shape_factor4 = float(sample["ShapeFactor4"])
    solidity = float(sample["Solidity"])
    extent = float(sample["Extent"])

    st.write("### Real Dataset Values")

    st.dataframe(
        pd.DataFrame([{
            "Perimeter": perimeter,
            "AspectRation": aspect_ratio,
            "roundness": roundness,
            "ShapeFactor4": shape_factor4,
            "Solidity": solidity,
            "Extent": extent
        }]),
        use_container_width=True
    )

    actual_class = selected_class

else:

    st.subheader("Enter Bean Measurements")

    perimeter = st.number_input(
        "Perimeter",
        min_value=0.0,
        value=500.0,
        format="%.4f"
    )

    aspect_ratio = st.number_input(
        "Aspect Ratio",
        min_value=0.0,
        value=1.5,
        format="%.4f"
    )

    roundness = st.number_input(
        "Roundness",
        min_value=0.0,
        value=0.8,
        format="%.4f"
    )

    shape_factor4 = st.number_input(
        "Shape Factor 4",
        min_value=0.0,
        value=0.99,
        format="%.4f"
    )

    solidity = st.number_input(
        "Solidity",
        min_value=0.0,
        value=0.98,
        format="%.4f"
    )

    extent = st.number_input(
        "Extent",
        min_value=0.0,
        value=0.75,
        format="%.4f"
    )

    actual_class = None

if st.button(
    "🔍 Predict Bean Class",
    use_container_width=True
):

    input_data = pd.DataFrame([{
        "Perimeter": perimeter,
        "AspectRation": aspect_ratio,
        "roundness": roundness,
        "ShapeFactor4": shape_factor4,
        "Solidity": solidity,
        "Extent": extent
    }])

    input_data = input_data[features]

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    predicted_class = le.inverse_transform(prediction)[0]

    st.success(
        f"Predicted Bean Class: **{predicted_class}**"
    )

    if actual_class is not None:

        if predicted_class == actual_class:
            st.success(
                f"Correct Prediction ✓ | Actual: {actual_class}"
            )
        else:
            st.error(
                f"Incorrect Prediction | Actual: {actual_class}"
            )

    st.subheader("Input Data")

    st.dataframe(
        input_data,
        use_container_width=True
    )


# In[ ]:




