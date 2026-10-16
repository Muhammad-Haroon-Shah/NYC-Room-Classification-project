
import streamlit as st
import requests


# Page configuration
st.set_page_config(
    page_title="NYC Room Classification",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 NYC Room Classification")
st.write("Enter the Airbnb listing details to predict the room type.")


# Input fields
latitude = st.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=40.7128
)

longitude = st.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-74.0060
)

price = st.number_input(
    "Price per night",
    min_value=0.01,
    value=100.0
)

minimum_nights = st.number_input(
    "Minimum nights",
    min_value=1,
    max_value=365,
    value=1,
    step=1
)

number_of_reviews = st.number_input(
    "Number of reviews",
    min_value=0,
    value=10,
    step=1
)

reviews_per_month = st.number_input(
    "Reviews per month",
    min_value=0.0,
    value=1.0
)

calculated_host_listings_count = st.number_input(
    "Host listings count",
    min_value=0,
    value=1,
    step=1
)

availability_365 = st.number_input(
    "Availability (365 days)",
    min_value=0,
    max_value=365,
    value=100,
    step=1
)

neighbourhood_group = st.text_input(
    "Neighbourhood Group",
    value="Manhattan"
)

neighbourhood = st.text_input(
    "Neighbourhood",
    value="Harlem"
)


# Prediction button
if st.button("Predict Room Type"):

    data = {
        "latitude": latitude,
        "longitude": longitude,
        "price": price,
        "minimum_nights": minimum_nights,
        "number_of_reviews": number_of_reviews,
        "reviews_per_month": reviews_per_month,
        "calculated_host_listings_count": calculated_host_listings_count,
        "availability_365": availability_365,
        "neighbourhood_group": neighbourhood_group,
        "neighbourhood": neighbourhood
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                f"Predicted Room Type: {result['Predicted_room_type']}"
            )

            st.subheader("Prediction Probabilities")

            st.write(result["Probability"])

        else:
            st.error(
                f"API Error: {response.status_code}"
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to FastAPI. "
            "Make sure your FastAPI server is running."
        )

