import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import dill

def run():
    # Make title
    st.title("Travel Package Popularity Forecasting")
    st.image("deployment/image.jpg")
    st.markdown("## Background")
    st.markdown("""The travel industry is highly influenced by seasonal trends and shifting consumer preferences. For a travel agency, staying ahead of these fluctuations is critical for optimizing marketing budgets and operational resources.
                To maintain a competitive edge, the company requires a data-driven approach to understand the historical demand for destinations like Pari Island and anticipate how its popularity will evolve in the future.
                """)

    st.markdown("## Objective")
    st.markdown("""The primary goal of this project is to forecast the popularity of tour packages to Pari Island for the next 12 months.
                By utilizing Time-Series Analysis algorithms—such as the ARIMA/SARIMAX model currently being developed—this project aims to provide accurate projections that will assist the company in strategic planning and decision-making for the upcoming year.
                """)
    
    # Data Preparation
    data = pd.read_csv('deployment/cleaned_data.csv')
    data['date'] = pd.to_datetime(data['date']) 
    data.set_index('date', inplace=True)

    st.markdown("## Dataset")
    fig, ax = plt.subplots(figsize=(20, 10))
    data['value'].plot(ax=ax, title='Popularity', fontsize=15)
    st.pyplot(fig)

    # Load Model
    with open('deployment/final_model.pkl', 'rb') as a:
        final_model = dill.load(a)

    # Define Forecast Period
    # period=52
    with st.form('form_input'):
        period_input = st.number_input('Input Period (week)', value=0, step=1)
        submit_btn = st.form_submit_button('Predict')

    if submit_btn:
        period = period_input
        forecast = final_model.forecast(steps=period)
        
        # Define date for forecasted values
        last_date = data.index.max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                    periods=period, freq='W')
        # Set index
        forecast.index = future_dates
        st.markdown("## Forecasted Data")
        st.dataframe(forecast)

        # Concate data
        df_forecast = pd.concat([data, forecast.rename('forecast')], axis=1)

        # Visualization
        st.markdown("## Forecast Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_forecast.index, df_forecast['value'], label='Actual', color='blue')
        ax.plot(forecast.index, forecast, label='Forecast', color='orange', linestyle='--')

        plt.title('Weekly Forecast')
        plt.xlabel('Date')
        plt.ylabel('Hits')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)

run()
# streamlit run deployment/streamlit_app.py