import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.statespace.sarimax import SARIMAX

import os
import warnings
warnings.filterwarnings('ignore')
#cd /Users/priyamadhurigattem/Downloads/weather-data-analysis-dash-main
#streamlit run Streamlit_Dash.py
#Setting the title and page icon
st.set_page_config(page_title="WEATHER", page_icon=":sun_behind_rain_cloud:", layout="wide")
st.title(':partly_sunny_rain: Weather Data Analysis')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>',unsafe_allow_html=True)



# Options to upload a file
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "xlsx", "xls"])
#partitioning the page
col1, col2 = st.columns((2))
with col1:
    if fl is not None:
        # Check file type and read accordingly
        if fl.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Excel file
            df = pd.read_excel(fl)
            st.write("Data from uploaded Excel file:")
            st.write(df)
        else:
            # Assume other formats as CSV
            df = pd.read_csv(fl, encoding="ISO-8859-1")
            st.write("Data from uploaded CSV file:")
            st.write(df)
    else:
        # Read data from the default Excel file
        #os.chdir(r"default directory containing the file")
        df = pd.read_excel("temp_humid_data_one.xlsx")
        st.write("Data from default Excel file:")
        st.write(df)

df["time"]=pd.to_datetime(df["time"])
# Create a new column "month" with the month information
df["month"] = df["time"].dt.strftime('%B')
# Create a new column "year" with the year information
df["year"] = df["time"].dt.year
# Create a new column "day" with the year information
df["day"] = df["time"].dt.day

st.sidebar.header("Choose your filter: ")
# Create for Month
month = st.sidebar.multiselect("Pick the Months", df["month"].unique())
if not month:
    df2 = df.copy()
else:
    df2 = df[df["month"].isin(month)].copy()

# Create for Year
year = st.sidebar.multiselect("Pick the Years", df["year"].unique())
if not year:
    df3 = df.copy()
else:
    df3 = df[df["year"].isin(year)].copy()

# Filter the data based on Region, State, and City
if not month and not year:
    filtered_df = df.copy()
elif not year:
    filtered_df = df3
elif not month:
    filtered_df = df2
else:
    filtered_df = df[df2["month"].isin(month) & df3["year"].isin(year)].copy()

# Reset index after filtering
filtered_df.reset_index(drop=True, inplace=True)

with col2:
    st.write("Data After Filtering:")
    st.write(filtered_df)

filter_dataset=filtered_df.copy()

# Heatmap for Temperature
heatmap_temp = filter_dataset.pivot_table(values='temperature_mean', index='month', columns='day', aggfunc='mean')
fig_heatmap_temp = px.imshow(heatmap_temp, x=heatmap_temp.columns, y=heatmap_temp.index, color_continuous_scale='Viridis', 
                             labels={'x': 'Day', 'y': 'Month', 'color': 'Temperature'},
                             title='Heatmap of Temperature')

# Heatmap for Relative Humidity
heatmap_rh = filter_dataset.pivot_table(values='relativehumidity_mean', index='month', columns='day', aggfunc='mean')
fig_heatmap_rh = px.imshow(heatmap_rh, x=heatmap_rh.columns, y=heatmap_rh.index, color_continuous_scale='Viridis', 
                           labels={'x': 'Day', 'y': 'Month', 'color': 'Relative Humidity'},
                           title='Heatmap of Relative Humidity')

# Show the plots
st.plotly_chart(fig_heatmap_temp, use_container_width=True)
st.plotly_chart(fig_heatmap_rh, use_container_width=True)

# Overview of the dataset - Temperature
with col1:
    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    filter_dataset['month'] = pd.Categorical(filter_dataset['month'], categories=month_order, ordered=True)

    if month:
        selected_months_df = filter_dataset[filter_dataset['month'].isin(month)]
        grouped_df = selected_months_df.groupby('month')['temperature_mean'].agg(['mean', 'median', 'max']).reset_index()
        title = 'Mean, Median, and Maximum Temperature of Selected Months'
    else:
        grouped_df = filter_dataset.groupby('month')['temperature_mean'].agg(['mean', 'median', 'max']).reset_index()
        title = 'Mean, Median, and Maximum Temperature of All Months'

    fig = px.line(grouped_df, x='month', y=['mean', 'median', 'max'],
                  labels={'value': 'Temperature'},
                  title=title)

    fig.update_layout(xaxis=dict(title='Month', categoryorder='array', categoryarray=month_order), yaxis=dict(title='Temperature'))

    st.plotly_chart(fig, use_container_width=True)

# Overview of the dataset - Relative Humidity
with col2:
    if month:
        selected_months_df = filter_dataset[filter_dataset['month'].isin(month)]
        grouped_df = selected_months_df.groupby('month')['relativehumidity_mean'].agg(['mean', 'median', 'max']).reset_index()
        title = 'Mean, Median, and Maximum Relative Humidity of Selected Months'
    else:
        grouped_df = filter_dataset.groupby('month')['relativehumidity_mean'].agg(['mean', 'median', 'max']).reset_index()
        title = 'Mean, Median, and Maximum Relative Humidity of all Months'

    fig = px.line(grouped_df, x='month', y=['mean', 'median', 'max'],
                  labels={'value': 'Relative Humidity'},
                  title=title)

    fig.update_layout(xaxis=dict(title='Month', categoryorder='array', categoryarray=month_order), yaxis=dict(title='Relative Humidity'))

    st.plotly_chart(fig, use_container_width=True)

#Scatter Plot with Trend Line: Temperature vs Relative Humidity
with col1:
    #Scatter plot 
    from scipy.stats import linregress
    import numpy as np
    # Create a scatter plot using Plotly Express
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_df['temperature_mean'], y=filtered_df['relativehumidity_mean'],
                            mode='markers', name='Temperature vs Relative Humidity',
                            marker=dict(color='light blue'),showlegend=False))

    # Calculate linear regression
    slope, intercept, _, _, _ = linregress(filtered_df['temperature_mean'], filtered_df['relativehumidity_mean'])
    x_line = np.linspace(filtered_df['temperature_mean'].min(), filtered_df['temperature_mean'].max(), 100)
    y_line = slope * x_line + intercept

    # Add trend line
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name='Trend Line', line=dict(color='red', dash='dash'),showlegend=False))

    # Update layout
    fig.update_layout(title='Scatter Plot with Trend Line: Temperature vs Relative Humidity',
                    xaxis=dict(title='Temperature'),
                    yaxis=dict(title='Relative Humidity', showgrid=False))

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
#Interactive Correlation Plot: Temperature vs Relative Humidit
with col2:
    # Create a DataFrame with temperature and relative humidity
    correlation_df = filtered_df[['temperature_mean', 'relativehumidity_mean']]

    # Calculate the correlation matrix
    correlation_matrix = correlation_df.corr()

    # Create an interactive heatmap using plotly express
    fig = px.imshow(correlation_matrix, labels=dict(color="Correlation"), color_continuous_scale='Blues')

    # Update layout
    fig.update_layout(title='Interactive Correlation Plot: Temperature vs Relative Humidity',
                    xaxis=dict(title='Variable'),
                    yaxis=dict(title='Variable'))

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)


#for day Comparison between Temperature and Humidity over Days
# Group by day and calculate the average for each day
grouped_df = filtered_df.groupby('day').agg({'temperature_mean': 'mean', 'relativehumidity_mean': 'mean'}).reset_index()

# Melt the DataFrame to have a single column for values and another for the variable
melted_df = pd.melt(grouped_df, id_vars=['day'], value_vars=['temperature_mean', 'relativehumidity_mean'],
                    var_name='Variable', value_name='Value')


import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Assuming you have grouped and melted the DataFrame as in the provided code

# Create a dual-axis bar chart using Plotly Express and make_subplots
fig = make_subplots(specs=[[{'secondary_y': True}]])

# Bar colors for temperature and relative humidity with transparency
temperature_color = 'blue'  # Light Blue with alpha = 0.7
humidity_color = 'rgba(150, 170, 230,0.5)'  # Orange with alpha = 0.7


# Add bar traces with specified colors and transparency
fig.add_trace(go.Bar(x=melted_df['day'], y=melted_df['Value'][melted_df['Variable']=='temperature_mean'],
                     name='Temperature', marker=dict(color=temperature_color)))
fig.add_trace(go.Bar(x=melted_df['day'], y=melted_df['Value'][melted_df['Variable']=='relativehumidity_mean'],
                     name='Relative Humidity', marker=dict(color=humidity_color)),
              secondary_y=True)

# Update layout
fig.update_layout(title='Comparison between Temperature and Humidity over Days',
                  xaxis=dict(title='Day'),
                  yaxis=dict(title='Temperature', showgrid=False),
                  yaxis2=dict(title='Relative Humidity', showgrid=False, overlaying='y', side='right'))

# Show the plot
st.plotly_chart(fig, use_container_width=True, height=400)





#For Months Comparison between Temperature and Humidity over Months
# Group by month and calculate the average for each month
grouped_df = filtered_df.groupby('month').agg({'temperature_mean': 'mean', 'relativehumidity_mean': 'mean'}).reset_index()

# Melt the DataFrame to have a single column for values and another for the variable
melted_df = pd.melt(grouped_df, id_vars=['month'], value_vars=['temperature_mean', 'relativehumidity_mean'],
                    var_name='Variable', value_name='Value')

# Create a plot using Plotly Express and make_subplots
fig = make_subplots(specs=[[{'secondary_y': True}]])

# Check if only one month is selected
if len(filtered_df['month'].unique()) == 1:
    # Show a bar plot for mean values when a single month is selected
    fig.add_trace(go.Bar(x=['Temperature', 'Relative Humidity'], y=[grouped_df['temperature_mean'].iloc[0], grouped_df['relativehumidity_mean'].iloc[0]],
                         marker=dict(color=['blue', 'orange'])))
else:
    # Show a line plot when multiple months are selected
    fig.add_trace(go.Scatter(x=melted_df['month'], y=melted_df['Value'][melted_df['Variable']=='temperature_mean'],
                             mode='lines', name='Temperature Mean', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=melted_df['month'], y=melted_df['Value'][melted_df['Variable']=='relativehumidity_mean'],
                             mode='lines', name='Relative Humidity Mean', line=dict(color='orange')),
                  secondary_y=True)

# Round the 'Value' column to 2 decimal places
melted_df['Value'] = melted_df['Value'].round(2)

# Create a plot using Plotly Express
fig = px.bar(melted_df, x='month', y='Value', color='Variable',
             labels={'Value': 'Value', 'Variable': 'Metric'},
             text='Value', # Add this line to display exact values on each bar
             barmode='group')

# Update layout
fig.update_layout(title='Comparison between Temperature and Humidity over Months',
                  xaxis=dict(title='Months'),
                  yaxis=dict(title='Value'),
                  legend=dict(title='Metric'))

# Rotate the text on the bars to be horizontal
fig.update_traces(textposition='outside', textangle=0)

# Show the plot
st.plotly_chart(fig, use_container_width=True)






import streamlit as st
import pandas as pd

# Data
data = {
    'Model Name': ['ARIMA', 'SARIMA', 'SARIMAX', 'VAR', 'LSTM', 'GRU'],
    'MSE Value': [63.59, 39.02, 398.78, 74.62, 723.02, 594.92],
    'RMSE Value': [7.97, 6.25, 19.97, 8.64, 26.89, 24.39]
}

# Create DataFrame
df = pd.DataFrame(data)

# Highlight row where Model is 'SARIMA' with light blue background and black font color
highlighted_df = df.style.apply(lambda x: ['background: lightblue; color: black' if x['Model Name'] == 'SARIMA' else '' for i in x], axis=1)

# Streamlit App
st.markdown("<h1 style='font-size:24px;'>Model Evaluation Metrics</h1>", unsafe_allow_html=True)

# Display highlighted table with 2 decimal places
st.table(highlighted_df.format({
    'MSE Value': '{:.2f}',
    'RMSE Value': '{:.2f}'
}))







# SARIMA Forecasting
st.header("SARIMA Forecasting")

# Sidebar for SARIMA order and seasonal_order parameters
p_order_sarima = st.sidebar.slider("Select p (non-seasonal AR order)", min_value=0, max_value=5, value=1)
d_order_sarima = st.sidebar.slider("Select d (non-seasonal differencing order)", min_value=0, max_value=2, value=0)
q_order_sarima = st.sidebar.slider("Select q (non-seasonal MA order)", min_value=0, max_value=5, value=1)

P_order_sarima = st.sidebar.slider("Select P (seasonal AR order)", min_value=0, max_value=5, value=1)
D_order_sarima = st.sidebar.slider("Select D (seasonal differencing order)", min_value=0, max_value=2, value=0)
Q_order_sarima = st.sidebar.slider("Select Q (seasonal MA order)", min_value=0, max_value=5, value=1)
seasonal_period_sarima = st.sidebar.slider("Select seasonal period", min_value=1, max_value=24, value=12)

# Fit SARIMA model
order_sarima = (p_order_sarima, d_order_sarima, q_order_sarima)
seasonal_order_sarima = (P_order_sarima, D_order_sarima, Q_order_sarima, seasonal_period_sarima)
model_sarima = SARIMAX(filtered_df['temperature_mean'], order=order_sarima, seasonal_order=seasonal_order_sarima)
fitted_sarima = model_sarima.fit()

# Forecast
forecast_steps_sarima = 12  # Adjust the forecast horizon as needed
forecast_sarima = fitted_sarima.get_forecast(steps=forecast_steps_sarima, alpha=0.05)
forecast_index_sarima = pd.date_range(filtered_df['time'].max(), periods=forecast_steps_sarima + 1, freq='M')[1:]
forecast_values_sarima = forecast_sarima.predicted_mean

# Plot SARIMA Forecast
fig_sarima = go.Figure()
fig_sarima.add_trace(go.Scatter(x=filtered_df['time'], y=filtered_df['temperature_mean'], mode='lines', name='Actual Temperature'))
fig_sarima.add_trace(go.Scatter(x=forecast_index_sarima, y=forecast_values_sarima, mode='lines', name='SARIMA Forecast'))

# Update layout
fig_sarima.update_layout(title='SARIMA Forecast vs Actual Temperature',
                         xaxis=dict(title='Time'),
                         yaxis=dict(title='Temperature'))

# Show the plot
st.plotly_chart(fig_sarima, use_container_width=True)






