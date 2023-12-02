# app.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import feedparser
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load the data
@st.cache_data
def load_data():
    return pd.read_csv("Crypto_simi.csv")

def main():

    st.title("SOLIGENCE CRYPTO CURRENCY PREDICTION SYSTEM")

    menu = ["Home","EDA", "News", "Buy and Sell Signal", "Trends", "Detection", "Up and Down Prediction", "Prediction"]
    choice = st.sidebar.selectbox("Main Menu", menu)

    if choice == "Home":
        st.image("btc.jpg")
        st.subheader("Welcome to our Exploratory Data Analysis and Prediction System")


    elif choice == "EDA":

        st.subheader("Exploratory Data Analysis")

        # Load the data

        data = load_data()

        # Get the list of unique cryptocurrencies

        crypto_list = data['coin_name'].unique()

        # Sidebar options

        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", crypto_list)

        st.write(f"Showing data for {selected_crypto}")

        # Filter data for selected crypto

        selected_data = data[data['coin_name'] == selected_crypto]

        # Date range slider

        min_date = pd.to_datetime(selected_data['Date']).min().date()

        max_date = pd.to_datetime(selected_data['Date']).max().date()

        date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])

        if date_range:
            start_date, end_date = date_range

            mask = (pd.to_datetime(selected_data['Date']).dt.date >= start_date) & (
                        pd.to_datetime(selected_data['Date']).dt.date <= end_date)

            selected_data = selected_data[mask]

        # Plotting the price

        st.line_chart(selected_data[['Date', 'Close']].set_index('Date'))




    elif choice == "News":

        st.subheader("Latest Crypto News")

        # Define a list of RSS feeds from popular crypto news websites

        RSS_FEEDS = {

            'CoinTelegraph': 'https://cointelegraph.com/rss',

            # Add more RSS feeds here

        }

        # Sidebar option to select a cryptocurrency

        crypto_list = ["Bitcoin", "Ethereum", "Ripple", "Litecoin", "Dodgecoin", "Stelar"]  # Add more crypto names or fetch dynamically

        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency for News", crypto_list)


        # Fetch and display news
        for publisher, url in RSS_FEEDS.items():

            feed = feedparser.parse(url)

            st.write(f"Top stories from {publisher}:")

            count = 0  # To limit the number of news articles displayed

            for entry in feed.entries:
                if selected_crypto in entry.title or selected_crypto == "All":
                    st.write(f"**{entry.title}**")

                    st.write(f"[Read more]({entry.link})")
                    st.write("---")
                    count += 1
                if count >= 5:  # Limit to top 5 stories per source

                    break

    elif choice == ("Buy and Sell Signal"):
        st.subheader("Best Time to Buy and Sell")

        # Load the data
        data = load_data()

        # Get the list of unique cryptocurrencies
        crypto_list = data['coin_name'].unique()

        # Sidebar option to select a cryptocurrency
        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency for Analysis", crypto_list)

        # Filter data for the selected cryptocurrency
        selected_data = data[data['coin_name'] == selected_crypto]

        # Calculate short-term and long-term moving averages
        short_window = 50
        long_window = 200

        selected_data['Short_MA'] = selected_data['Close'].rolling(window=short_window).mean()
        selected_data['Long_MA'] = selected_data['Close'].rolling(window=long_window).mean()

        # Drop NaN values after calculating the moving averages
        selected_data.dropna(subset=['Short_MA', 'Long_MA'], inplace=True)


        # Identifying the golden cross and death cross
        golden_cross = (selected_data['Short_MA'] > selected_data['Long_MA']).astype(int)
        death_cross = (selected_data['Short_MA'] <= selected_data['Long_MA']).astype(int)

        # Filter out where crosses occur
        buy_signals = selected_data[(golden_cross == 1) & (golden_cross.shift(1) == 0)]
        sell_signals = selected_data[(death_cross == 1) & (death_cross.shift(1) == 0)]

        st.write("Golden Cross (Buy Signals):")
        st.write(buy_signals[['Date', 'Close', 'Short_MA', 'Long_MA']])

        st.write("Death Cross (Sell Signals):")
        st.write(sell_signals[['Date', 'Close', 'Short_MA', 'Long_MA']])

        # Calculating anticipated profit/loss
        if not buy_signals.empty and not sell_signals.empty:
            # Ensure we buy before we sell
            if buy_signals.iloc[0]['Date'] < sell_signals.iloc[0]['Date']:
                potential_profit = sell_signals.iloc[0]['Close'] - buy_signals.iloc[0]['Close']
            else:
                potential_profit = 0  # We only sell if we have bought before

            st.write(f"Anticipated Profit from first buy to first sell: ${potential_profit:.2f}")
        else:
            st.write("No buy or sell signals detected in the selected data range.")

    elif choice == "Trends":
        st.subheader("Crypto Trends")
        # Load the data
        data = load_data()

        # Get the list of unique cryptocurrencies
        crypto_list = data['coin_name'].unique()

        # Sidebar options
        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency for Correlation Analysis", crypto_list)
        st.write(f"Analyzing correlations for {selected_crypto}")

        # Pivot the data to have coins as columns, dates as index and Close prices as values
        pivot_data = data.pivot(index='Date', columns='coin_name', values='Close')

        # Compute the correlation matrix
        correlation_matrix = pivot_data.corr()

        # Get correlations for the selected cryptocurrency
        correlations = correlation_matrix[selected_crypto].sort_values(ascending=False)

        # Display the top positive correlations (excluding the selected cryptocurrency itself)
        image_path = "plot.png"
        st.image(image_path, caption="Correlation Plot", use_column_width=True)
        st.write("Top 10 Positive Correlations:")
        st.write(correlations[1:11])

        # Display the top negative correlations
        st.write("Top 10 Negative Correlations:")
        st.write(correlations[-10:])

        # Integrate your trends function here

    elif choice == "Detection":
        st.subheader("Crypto Detection")
        data = load_data()

        # Get the list of unique cryptocurrencies
        crypto_list = data['coin_name'].unique()

        # Sidebar options for selecting cryptocurrency and moving average window
        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency for Moving Average Analysis", crypto_list)
        ma_window = st.sidebar.slider("Select Moving Average Window", 1, 60, 7)  # Default is 7 days

        # Filter data for the selected cryptocurrency
        selected_data = data[data['coin_name'] == selected_crypto]

        # Compute the moving average
        selected_data['Moving Average'] = selected_data['Close'].rolling(window=ma_window).mean()

        # Plotting the close price and moving average
        st.line_chart(selected_data[['Date', 'Close', 'Moving Average']].set_index('Date'))

        # Integrate your detection function here

    elif choice == "Up and Down Prediction":
        st. subheader("Prediction for Coins Going UP and DOWN")
        data = load_data()

        # Get the list of unique cryptocurrencies
        crypto_list = data['coin_name'].unique()

        # Sidebar option to select multiple cryptocurrencies
        selected_cryptos = st.sidebar.multiselect("Select Cryptocurrencies for Market State Prediction", crypto_list)

        if not selected_cryptos:
            st.write("Please select at least one cryptocurrency.")
            return

        # Filter data for the selected cryptocurrencies
        filtered_data = data[data['coin_name'].isin(selected_cryptos)].copy()

        # Calculate daily returns
        filtered_data['Return'] = filtered_data.groupby('coin_name')['Close'].pct_change()

        # Create binary target: 1 if return is positive (market went up), 0 if return is negative (market went down)
        filtered_data['Target'] = (filtered_data['Return'] > 0).astype(int)

        # Shift target one day forward to predict next day's movement
        filtered_data['Target'] = filtered_data.groupby('coin_name')['Target'].shift(-1)



        # Drop rows with NaN values
        filtered_data.dropna(inplace=True)

        # Split data into train and test sets
        X = filtered_data[['Return']]
        y = filtered_data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a logistic regression model
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        st.write(f"Model Accuracy on Test Data: {accuracy:.2f}")

        # Predict next day's movement
        last_return = X.iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_return)[0]
        st.write("Predicted Market State for Next Day:")
        st.write("Up" if prediction == 1 else "Down")



    elif choice == "Prediction":

        st.subheader("Crypto Prediction")

        # Load the data
        data = load_data()

        # Get the list of unique cryptocurrencies

        crypto_list = data['coin_name'].unique()

        # Sidebar option to select a cryptocurrency

        selected_crypto = st.sidebar.selectbox("Select Cryptocurrency for Prediction", crypto_list)

        # Filter data for the selected cryptocurrency

        selected_data = data[data['coin_name'] == selected_crypto]

        # Prepare data for modeling

        selected_data['Days'] = (
                    pd.to_datetime(selected_data['Date']) - pd.to_datetime(selected_data['Date'].min())).dt.days

        X = selected_data['Days'].values.reshape(-1, 1)

        y_high = selected_data['High'].values

        y_low = selected_data['Low'].values

        # Splitting the data
        X_train, X_test, y_high_train, y_high_test = train_test_split(X, y_high, test_size=0.2, random_state=42)
        X_train, X_test, y_low_train, y_low_test = train_test_split(X, y_low, test_size=0.2, random_state=42)

        # Train models for 'High' prices
        model_high = LinearRegression().fit(X_train, y_high_train)
        y_high_pred = model_high.predict(X_test)
        mae_high = mean_absolute_error(y_high_test, y_high_pred)
        mse_high = mean_squared_error(y_high_test, y_high_pred)
        rmse_high = np.sqrt(mse_high)
        r2_high = r2_score(y_high_test, y_high_pred)

        # Train models for 'Low' prices
        model_low = LinearRegression().fit(X_train, y_low_train)
        y_low_pred = model_low.predict(X_test)
        mae_low = mean_absolute_error(y_low_test, y_low_pred)
        mse_low = mean_squared_error(y_low_test, y_low_pred)
        rmse_low = np.sqrt(mse_low)
        r2_low = r2_score(y_low_test, y_low_pred)

        # Display the metrics in Streamlit
        st.write("Evaluation Metrics for 'High' Price Predictions:")
        st.write(f"MAE: {mae_high:.2f}, MSE: {mse_high:.2f}, RMSE: {rmse_high:.2f}, R2: {r2_high:.2f}")

        st.write("Evaluation Metrics for 'Low' Price Predictions:")
        st.write(f"MAE: {mae_low:.2f}, MSE: {mse_low:.2f}, RMSE: {rmse_low:.2f}, R2: {r2_low:.2f}")

        # Predicting the next day
        next_day = np.array([selected_data['Days'].max() + 1]).reshape(-1, 1)
        predicted_high = model_high.predict(next_day)[0]
        predicted_low = model_low.predict(next_day)[0]

        st.write(f"Predicted High for {selected_crypto} on the next day: ${predicted_high:.2f}")
        st.write(f"Predicted Low for {selected_crypto} on the next day: ${predicted_low:.2f}")

        # # Train models
        #
        # model_high = LinearRegression().fit(X_train, y_high_train)
        #
        # model_low = LinearRegression().fit(X_train, y_low_train)
        #
        # # Predicting the next day
        #
        # next_day = np.array([selected_data['Days'].max() + 1]).reshape(-1, 1)
        #
        # predicted_high = model_high.predict(next_day)[0]
        #
        # predicted_low = model_low.predict(next_day)[0]
        #
        # st.write(f"Predicted High for {selected_crypto} on the next day: ${predicted_high:.2f}")
        #
        # st.write(f"Predicted Low for {selected_crypto} on the next day: ${predicted_low:.2f}")
        #
        # # Integrate your prediction function here


if __name__ == "__main__":
    main()
