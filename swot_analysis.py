import requests
import streamlit as st

# Replace with your SerpAPI key
API_KEY = 'c5cca008138e523b624789405406946f232352c74e551780d0458a41833d5969'


# Function to get stock data from SerpAPI
def get_stock_data_from_serpapi(ticker):
    api_url = 'https://serpapi.com/search'

    query_parameters = {
        'q': f'stock market {ticker}',  # Prefix the query with 'stock market' to filter results
        'hl': 'en',  # Language (English)
        'gl': 'us',  # Country (United States)
        'api_key': API_KEY,
        'engine': 'google',  # Search engine
        'start': '0',  # Pagination (offset)
    }

    response = requests.get(api_url, params=query_parameters)

    if response.status_code == 200:
        response_body = response.json()
        return response_body
    else:
        st.error(f"Error fetching stock data: {response.status_code}")
        return None


# Function to generate SWOT analysis
def generate_swot_analysis(ticker):
    stock_data = get_stock_data_from_serpapi(ticker)

    if stock_data:
        # Extracting search results from SerpAPI
        search_results = stock_data.get('organic_results', [])

        if search_results:
            # Analyze the search results and generate SWOT analysis
            strengths = []
            weaknesses = []
            opportunities = []
            threats = []

            # Expanded logic for matching SWOT categories
            for result in search_results:
                snippet = result.get('snippet', '').lower()

                # Strengths: Look for positive phrases like "growth", "strong", "increase"
                if any(keyword in snippet for keyword in
                       ['growth', 'strong', 'increase', 'positive', 'improved', 'expand']):
                    strengths.append(result.get('title', 'No title') + ": " + snippet)

                # Weaknesses: Look for negative phrases like "decline", "drop", "weak", "loss"
                elif any(keyword in snippet for keyword in ['decline', 'drop', 'weak', 'loss', 'decrease', 'fall']):
                    weaknesses.append(result.get('title', 'No title') + ": " + snippet)

                # Opportunities: Look for phrases like "opportunity", "expansion", "new markets"
                elif any(keyword in snippet for keyword in
                         ['opportunity', 'expansion', 'new markets', 'growth potential']):
                    opportunities.append(result.get('title', 'No title') + ": " + snippet)

                # Threats: Look for phrases like "competition", "risk", "threat", "uncertainty"
                elif any(keyword in snippet for keyword in
                         ['competition', 'risk', 'threat', 'uncertainty', 'challenges']):
                    threats.append(result.get('title', 'No title') + ": " + snippet)

            # If no insights are found, add default messages
            if not strengths:
                strengths.append(
                    "No specific strengths found. Try rephrasing your query or look for broader market trends.")
            if not weaknesses:
                weaknesses.append("No specific weaknesses found. The market seems stable for this stock.")
            if not opportunities:
                opportunities.append(
                    "No specific opportunities found. Consider looking into market expansion or new ventures.")
            if not threats:
                threats.append("No specific threats found. However, keep an eye on potential market risks.")

            return strengths, weaknesses, opportunities, threats
        else:
            return [], [], [], []

    else:
        return [], [], [], []


# Function to directly show SWOT analysis based on ticker
# Function to directly show SWOT analysis based on ticker
def show_swot_analysis(ticker):
    st.title(f"SWOT Analysis for {ticker}")

    strengths, weaknesses, opportunities, threats = generate_swot_analysis(ticker)

    # Display strengths
    if strengths:
        st.subheader("Strengths")
        for point in strengths:
            st.write(f"- {point}")

    # Display weaknesses
    if weaknesses:
        st.subheader("Weaknesses")
        for point in weaknesses:
            st.write(f"- {point}")

    # Display opportunities
    if opportunities:
        st.subheader("Opportunities")
        for point in opportunities:
            st.write(f"- {point}")

    # Display threats
    if threats:
        st.subheader("Threats")
        for point in threats:
            st.write(f"- {point}")

    # Provide a recommendation based on the analysis
    st.subheader("Recommendation for Long Term Investors")
    if len(strengths) + len(opportunities) > len(weaknesses) + len(threats):
        st.markdown(
            "<div style='color:green; font-size:18px; font-weight:bold;'>"
            "Based on the analysis, this stock shows more positive factors. It is recommended to <u>BUY</u>."
            "</div>",
            unsafe_allow_html=True,
        )
    elif len(weaknesses) + len(threats) > len(strengths) + len(opportunities):
        st.markdown(
            "<div style='color:red; font-size:18px; font-weight:bold;'>"
            "Based on the analysis, this stock shows more negative factors. It is recommended to <u>SELL</u>."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='color:orange; font-size:18px; font-weight:bold;'>"
            "The analysis shows a balance of positive and negative factors. It is recommended to <u>HOLD</u> or investigate further."
            "</div>",
            unsafe_allow_html=True,
        )
