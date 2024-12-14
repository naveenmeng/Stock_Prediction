import requests
import streamlit as st

class StockMarketChatBot:
    def __init__(self):
        self.api_key = 'c5cca008138e523b624789405406946f232352c74e551780d0458a41833d5969'  # Replace with your SerpAPI key
        self.messages = []

    # Method to send message to the bot
    def send_message(self, user_message):
        if not user_message.strip():
            return

        self.messages.append({'role': 'user', 'message': user_message})

        # Fetch stock market data
        self.fetch_stock_market_data(user_message)

    # Method to fetch stock market data from SerpAPI
    def fetch_stock_market_data(self, query):
        api_url = 'https://serpapi.com/search'

        query_parameters = {
            'q': f'stock market {query}',  # Prefix the query with 'stock market' to filter results
            'hl': 'en',  # Language (English)
            'gl': 'us',  # Country (United States)
            'api_key': self.api_key,
            'engine': 'google',  # Search engine
            'start': '0',  # Pagination (offset)
        }

        response = requests.get(api_url, params=query_parameters)

        if response.status_code == 200:
            response_body = response.json()
            search_results = response_body.get('organic_results', [])

            if search_results:
                combined_response = self.generate_bot_response(search_results)
                self.messages.append({'role': 'bot', 'message': combined_response})
            else:
                self.show_error("Sorry, I couldn't find relevant information about the stock market.")
        else:
            self.show_error(f"Error: {response.status_code} - {response.text}")

    # Generate a response from the search results in point-wise format
    def generate_bot_response(self, search_results):
        snippets = []
        for result in search_results:
            snippet = result.get('snippet', '')
            if snippet:
                snippets.append(snippet)

        if not snippets:
            return "Sorry, I couldn't find relevant information. Please try asking in a different way."

        # Format the response as bullet points
        response = "Here's what I found about the stock market:\n\n"
        for idx, snippet in enumerate(snippets, start=1):
            response += f"{idx}. {snippet}\n"  # Bullet points with numbering
        return response

    # Show error message if something goes wrong
    def show_error(self, message):
        self.messages.append({'role': 'bot', 'message': message})

    def display_chat(self):
        # Display the conversation
        for message in self.messages:
            role = message['role']
            msg = message['message']
            if role == 'user':
                st.write(f'**User**: {msg}')
            elif role == 'bot':
                st.write(f'**Bot**: {msg}')

def show():
    chatbot = StockMarketChatBot()

    st.title("Stock Market Chatbot")

    # Display chat history
    for message in chatbot.messages:
        role = message['role']
        msg = message['message']
        if role == 'user':
            st.write(f'**User**: {msg}')
        elif role == 'bot':
            st.write(f'**Bot**: {msg}')

    # Input field for user to send a message
    user_input = st.text_input("Enter your stock market query:", "")

    if user_input:
        chatbot.send_message(user_input)
        chatbot.display_chat()

    # Add an option to clear chat history
    if st.button("Clear Chat"):
        chatbot.messages = []

# Run the Streamlit app
if __name__ == "__main__":
    show()
