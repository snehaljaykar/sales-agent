import os
import requests
import json

SERVER_URL = os.getenv("SERVER_URL", "http://flask_server:5000")

def query_user(user_id):
    """Original function for processing user files"""
    url = f"{SERVER_URL}/users/{user_id}"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def send_chat_query(user_id, query_text):
    """Send a chat query to the server"""
    url = f"{SERVER_URL}/chat"
    payload = {
        "user_id": user_id,
        "query": query_text
    }
    try:
        r = requests.post(url, json=payload, timeout=300)
        return r.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def print_welcome():
    print("=" * 60)
    print("Sales Agent Chatbot CLI")
    print("=" * 60)
    print("Welcome to the Sales Agent AI assistant!")
    print("Powered by local LLM for intelligent query processing")
    print("\nSample commands you can try:")
    print("- list my call ids")
    print("- summarise the last call") 
    print("- give me all negative comments when pricing was mentioned")
    print("\nType 'help' for more information")
    print("Type 'exit' or 'quit' to leave")
    print("=" * 60)

def print_help():
    print("\nAvailable Commands:")
    print("- Ask any question about your call transcripts")
    print("- Request summaries, analysis, or specific information")
    print("- Use natural language - the AI will understand!")
    print("\nExamples:")
    print("- 'What were the main objections in my calls?'")
    print("- 'Show me calls where customers mentioned budget concerns'") 
    print("- 'Summarize the pricing discussions from last week'")

def main():
    print_welcome()
    
    # Get user ID
    while True:
        user_input = input("\nEnter your user ID: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            return
        if user_input.isdigit():
            user_id = user_input
            print(f"Welcome User {user_id}! You can now start chatting.")
            response = query_user(user_input)
            print("Response:", response)
            break
        else:
            print("Please enter a numeric user ID")
    
    # Start chat loop
    print(f"\nChat started for User {user_id}")
    print("Ask me anything about your call transcripts...")
    
    while True:
        try:
            query = input(f"\n[User {user_id}] >>> ").strip()
            
            if not query:
                continue
                
            if query.lower() in ("exit", "quit", "bye"):
                print("Thanks for using Sales Agent Chatbot! Goodbye!")
                break
                
            if query.lower() == "help":
                print_help()
                continue
            
            print("Processing your query...")

            response = send_chat_query(user_id, query)
            
            # Display response
            if "error" in response:
                print(f"Error: {response['error']}")
            elif "answer" in response:
                print(f"\nAssistant: {response['answer']}")
                if "sources" in response and response["sources"]:
                    print(f"\nSources: {response['sources']} relevant documents found")
            else:
                print(f"Response: {json.dumps(response, indent=2)}")
                
        except KeyboardInterrupt:
            print("\n\nReceived Ctrl+C. Goodbye!")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
