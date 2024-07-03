import os
import requests
import json
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt import App
from flask import Flask, request, jsonify
from flask_cors import CORS
import tiktoken
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if not in production
if os.environ.get('FLASK_ENV') != 'production':
    load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_SIGNING_SECRET = os.environ["SLACK_SIGNING_SECRET"]
SLACK_BOT_USER_ID = os.environ["SLACK_BOT_USER_ID"]
CANOPY_API_URL = os.environ["CANOPY_API_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = os.environ["INDEX_NAME"]

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

# Initialize the Flask app
flask_app = Flask(__name__)
CORS(flask_app)
handler = SlackRequestHandler(app)

def count_tokens(text, model="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def trim_conversation_context(conversation_history, max_tokens=4000, model="gpt-3.5-turbo"):
    total_tokens = sum(count_tokens(msg["content"], model) for msg in conversation_history)
    
    while total_tokens > max_tokens and len(conversation_history) > 1:
        removed_msg = conversation_history.pop(0)
        total_tokens -= count_tokens(removed_msg["content"], model)
    
    return conversation_history

def send_to_canopy(query):
    custom_prompt = """You are a helpful assistant for forageSF. Your job is to help anyone who has a question
        about our data. Always respond in an upbeat and friendly way, and ensure your answers are 
        informative and supportive. When you provide an answer, also include a quote of the specific 
        content you used to arrive at your answer. """
    modified_query = f"{custom_prompt} {query}"
    
    payload = {
        "messages": [{"role": "user", "content": modified_query}],
        "stream": False,
        "model": "GPT-4",
        "temperature": 0
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    try:
        response = requests.post(CANOPY_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            message_content = response_data["choices"][0]["message"]["content"]
            return message_content
        else:
            return "Error: Unexpected response format from Canopy server"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Canopy server: {e}")
        return "Error: Could not get a response from the Canopy server"

@app.event("app_mention")
def handle_mentions(body, say):
    text = body["event"]["text"]
    mention = f"<@{SLACK_BOT_USER_ID}>"
    text = text.replace(mention, "").strip()

    response = send_to_canopy(text)
    say(response)

@app.event("message")
def handle_message_events(body, logger):
    logger.info(body)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    if request.json.get("type") == "url_verification":
        # Respond with the challenge if it's a URL verification request
        return request.json["challenge"], 200, {"Content-Type": "text/plain"}
    else:
        # Handle other events properly
        return handler.handle(request)

@flask_app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    flask_app.run(host='0.0.0.0', port=port)