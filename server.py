import os
import logging
import openai

from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
from bs4 import BeautifulSoup

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
)
logger = logging.getLogger()

openai.organization = os.environ["ORGANIZATION"]
openai.api_key = os.environ["OPENAI_API_KEY"]

@app.route("/process_page", methods=["POST"])
def process_page():
  body = request.json["body"]
  soup = BeautifulSoup(body)
  
  print(soup.prettify())
  

@app.route("/complete_chat", methods=["POST"])
def complete_chat():
    user_input = request.json["user_input"]    

    response = openai.Moderation.create(
      input=user_input
    )
    
    if response.results[0]['flagged']:
      return {
        "message": "Your message contains inappropiate content."
      }

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": f"{user_input}"}]
    )

    message = response["choices"][0]["message"]["content"]

    logger.info(message)
    return {"message": message}


if __name__ == "__main__":
    app.run()
