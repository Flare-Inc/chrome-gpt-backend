import os
import logging
import openai
import json

from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from scipy import spatial

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
  soup = BeautifulSoup(body, 'html.parser')
  
  embeddings = []
  for p in soup.find_all("p"):
    text = p.get_text(separator='\n')
    
    if not text.isspace():
      response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text,
      )
      
      embeddings.append({
        "text": text,
        "embedding": response.data[0]['embedding'],
        })
  
  return {"embeddings": embeddings}
  

@app.route("/complete_chat", methods=["POST"])
def complete_chat():
    user_input = request.json["user_input"]
    embeddings = json.loads(request.json["embeddings"])['embeddings']

    response = openai.Moderation.create(
      input=user_input
    )
    
    if response.results[0]['flagged']:
      return {
        "message": "Your message contains inappropiate content."
      }
      
    response = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=user_input,
    )
    input_embedding = response.data[0]['embedding']
    
    opt, opt_injection = 0, ""
    for embd in embeddings:
      similarity = 1 - spatial.distance.cosine(input_embedding, embd['embedding'])
      
      if similarity > opt:
        opt = similarity
        opt_injection = embd['text']
      
    prompt = f"""Given the following context from an article, please answer the user's question as well as possible.
    
    Article Context:
    {opt_injection}
    
    User Question:
    {user_input}"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
    )

    message = response["choices"][0]["message"]["content"]

    logger.info(message)
    return {"message": message}


if __name__ == "__main__":
    app.run()
