from openai import OpenAI
import os

client = OpenAI(
    base_url='https://xiaoai.plus/v1',
    api_key='sk-ZWdhVK5dCaLEhZPPAgBdCETYkNws4TVic7JXVpNKhmrGR8fa',  
)

model_PATH = "model/emotion_model.h5"


DB_PATH = "emotion_chat.db"