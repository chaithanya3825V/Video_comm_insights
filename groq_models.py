from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

models = client.models.list()

print("\n=== AVAILABLE MODELS IN YOUR GROQ ACCOUNT ===\n")
for model in models.data:
    print(model.id)
