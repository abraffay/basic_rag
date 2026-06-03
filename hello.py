import os

from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

history = [{"role": "system", "content": "You are a helpful tutor."}]
while True:
    input_text = input("Enter your question: ")
    history.append({"role": "user", "content": input_text})
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=history[-5:],
    )
    history.append({"role": "assistant", "content": response.output_text})
    print(response.output_text)
