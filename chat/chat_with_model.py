from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 학습된 모델과 토크나이저 로드
def load_model(model_path):
    try:
        print("Loading model...")
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# 채팅 함수 구현
def generate_response(prompt, model, tokenizer, max_length=100):
    try:
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an error."

# 채팅 인터페이스
def chat_with_model(model_path='./finetuned_gpt2'):
    model, tokenizer = load_model(model_path)
    if model is None or tokenizer is None:
        print("Failed to load model and tokenizer.")
        return

    print("Chatbot is ready to chat! Type 'exit' to end the conversation.")
    while True:
        prompt = input("You: ")
        if prompt.lower() == 'exit':
            break
        response = generate_response(prompt, model, tokenizer)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat_with_model()
