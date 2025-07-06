from transformers import AutoTokenizer, AutoModelForCausalLM

class RAG:
    def __init__(self, generator_model="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(generator_model)
        self.model = AutoModelForCausalLM.from_pretrained(generator_model)
    
    def generate(self, question, contexts):
        context_text = "\n".join(contexts)
        prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
