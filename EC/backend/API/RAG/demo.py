from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


# Load the text-generation pipeline
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
generator.model.to("cpu")


app = Flask(__name__)


@app.route("/generate", methods=["POST"])
def generate_text():
    # Get the prompt from the request body
    prompt = request.json.get("prompt")


    # Generate text using the pipeline
    response = generator(prompt)


    # Return the generated text as JSON
    return jsonify({"text": response[0]["generated_text"]})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8026)