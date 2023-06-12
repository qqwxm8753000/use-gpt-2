import argparse
import tensorflow as tf
from transformers import AutoTokenizer, TFGPT2LMHeadModel


def generate_text(model, tokenizer, prompt_text, max_length):
    # Encode prompt text
    input_ids = tokenizer.encode(prompt_text, return_tensors='tf')

    # Generate text
    sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=max_length, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1
    )

    # Decode generated text
    generated_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

    return generated_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='model', help='Directory containing the saved model')
    parser.add_argument('--max-length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--prompt-text', type=str, default='今天天气很好', help='Prompt text for generation')
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-gpt2')
    model = TFGPT2LMHeadModel.from_pretrained(args.model_dir)

    # Generate text
    generated_text = generate_text(model, tokenizer, args.prompt_text, args.max_length)

    # Save generated text to file
    with open('generated_text.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
