import torch
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

# Load the "ambig_qa" dataset
dataset = load_dataset("ambig_qa")

print("Dataset keys:", dataset.keys())

# Print some example questions and answers
print("Example questions and answers:")
for example in dataset["train"][:5]:
    question = example["question"]
    answers = example["answers"]
    print("Question:", question)
    print("Answers:", answers)
    print()

# Prepare data for fine-tuning DialoGPT
conversations = []
for example in dataset["train"]:
    context = example["question"]
    # Check if the answers are available and non-empty
    if "answers" in example and example["answers"]:
        responses = example["answers"]["text"]
        for response in responses:
            conversation = f"User: {context} Bot: {response}"
            conversations.append(conversation)

if not conversations:
    print("No conversations found in the dataset. Exiting.")
    exit()

print("Conversations:", conversations)

# Load pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # or any other size of DialoGPT
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Set padding token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize the conversations
tokenized_conversations = tokenizer(conversations, truncation=True, padding=True)

if not tokenized_conversations:
    print("No tokenized conversations generated. Exiting.")
    exit()

print("Tokenized conversations:", tokenized_conversations)

# Fine-tuning settings
training_args = TrainingArguments(
    output_dir="./chatbot_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_conversations,
)

trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_chatbot_dialogpt_ambig_qa")
tokenizer.save_pretrained("./fine_tuned_chatbot_dialogpt_ambig_qa")


# Load the fine-tuned DialoGPT model and tokenizer
model_path = "./fine_tuned_chatbot_dialogpt_ambig_qa"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Start a conversation loop
print("Welcome to the chatbot! Type 'exit' to end the conversation.")
while True:
    # Get user input
    user_input = input("You: ")

    # Check if user wants to exit
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Tokenize user input
    input_ids = tokenizer.encode("User: " + user_input, return_tensors="pt")

    # Generate response
    response_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode and print response
    bot_response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Chatbot:", bot_response)
