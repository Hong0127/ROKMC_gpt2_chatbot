# GPT-2 Finetune Chatbot

This project demonstrates how to fine-tune a GPT-2 model using `.hwp` and `.pdf` files and deploy it for a simple chat application.

TODO.
 1. train koQuad dataset / KLUE dataset

## Setup

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/gpt2_finetune_chatbot.git
    cd gpt2_finetune_chatbot
    ```

2. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Create a dataset from your documents:
    ```sh
    python data/create_dataset.py
    ```

4. Fine-tune the GPT-2 model:
    ```sh
    python model/train_model.py
    ```

5. Start chatting with the fine-tuned model:
    ```sh
    python chat/chat_with_model.py
    ```

## Project Structure

- `data/create_dataset.py`: Script for extracting text from `.hwp` and `.pdf` files and creating a training dataset.
- `model/train_model.py`: Script for fine-tuning the GPT-2 model using the generated dataset.
- `chat/chat_with_model.py`: Script for loading the fine-tuned model and interacting with it through a chat interface.
- `requirements.txt`: List of required dependencies.
- `README.md`: Project overview and setup instructions.

## Notes

- Ensure that your file paths in the scripts are correctly set to your local paths where the `.hwp` and `.pdf` files are stored.
- Modify the training arguments in `train_model.py` to suit your hardware and dataset size.
