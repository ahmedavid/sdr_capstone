from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters
import torch

# Load the Tolkien model
tolkien_model_name = "jeremyarancio/llm-tolkien"
tolkien_tokenizer = AutoTokenizer.from_pretrained(tolkien_model_name)
tolkien_model = AutoModelForCausalLM.from_pretrained(tolkien_model_name)
tolkien_generator = pipeline("text-generation", model=tolkien_model, tokenizer=tolkien_tokenizer)

# Store conversation histories per user
user_histories = {}

# Define the custom keyboard
menu_keyboard = [["/start", "/clear"]]
menu_markup = ReplyKeyboardMarkup(menu_keyboard, one_time_keyboard=True, resize_keyboard=True)

async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send welcome message when user sends /start command"""
    user_id = update.effective_user.id
    welcome_text = (
        "🌟 Welcome to The Prancing Pony Tale Teller! 🌟\n\n"
        "I'm Goldberry and I'm here to craft a magical story in Middle Earth with your help.\n"
        "Just send me a story prompt or idea, and I'll continue the narrative!\n\n"
        "Type something to begin our adventure..."
    )
    
    # Reset history for new users or existing users starting fresh
    user_histories[user_id] = []
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=welcome_text,
        reply_markup=menu_markup  # Show the menu keyboard
    )

async def handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear the conversation history for the user"""
    user_id = update.effective_user.id
    
    # Reset the user's history
    user_histories[user_id] = []
    
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="🗑️ Your conversation history has been cleared. Let's start a new adventure!",
        reply_markup=menu_markup  # Show the menu keyboard
    )

def generate_response(user_id, input_text):
    # Get or create history for user
    history = user_histories.get(user_id, [])
    
    # Combine history with new input
    context = " ".join([entry["content"] for entry in history] + [input_text])
    
    # Generate story continuation
    generated_text = tolkien_generator(
        context,
        max_length=150,
        temperature=0.8,
        do_sample=True,
        pad_token_id=tolkien_tokenizer.eos_token_id
    )[0]['generated_text']
    
    continuation = generated_text[len(context):].strip()
    
    # Update history
    history.extend([
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": continuation}
    ])
    
    # Store updated history
    user_histories[user_id] = history
    
    return continuation

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_input = update.message.text
    
    # Check if the input is a command (e.g., /start or /clear)
    if user_input in ["/start", "/clear"]:
        # Let the command handler deal with it
        return
    
    # Generate response
    response = generate_response(user_id, user_input)
    
    # Send response back to user
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=response,
        reply_markup=menu_markup  # Show the menu keyboard after each message
    )

if __name__ == '__main__':
    # Set up Telegram bot
    application = ApplicationBuilder().token('telegram api key').build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("clear", handle_clear))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start polling
    print("Bot is running...")
    application.run_polling()