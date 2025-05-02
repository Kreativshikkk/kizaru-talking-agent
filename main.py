from model import Model


def main():
    bot = Model()
    print("Say 'exit' to stop the assistant.")

    while True:
        user_text = bot.listen()
        if user_text.lower() in ["exit.", "exit"]:
            break
        print(f"📝  You: {user_text}")
        answer = bot.get_response(user_text)
        # print(f"🤖  AI: {answer}")
        # answer = "Hello! Now you will see the presentation of 2 AI agents by Nikita Magomedeminov and Artem Grigorash. They will debate on the topic: 'Is AI a threat to humanity?'."
        bot.speak(answer)


if __name__ == "__main__":
    main()
