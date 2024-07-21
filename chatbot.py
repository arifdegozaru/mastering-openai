import openai
from dotenv import dotenv_values

config = dotenv_values(".env")

# openai.api_key = config['OPENAI_API_KEY']
openai.api_key = config['LM_API_KEY']
openai.base_url = config['LM_API_URL']


def main():
    messages = []

    while True:
        try:
            user_input = input("You: ")
            messages.append({"role": "user", "content": user_input})

            print("Assistant: ", end="", flush=True)
            assistant_response = ""
            for res in openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                stream=True
            ):
                if hasattr(res.choices[0].delta, 'content'):
                    delta_content = res.choices[0].delta.content
                    print(delta_content, end="", flush=True)
                    assistant_response += str(delta_content)

            messages.append({"role": "assistant", "content": assistant_response})
            print("\n")
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
            break

if __name__ == "__main__":
    main()