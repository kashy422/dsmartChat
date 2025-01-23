from colorama import Fore, Style


def write(text: str, role: str = "user"):
    if role == "assistant":
        prGreen(f"Assistant: {text}")
    else:
        print(f"User: {text}")


def prGreen(skk):
    print(f"{Fore.GREEN}{Style.BRIGHT}{skk}{Style.RESET_ALL}")
