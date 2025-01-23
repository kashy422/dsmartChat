import argparse

from src.agent import repl_chat

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--session_id", type=str, default="default", help="Session ID", required=True)
args = parser.parse_args()


def run_repl_chat():
    repl_chat(session_id=args.session_id)


if __name__ == "__main__":
    run_repl_chat()
