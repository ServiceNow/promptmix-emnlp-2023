"""This script has code for prompting GPT like models"""
import os, openai, time

pjoin = os.path.join


def openai_complete(
    prompt,
    n,
    engine,
    temp,
    top_p,
    max_tokens=256,
    echo=False,
    logprobs=None,
    max_retries=3,
):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    while True:
        try:
            completion = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=max_tokens,
                n=1 if n is None else n,
                stop=None if n is None else ["\n"],
                temperature=temp,
                echo=echo,
                logprobs=logprobs,
                top_p=1 if not top_p else top_p,
            )
            return (
                completion.choices
                if logprobs
                else [c.text.strip() for c in completion.choices]
            )
        except openai.error.RateLimitError as e:
            print("Sleeping for 100 seconds...")
            time.sleep(100)
        except Exception as e:
            raise e


def openai_chat_complete(
    prompt,
    n,
    engine,
    temp,
    top_p,
    max_tokens=256,
    echo=False,
    logprobs=None,
    max_retries=3,
):
    if engine in [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
    ]:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_base = "https://api.endpoints.anyscale.com/v1"
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    while True:
        try:
            completion = openai.ChatCompletion.create(
                model=engine,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                n=1 if n is None else n,
                stop=None if n is None else ["\n"],
                temperature=temp,
                # echo=echo,
                # logprobs=logprobs,
                top_p=1 if not top_p else top_p,
            )
            return [c.message.content.strip() for c in completion.choices]
        except openai.error.RateLimitError as e:
            print("RateLimitError, Sleeping for 100 seconds...")
            time.sleep(100)
        except openai.error.APIError as e:
            print(f"APIError, {e}\nSleeping for 100 seconds...")
            time.sleep(100)
        except Exception as e:
            print(f"{e}, Sleeping for 100 seconds...")
            time.sleep(100)
