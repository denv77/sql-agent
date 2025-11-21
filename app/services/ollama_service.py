from ollama import AsyncClient

ollama_client = AsyncClient()


async def run_chat(model: str, messages: list[dict], think: bool, num_ctx: int):
    stream_sql = await ollama_client.chat(
        stream=True,
        model=model,
        messages=messages,
        think=think,
        options={
            'num_ctx': num_ctx,
            'temperature': 0.2,
            'top_p': 0.9
        }
    )

    response = ''

    async for chunk in stream_sql:
        # content = chunk['message']['content']
        # print(content, end='', flush=True)
        # response += content

        if chunk.message.thinking:
            thinking = chunk.message.thinking
            print(thinking, end='', flush=True)
        else:
            content = chunk.message.content
            print(content, end='', flush=True)
            response += content

    print("\n")
    return response
