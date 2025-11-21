import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


# Примерная, не точная оценка количества токенов
def estimate_tokens_ollama(messages: list[dict], model_name: str) -> int:
    total_tokens = 0
    for m in messages:
        total_tokens += estimate_text_tokens(m.get("content", ""), model_name)
    return total_tokens


def estimate_text_tokens(text: str, model_name: str) -> int:
    word_count = len(text.split())
    char_count = len(text)

    if model_name.startswith("qwen"):
        return int(char_count / 2)  # Qwen сильно дробит Unicode
    elif model_name.startswith("mistral") or model_name.startswith("devstral"):
        return int(word_count * 1.5)
    elif model_name.startswith("phi") or model_name.startswith("gemma"):
        return int(word_count * 2)
    else:
        return int(word_count * 1.8)  # по умолчанию


def trim_messages_to_token_limit(messages: list[dict], model_name: str, max_tokens: int, step: int = 2) -> tuple[
    list[dict], int]:
    """
    Обрезает сообщения, начиная с самых старых (после system), пока суммарное число токенов не станет меньше max_tokens.
    Всегда удалять на количество кратное двойке, чтобы assistant не оказался первым после system
    """
    if not messages:
        return []

    # system сообщение оставляем всегда (если оно есть)
    system_msg = []
    user_msgs = messages

    if messages[0]["role"] == "system":
        system_msg = [messages[0]]
        user_msgs = messages[1:]

    # Идем справа налево (сначала всё оставляем)
    trimmed_msgs = list(user_msgs)

    total_tokens = estimate_tokens_ollama(system_msg + trimmed_msgs, model_name)
    logger.info(f"Оценка токенов: {total_tokens}, max_tokens: {max_tokens}")
    while total_tokens > max_tokens and trimmed_msgs:
        logger.info(f"Удаляем {step} самых старых сообщений")
        trimmed_msgs = trimmed_msgs[step:]
        total_tokens = estimate_tokens_ollama(system_msg + trimmed_msgs, model_name)
        logger.info(f"Оценка токенов: {total_tokens}, max_tokens: {max_tokens}")

    if total_tokens > max_tokens:
        raise HTTPException(status_code=500,
                            detail=f"Число токенов ({total_tokens}) превышает лимит ({max_tokens}), даже после обрезки.")

    return system_msg + trimmed_msgs, total_tokens
