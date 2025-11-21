import re


def to_camel_case(s: str) -> str:
    """snake_case → camelCase"""
    return re.sub(r'_([a-z])', lambda m: m.group(1).upper(), s)


def snake_to_camel_dict(data):
    """
    Рекурсивно конвертирует ключи dict/списков из snake_case в camelCase.
    Сохраняет типы и структуру данных.
    """
    if isinstance(data, list):
        return [snake_to_camel_dict(item) for item in data]
    elif isinstance(data, dict):
        return {
            to_camel_case(k): snake_to_camel_dict(v)
            for k, v in data.items()
        }
    else:
        return data
