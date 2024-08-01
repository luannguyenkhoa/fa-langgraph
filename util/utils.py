def strip_final_answer(answer: str):
    key = 'Answer:'
    items = answer.split(key)
    if len(items) == 2:
        return items[1].strip()
    if len(items) == 1:
        return items[0].strip()
    return answer.replace(key, '').strip()