def isiterable(obj) -> bool:
    return not isinstance(obj, (str, bytes)) and isinstance(obj, list)
