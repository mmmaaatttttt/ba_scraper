def truncate(s, num_chars=30):
    """Truncate a string to the specified number of characters.

    If truncation occurs, include an ellipsis at the end.

    >>> truncate("Hello world", 5)
    'He...'

    >>> truncate("Hello world")
    'Hello world'

    >>> truncate("Hello world", 11)
    'Hello world'

    >>> truncate("Hello world", 10)
    'Hello w...'

    """
    if len(s) <= num_chars:
        return s

    return s[:num_chars - 3] + "..."
