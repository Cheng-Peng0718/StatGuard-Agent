import time


def typewriter_effect(text, speed=0.015):
    """Character-by-character typewriter streaming."""
    for char in text:
        yield char
        time.sleep(speed)