"""Cięcie tekstu na „słowa” przed BPE.

Pojedyncza spacja dokleja się do następnego słowa (jak w GPT-2), pozostałe białe znaki
tworzą osobne kawałki. BPE skleja bajty tylko wewnątrz jednego kawałka.
"""
import re

PAT = re.compile(r" ?\S+|\s+")


def pretokenize(text: str) -> list[str]:
    return PAT.findall(text)
