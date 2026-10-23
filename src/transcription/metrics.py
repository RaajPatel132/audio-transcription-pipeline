import re


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance over the number of reference words; case and punctuation
    are ignored. It can pass 1.0 when the hypothesis has many extra words."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise ValueError("the reference has no words")
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)
