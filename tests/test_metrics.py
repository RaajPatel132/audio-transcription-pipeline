import pytest

from transcription.metrics import word_error_rate, words


def test_words_ignore_case_and_punctuation_but_keep_apostrophes() -> None:
    assert words("Don't STOP, we're nearly there!") == ["don't", "stop", "we're", "nearly", "there"]


def test_words_of_text_without_letters_is_empty() -> None:
    assert words("  ... 123 --  ") == []


def test_identical_text_has_no_errors() -> None:
    assert word_error_rate("the garden in spring", "the garden in spring") == 0


def test_case_and_punctuation_do_not_count_as_errors() -> None:
    assert word_error_rate("The garden, in spring.", "the GARDEN in spring") == 0


@pytest.mark.parametrize(
    ("hypothesis", "errors"),
    [
        ("the garden in summer", 1),
        ("the garden in the spring", 1),
        ("the garden spring", 1),
        ("a garden in summer", 2),
        ("the in garden spring", 2),
    ],
)
def test_each_substitution_insertion_and_deletion_costs_one(hypothesis: str, errors: int) -> None:
    assert word_error_rate("the garden in spring", hypothesis) == errors / 4


def test_empty_hypothesis_misses_every_word() -> None:
    assert word_error_rate("the garden in spring", "") == 1


def test_extra_words_can_push_the_rate_past_one() -> None:
    assert word_error_rate("hello", "oh hello there my friend") == 4


def test_reference_without_words_is_an_error() -> None:
    with pytest.raises(ValueError, match="no words"):
        word_error_rate("...", "anything")
