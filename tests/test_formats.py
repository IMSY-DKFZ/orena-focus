"""Tests for focus.data.formats."""

from datetime import timedelta

import pytest

from focus.data.formats import (
    Binary,
    FOClass,
    Matching,
    MultipleChoice,
    Number,
    OpenEnded,
    Percentage,
    Time,
    get_format_class,
    ts_to_seconds,
)

# ── ts_to_seconds ────────────────────────────────────────────────────


def test_ts_to_seconds_zero():
    assert ts_to_seconds("00:00:00") == 0.0


def test_ts_to_seconds_minutes():
    assert ts_to_seconds("00:01:30") == 90.0


def test_ts_to_seconds_hours():
    assert ts_to_seconds("01:00:00") == 3600.0


def test_ts_to_seconds_combined():
    assert ts_to_seconds("01:23:45") == 3600 + 23 * 60 + 45


# ── registry ─────────────────────────────────────────────────────────


def test_get_format_class_known():
    assert get_format_class("binary") is Binary
    assert get_format_class("number") is Number
    assert get_format_class("time") is Time


def test_get_format_class_unknown():
    with pytest.raises(ValueError, match="Unknown format type"):
        get_format_class("nonexistent")


# ── Binary ───────────────────────────────────────────────────────────


class TestBinary:
    def test_read_yes(self):
        assert Binary().read("yes") is True

    def test_read_no(self):
        assert Binary().read("no") is False

    def test_case_insensitive(self):
        assert Binary().read("YES") is True
        assert Binary().read("No") is False

    def test_verify_invalid(self):
        with pytest.raises(ValueError):
            Binary().verify("maybe")

    def test_compare(self):
        fmt = Binary()
        assert fmt.compare(True, True)
        assert not fmt.compare(True, False)


# ── Number ───────────────────────────────────────────────────────────


class TestNumber:
    def test_read_integer(self):
        assert Number().read("42") == 42

    def test_read_zero(self):
        assert Number().read("0") == 0

    def test_verify_negative(self):
        with pytest.raises(ValueError):
            Number().verify("-1")

    def test_verify_float(self):
        with pytest.raises(ValueError):
            Number().verify("3.14")

    def test_verify_text(self):
        with pytest.raises(ValueError):
            Number().verify("abc")

    def test_compare(self):
        fmt = Number()
        assert fmt.compare(5, 5)
        assert not fmt.compare(5, 6)


# ── Percentage ───────────────────────────────────────────────────────


class TestPercentage:
    def test_read_plain(self):
        assert Percentage().read("50") == 50.0

    def test_read_with_symbol(self):
        assert Percentage().read("75%") == 75.0

    def test_read_decimal(self):
        assert Percentage().read("12.5") == 12.5

    def test_verify_invalid(self):
        with pytest.raises(ValueError):
            Percentage().verify("not a number")

    def test_compare_exact(self):
        fmt = Percentage()
        assert fmt.compare(50.0, 50.0)
        assert not fmt.compare(50.0, 51.0)


# ── FOClass ──────────────────────────────────────────────────────────


class TestFOClass:
    def setup_method(self):
        self.fmt = FOClass(valid_names=("Sponge", "Clip", "Needle"))

    def test_read_exact(self):
        assert self.fmt.read("Sponge") == frozenset({"Sponge"})

    def test_read_case_insensitive(self):
        assert self.fmt.read("sponge") == frozenset({"Sponge"})
        assert self.fmt.read("CLIP") == frozenset({"Clip"})

    def test_verify_invalid(self):
        with pytest.raises(ValueError):
            self.fmt.verify("Staple")

    def test_empty_valid_names(self):
        fmt = FOClass(valid_names=())
        with pytest.raises(ValueError):
            fmt.verify("Sponge")

    def test_none_answer_always_valid(self):
        assert self.fmt.read("none") == frozenset({"None"})
        assert self.fmt.read("None") == frozenset({"None"})
        assert self.fmt.read("NONE") == frozenset({"None"})

    def test_none_valid_even_with_empty_names(self):
        fmt = FOClass(valid_names=())
        assert fmt.read("none") == frozenset({"None"})

    def test_none_compare(self):
        assert self.fmt.compare(frozenset({"None"}), frozenset({"None"}))
        assert not self.fmt.compare(frozenset({"None"}), frozenset({"Sponge"}))


    # ── multi-class-behavior ────────────────────────────────────────────────────────


    def test_read_multi_class(self):
        assert self.fmt.read("Clip, Sponge") == frozenset({"Clip", "Sponge"})

    def test_read_multi_class_order_independent(self):
        assert self.fmt.read("Sponge, Clip") == self.fmt.read("Clip, Sponge")
        assert self.fmt.compare(
            self.fmt.read("Sponge, Clip"), self.fmt.read("Clip, Sponge")
        )

    def test_read_multi_class_dedups_and_normalises(self):
        assert self.fmt.read("clip, CLIP, sponge") == frozenset({"Clip", "Sponge"})

    def test_compare_distinguishes_subset(self):
        assert not self.fmt.compare(
            self.fmt.read("Clip"), self.fmt.read("Clip, Sponge")
        )

    def test_none_cannot_combine_with_class(self):
        with pytest.raises(ValueError):
            self.fmt.verify("none, Clip")

    def test_empty_answer_invalid(self):
        with pytest.raises(ValueError):
            self.fmt.verify("")
        with pytest.raises(ValueError):
            self.fmt.verify(" , ")


# ── OpenEnded ────────────────────────────────────────────────────────


class TestOpenEnded:
    def test_read_strips_whitespace(self):
        assert OpenEnded().read("  hello  ") == "hello"

    def test_verify_too_long(self):
        with pytest.raises(ValueError):
            OpenEnded(max_length=5).verify("too long answer")

    def test_verify_at_limit(self):
        OpenEnded(max_length=5).verify("12345")


# ── Matching ─────────────────────────────────────────────────────────


class TestMatching:
    def test_read_matching_pattern(self):
        fmt = Matching(pattern=r"\d{2}:\d{2}")
        assert fmt.read("12:34") == "12:34"

    def test_verify_non_matching(self):
        fmt = Matching(pattern=r"\d+")
        with pytest.raises(ValueError):
            fmt.verify("abc")

    def test_verify_too_long(self):
        fmt = Matching(pattern=r".*", max_length=3)
        with pytest.raises(ValueError):
            fmt.verify("toolong")


# ── MultipleChoice ───────────────────────────────────────────────────


class TestMultipleChoice:
    def test_read_strips_whitespace(self):
        assert MultipleChoice().read("  B  ") == "B"

    def test_verify_too_long(self):
        with pytest.raises(ValueError):
            MultipleChoice(max_length=3).verify("toolong")

    def test_is_judge_format(self):
        from focus.data.formats import JUDGE_FORMATS

        assert "multiple_choice" in JUDGE_FORMATS


# ── Time ─────────────────────────────────────────────────────────────


class TestTime:
    def test_read_returns_timedelta_tuple(self):
        result = Time().read("01:23:45")
        assert result == (timedelta(hours=1, minutes=23, seconds=45),)

    def test_verify_invalid_format(self):
        with pytest.raises(ValueError):
            Time().verify("1:2:3")

    def test_verify_invalid_minutes(self):
        with pytest.raises(ValueError):
            Time().verify("00:60:00")

    def test_verify_invalid_seconds(self):
        with pytest.raises(ValueError):
            Time().verify("00:00:60")

    def test_compare_within_threshold(self):
        fmt = Time(threshold_seconds=5.0)
        a = (timedelta(hours=1, minutes=0, seconds=0),)
        b = (timedelta(hours=1, minutes=0, seconds=4),)
        assert fmt.compare(a, b)

    def test_compare_outside_threshold(self):
        fmt = Time(threshold_seconds=5.0)
        a = (timedelta(hours=1, minutes=0, seconds=0),)
        b = (timedelta(hours=1, minutes=0, seconds=6),)
        assert not fmt.compare(a, b)

    def test_compare_non_timedelta(self):
        assert not Time().compare("string", "string")
        assert not Time().compare(("string",), ("string",))


    # ── multi-time-behavior ─────────────────────────────────────────────────────


    def test_read_multi_time(self):
        assert Time().read("00:12:30, 00:47:05") == (
            timedelta(minutes=12, seconds=30),
            timedelta(minutes=47, seconds=5),
        )

    def test_read_multi_time_whitespace_tolerant(self):
        assert Time().read(" 00:12:30 ,00:47:05 ") == Time().read("00:12:30, 00:47:05")

    def test_read_multi_time_sorted_chronologically(self):
        assert Time().read("00:47:05, 00:12:30") == Time().read("00:12:30, 00:47:05")

    def test_compare_multi_time_exact(self):
        fmt = Time(threshold_seconds=5.0)
        assert fmt.compare(fmt.read("00:12:30, 00:47:05"), fmt.read("00:12:30, 00:47:05"))

    def test_compare_multi_time_within_threshold(self):
        fmt = Time(threshold_seconds=5.0)
        assert fmt.compare(fmt.read("00:12:30, 00:47:05"), fmt.read("00:12:34, 00:47:01"))

    def test_compare_multi_time_outside_threshold(self):
        fmt = Time(threshold_seconds=5.0)
        assert not fmt.compare(fmt.read("00:12:30, 00:47:05"), fmt.read("00:12:30, 00:47:11"))

    def test_compare_multi_time_order_independent(self):
        fmt = Time(threshold_seconds=5.0)
        assert fmt.compare(fmt.read("00:47:05, 00:12:30"), fmt.read("00:12:30, 00:47:05"))

    def test_compare_count_mismatch(self):
        fmt = Time(threshold_seconds=5.0)
        assert not fmt.compare(fmt.read("00:12:30"), fmt.read("00:12:30, 00:47:05"))

    def test_verify_invalid_part(self):
        with pytest.raises(ValueError):
            Time().verify("00:12:30, 1:2:3")

    def test_empty_answer_invalid(self):
        with pytest.raises(ValueError):
            Time().verify("")
        with pytest.raises(ValueError):
            Time().verify(" , ")
