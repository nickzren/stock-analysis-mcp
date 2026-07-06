"""Tests for the vendored NYSE market calendar (2025-2030)."""

from datetime import date, datetime

import pytz

from stock_analysis.utils.market_calendar import (
    CALENDAR_FIRST_YEAR,
    CALENDAR_LAST_YEAR,
    EARLY_CLOSES,
    FULL_HOLIDAYS,
    classify_session,
    most_recent_trading_day,
    regular_session_minutes,
    session_method,
)

ET = pytz.timezone("America/New_York")


def at(y: int, m: int, d: int, hh: int, mm: int = 0) -> datetime:
    return ET.localize(datetime(y, m, d, hh, mm))


class TestTables:
    def test_precedence_no_overlap(self) -> None:
        assert frozenset() == (FULL_HOLIDAYS & EARLY_CLOSES)

    def test_holiday_counts_per_year(self) -> None:
        counts = {y: sum(1 for h in FULL_HOLIDAYS if h.year == y)
                  for y in range(CALENDAR_FIRST_YEAR, CALENDAR_LAST_YEAR + 1)}
        # 2028: Saturday New Year's is NOT observed (9 holidays).
        assert counts == {2025: 10, 2026: 10, 2027: 10, 2028: 9, 2029: 10, 2030: 10}

    def test_early_close_counts_per_year(self) -> None:
        counts = {y: sum(1 for h in EARLY_CLOSES if h.year == y)
                  for y in range(CALENDAR_FIRST_YEAR, CALENDAR_LAST_YEAR + 1)}
        # 2026 loses Jul 3 (observed holiday); 2027 loses Jul 3 (Saturday)
        # AND Dec 24 (observed Christmas); 2028 loses Dec 24 (Sunday).
        assert counts == {2025: 3, 2026: 2, 2027: 1, 2028: 2, 2029: 3, 2030: 3}

    def test_observed_shift_examples(self) -> None:
        assert date(2026, 7, 3) in FULL_HOLIDAYS      # Jul 4 Saturday -> Fri
        assert date(2027, 7, 5) in FULL_HOLIDAYS      # Jul 4 Sunday -> Mon
        assert date(2027, 6, 18) in FULL_HOLIDAYS     # Juneteenth Sat -> Fri
        assert date(2027, 12, 24) in FULL_HOLIDAYS    # Xmas Saturday -> Fri
        assert date(2028, 1, 1) not in FULL_HOLIDAYS  # Saturday NY not observed
        assert date(2027, 12, 31) not in FULL_HOLIDAYS


class TestClassification:
    def test_full_holiday_closed_all_day(self) -> None:
        # Owner-corrected precedence: 2026-07-03 is the observed closure.
        for hh in (8, 11, 14, 18):
            assert classify_session(at(2026, 7, 3, hh)) == "closed"

    def test_observed_xmas_beats_xmas_eve_early_close(self) -> None:
        assert classify_session(at(2027, 12, 24, 11)) == "closed"

    def test_early_close_ladder(self) -> None:
        d = (2025, 11, 28)  # day after Thanksgiving
        assert classify_session(at(*d, 8)) == "pre_market"
        assert classify_session(at(*d, 12, 59)) == "regular"
        assert classify_session(at(*d, 13)) == "after_hours"
        assert classify_session(at(*d, 16, 59)) == "after_hours"
        assert classify_session(at(*d, 17)) == "closed"

    def test_normal_day_ladder_unchanged(self) -> None:
        d = (2026, 3, 10)  # ordinary Tuesday
        assert classify_session(at(*d, 3)) == "closed"
        assert classify_session(at(*d, 8)) == "pre_market"
        assert classify_session(at(*d, 11, 30)) == "regular"
        assert classify_session(at(*d, 18, 30)) == "after_hours"
        assert classify_session(at(*d, 21)) == "closed"

    def test_outside_coverage_both_sides_clock_only(self) -> None:
        # 2024-07-04 was a real NYSE holiday, but it PREDATES coverage:
        # clock-only fallback classifies the Thursday as regular at 11:30.
        assert classify_session(at(2024, 7, 4, 11, 30)) == "regular"
        assert session_method(at(2024, 7, 4, 11, 30)) == "clock_only_no_holidays_fallback"
        assert classify_session(at(2031, 1, 1, 11, 30)) == "regular"  # Wed
        assert session_method(at(2031, 1, 1, 11, 30)) == "clock_only_no_holidays_fallback"
        assert session_method(at(2026, 3, 10, 11, 30)) == "calendar_static"

    def test_weekend_still_closed_everywhere(self) -> None:
        assert classify_session(at(2026, 7, 4, 11)) == "closed"   # Saturday
        assert classify_session(at(2024, 7, 6, 11)) == "closed"   # pre-coverage Sat


class TestSessionMinutes:
    def test_early_close_is_210(self) -> None:
        assert regular_session_minutes(date(2025, 11, 28)) == 210
        assert regular_session_minutes(date(2026, 11, 27)) == 210

    def test_normal_and_outside_coverage_390(self) -> None:
        assert regular_session_minutes(date(2026, 3, 10)) == 390
        assert regular_session_minutes(date(2024, 11, 29)) == 390  # pre-coverage


class TestWalkBack:
    def test_hood_case_monday_premarket_skips_holiday_friday(self) -> None:
        # The real false-downgrade case: Mon 2026-07-06 pre_market must
        # expect Thu 2026-07-02 (Fri 07-03 = observed holiday).
        now = at(2026, 7, 6, 8)
        assert most_recent_trading_day(now, "pre_market") == date(2026, 7, 2)

    def test_regular_day_is_same_day(self) -> None:
        assert most_recent_trading_day(at(2026, 3, 10, 11), "regular") == date(2026, 3, 10)

    def test_weekend_walks_to_friday(self) -> None:
        assert most_recent_trading_day(at(2026, 3, 8, 12), "closed") == date(2026, 3, 6)

    def test_day_after_holiday_monday(self) -> None:
        # Tue 2026-01-20 pre_market: Mon 01-19 is MLK -> expect Fri 01-16.
        assert most_recent_trading_day(at(2026, 1, 20, 8), "pre_market") == date(2026, 1, 16)

    def test_outside_coverage_weekday_walkback_only(self) -> None:
        # Pre-coverage: 2024-07-05 Fri pre_market expects Thu 2024-07-04
        # (clock-only fallback cannot know it was a holiday).
        assert most_recent_trading_day(at(2024, 7, 5, 8), "pre_market") == date(2024, 7, 4)
