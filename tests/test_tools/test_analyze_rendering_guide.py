"""Tests for the on-demand analyze rendering guide resource."""

import inspect

from stock_analysis.resources.analyze_guide import read_analyze_rendering_guide
from stock_analysis.server import analyze, get_analyze_rendering_guide


def test_analyze_docstring_points_to_resource_without_embedded_guide() -> None:
    doc = inspect.getdoc(analyze) or ""

    assert "stock-analysis://guides/analyze-rendering" in doc
    assert "DIP ASSESSMENT" not in doc
    assert len(doc) < 1_200


def test_analyze_rendering_guide_resource_contains_expected_sections() -> None:
    guide = get_analyze_rendering_guide()

    assert guide.startswith("# Analyze Rendering Guide")
    assert "Dislocation framework" in guide
    assert "Dip assessment" in guide
    assert "Decision context" in guide
    assert "Unprofitable Companies" in guide


def test_read_analyze_rendering_guide_returns_markdown_mime_type() -> None:
    guide, mime_type = read_analyze_rendering_guide()

    assert mime_type == "text/markdown"
    assert "Use this guide when presenting the result of the `analyze` tool." in guide

