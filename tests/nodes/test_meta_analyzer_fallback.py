# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for meta_analyzer heuristic fallback filter (--no-llm mode)."""

from __future__ import annotations

from skillspector.models import Finding
from skillspector.nodes.meta_analyzer import _fallback_filtered


def _finding(
    rule_id: str = "TM1",
    confidence: float = 0.8,
    context: str | None = "import subprocess\nsubprocess.run(cmd, shell=True)",
    matched_text: str = "subprocess.run(cmd, shell=True)",
    file: str = "tool.py",
) -> Finding:
    return Finding(
        rule_id=rule_id,
        message=f"Test {rule_id}",
        severity="HIGH",
        confidence=confidence,
        file=file,
        start_line=1,
        context=context,
        matched_text=matched_text,
    )


class TestConfidenceThreshold:
    """Findings below confidence threshold are dropped."""

    def test_low_confidence_dropped(self) -> None:
        """Finding with confidence 0.3 is below threshold and dropped."""
        findings = [_finding(confidence=0.3)]
        result = _fallback_filtered(findings)
        assert len(result) == 0

    def test_threshold_boundary_dropped(self) -> None:
        """Finding with confidence exactly 0.39 is dropped (< 0.4)."""
        findings = [_finding(confidence=0.39)]
        result = _fallback_filtered(findings)
        assert len(result) == 0

    def test_at_threshold_kept(self) -> None:
        """Finding with confidence exactly 0.4 is kept (>= 0.4)."""
        findings = [_finding(confidence=0.4)]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_high_confidence_kept(self) -> None:
        """Finding with high confidence passes through."""
        findings = [_finding(confidence=0.9)]
        result = _fallback_filtered(findings)
        assert len(result) == 1


class TestCodeExampleFiltering:
    """Findings in code example context are dropped."""

    def test_fenced_code_block_context_dropped(self) -> None:
        """Finding whose context contains ``` (fenced code block) is dropped."""
        findings = [
            _finding(
                context="```bash\ncurl -k https://api.example.com\n```",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 0

    def test_example_keyword_context_dropped(self) -> None:
        """Finding whose context contains 'example:' is dropped."""
        findings = [
            _finding(
                context="Example: how to use subprocess\nsubprocess.run(cmd)",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 0

    def test_normal_code_context_kept(self) -> None:
        """Finding with regular code context (no example indicators) passes."""
        findings = [
            _finding(
                context="import subprocess\nresult = subprocess.run(cmd, shell=True)",
                confidence=0.8,
            )
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 1

    def test_no_context_kept(self) -> None:
        """Finding with no context (None) passes through."""
        findings = [_finding(context=None, confidence=0.8)]
        result = _fallback_filtered(findings)
        assert len(result) == 1


class TestCombinedFiltering:
    """Both filters work together."""

    def test_mixed_findings_filtered(self) -> None:
        """Mix of low-confidence, code-example, and genuine findings."""
        findings = [
            _finding(confidence=0.2),  # dropped: low confidence
            _finding(
                confidence=0.8,
                context="```\ncurl -k https://example.com\n```",
            ),  # dropped: code example
            _finding(confidence=0.8),  # kept: genuine finding
            _finding(confidence=0.6),  # kept: above threshold, normal context
        ]
        result = _fallback_filtered(findings)
        assert len(result) == 2

    def test_remediation_applied(self) -> None:
        """Kept findings get default remediation if none set."""
        findings = [_finding(confidence=0.8)]
        result = _fallback_filtered(findings)
        assert len(result) == 1
        assert result[0].remediation is not None
        assert len(result[0].remediation) > 0

    def test_empty_input(self) -> None:
        """Empty findings list returns empty."""
        assert _fallback_filtered([]) == []
