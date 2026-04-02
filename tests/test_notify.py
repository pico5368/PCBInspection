"""Tests for alert notification system."""

from __future__ import annotations

from pcb_inspection.notify import AlertThresholds, check_health, check_and_notify, _last_status


class TestCheckHealth:
    def test_healthy_when_all_normal(self):
        stats = {"defect_rate": 2.0, "false_reject_rate": 1.0}
        result = check_health(stats)
        assert result["status"] == "healthy"
        assert len(result["alerts"]) == 0

    def test_warning_on_defect_rate(self):
        stats = {"defect_rate": 12.0, "false_reject_rate": 1.0}
        result = check_health(stats)
        assert result["status"] == "warning"
        assert any(a["metric"] == "defect_rate" for a in result["alerts"])

    def test_critical_on_defect_rate(self):
        stats = {"defect_rate": 25.0, "false_reject_rate": 1.0}
        result = check_health(stats)
        assert result["status"] == "critical"

    def test_critical_overrides_warning(self):
        stats = {"defect_rate": 25.0, "false_reject_rate": 6.0}
        result = check_health(stats)
        assert result["status"] == "critical"

    def test_custom_thresholds(self):
        thresholds = AlertThresholds(defect_rate_warning=5.0, defect_rate_critical=10.0)
        stats = {"defect_rate": 7.0, "false_reject_rate": 0}
        result = check_health(stats, thresholds)
        assert result["status"] == "warning"

    def test_false_reject_rate_alert(self):
        thresholds = AlertThresholds(false_reject_rate_warning=3.0)
        stats = {"defect_rate": 0, "false_reject_rate": 4.0}
        result = check_health(stats, thresholds)
        assert result["status"] == "warning"
        assert any(a["metric"] == "false_reject_rate" for a in result["alerts"])


class TestCheckAndNotify:
    def setup_method(self):
        _last_status.clear()

    def test_first_call_healthy_no_change(self):
        stats = {"defect_rate": 0, "false_reject_rate": 0}
        result = check_and_notify("test_session", stats)
        assert result["status"] == "healthy"
        assert "status_change" not in result  # No transition from healthy→healthy

    def test_transition_detected(self):
        # First call: healthy
        check_and_notify("sess1", {"defect_rate": 0, "false_reject_rate": 0})
        # Second call: critical
        result = check_and_notify("sess1", {"defect_rate": 25.0, "false_reject_rate": 0})
        assert result["status"] == "critical"
        assert "status_change" in result
        assert "healthy → critical" in result["status_change"]

    def test_no_spam_on_same_status(self):
        check_and_notify("sess2", {"defect_rate": 25.0, "false_reject_rate": 0})
        result = check_and_notify("sess2", {"defect_rate": 30.0, "false_reject_rate": 0})
        # Still critical, no transition
        assert "status_change" not in result

    def test_recovery_transition(self):
        check_and_notify("sess3", {"defect_rate": 25.0, "false_reject_rate": 0})
        result = check_and_notify("sess3", {"defect_rate": 1.0, "false_reject_rate": 0})
        assert result["status"] == "healthy"
        assert "status_change" in result
        assert "critical → healthy" in result["status_change"]

    def test_separate_sessions(self):
        check_and_notify("sessA", {"defect_rate": 25.0, "false_reject_rate": 0})
        result = check_and_notify("sessB", {"defect_rate": 25.0, "false_reject_rate": 0})
        # sessB's first critical should trigger transition
        assert "status_change" in result
