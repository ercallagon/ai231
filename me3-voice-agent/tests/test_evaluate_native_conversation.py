from scripts.evaluate_native_conversation import terminal_status


def test_pilot_terminal_reason_distinguishes_failure_from_budget_exhaustion():
    assert terminal_status("RUNTIME_FAILURE") == "STOPPED_PROVIDER_OR_RUNTIME_ERROR"
    assert terminal_status("TERMINAL_FAILURE") == "STOPPED_TERMINAL_CASE_FAILURE"
    assert terminal_status("PENDING_INDEPENDENT_REVIEW") == "STOPPED_CUMULATIVE_GROQ_BUDGET"
