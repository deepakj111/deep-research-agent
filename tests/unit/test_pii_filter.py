# tests/unit/test_pii_filter.py
"""
Unit tests for the PII filtering middleware.

Tests cover each PII pattern independently and in combination,
ensuring correct redaction and audit trail generation.
"""

from agent.middleware.pii_filter import RedactionResult, filter_pii, filter_pii_simple


class TestSSNRedaction:
    def test_standard_ssn_format(self) -> None:
        result = filter_pii("Contact SSN: 123-45-6789 for details.")
        assert "[SSN REDACTED]" in result.text
        assert "123-45-6789" not in result.text
        assert result.redaction_counts.get("ssn") == 1

    def test_multiple_ssns(self) -> None:
        text = "SSN1: 111-22-3333, SSN2: 444-55-6666"
        result = filter_pii(text)
        assert result.redaction_counts.get("ssn") == 2
        assert result.total_redactions >= 2


class TestCreditCardRedaction:
    def test_16_digit_card(self) -> None:
        result = filter_pii("Card number: 4111111111111111 on file.")
        assert "[CARD REDACTED]" in result.text
        assert "4111111111111111" not in result.text
        assert result.redaction_counts.get("credit_card") == 1

    def test_card_with_spaces(self) -> None:
        result = filter_pii("Card: 4111 1111 1111 1111")
        assert "[CARD REDACTED]" in result.text

    def test_card_with_dashes(self) -> None:
        result = filter_pii("Card: 4111-1111-1111-1111")
        assert "[CARD REDACTED]" in result.text


class TestEmailRedaction:
    def test_standard_email(self) -> None:
        result = filter_pii("Send to user@example.com for info.")
        assert "[EMAIL REDACTED]" in result.text
        assert "user@example.com" not in result.text
        assert result.redaction_counts.get("email") == 1

    def test_email_with_plus(self) -> None:
        result = filter_pii("Contact user+tag@domain.org")
        assert "[EMAIL REDACTED]" in result.text

    def test_no_false_positive_on_at_sign(self) -> None:
        # "@" alone should not trigger email redaction
        result = filter_pii("Use @ symbol in commands")
        assert result.redaction_counts.get("email", 0) == 0


class TestPhoneRedaction:
    def test_us_phone_dashes(self) -> None:
        result = filter_pii("Call 555-123-4567 today.")
        assert "[PHONE REDACTED]" in result.text
        assert "555-123-4567" not in result.text

    def test_us_phone_dots(self) -> None:
        result = filter_pii("Phone: 555.123.4567")
        assert "[PHONE REDACTED]" in result.text

    def test_phone_with_country_code(self) -> None:
        result = filter_pii("Call +1-555-123-4567")
        assert "[PHONE REDACTED]" in result.text


class TestIPRedaction:
    def test_standard_ipv4(self) -> None:
        result = filter_pii("Server at 192.168.1.100 responded.")
        assert "[IP REDACTED]" in result.text
        assert "192.168.1.100" not in result.text
        assert result.redaction_counts.get("ip_address") == 1

    def test_localhost(self) -> None:
        result = filter_pii("Connect to 127.0.0.1")
        assert "[IP REDACTED]" in result.text

    def test_no_false_positive_on_version_numbers(self) -> None:
        # Version numbers like "3.11.5" should NOT match (only 3 octets)
        result = filter_pii("Python 3.11.5 is installed")
        assert result.redaction_counts.get("ip_address", 0) == 0


class TestCombinedRedaction:
    def test_multiple_pii_types_in_one_text(self) -> None:
        text = (
            "Employee John (SSN: 123-45-6789) can be reached at "
            "john@company.com or 555-867-5309. Payment card: 4111111111111111."
        )
        result = filter_pii(text)
        assert result.total_redactions >= 4
        assert "[SSN REDACTED]" in result.text
        assert "[EMAIL REDACTED]" in result.text
        assert "[PHONE REDACTED]" in result.text
        assert "[CARD REDACTED]" in result.text

    def test_no_pii_returns_unchanged_text(self) -> None:
        text = "This is a clean research abstract about quantum computing."
        result = filter_pii(text)
        assert result.text == text
        assert result.total_redactions == 0
        assert result.redaction_counts == {}


class TestFilterPiiSimple:
    def test_returns_string_only(self) -> None:
        result = filter_pii_simple("Email: test@example.com")
        assert isinstance(result, str)
        assert "[EMAIL REDACTED]" in result

    def test_clean_text_unchanged(self) -> None:
        text = "No PII here."
        assert filter_pii_simple(text) == text


class TestRedactionResult:
    def test_dataclass_fields(self) -> None:
        r = RedactionResult(text="clean", total_redactions=0, redaction_counts={})
        assert r.text == "clean"
        assert r.total_redactions == 0
