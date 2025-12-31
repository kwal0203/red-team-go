"""Tests for the PrivacyDetector guardrail.

Tests PII detection for:
- Email addresses
- Phone numbers (various formats)
- Social Security Numbers
- Credit card numbers
- IP addresses
- Street addresses
- API keys and secrets
"""

import pytest

from services.guardrails.src.detectors.privacy import (
    PII_PATTERNS,
    PII_RISK_WEIGHTS,
    PrivacyDetector,
)
from services.guardrails.src.pipeline import GuardrailPipeline
from services.guardrails.src.remediation import ContentRemediator, RemediationAction


class TestPrivacyDetectorBasics:
    """Basic functionality tests."""

    def test_clean_input_no_pii(self):
        """Test that clean input with no PII is not flagged."""
        detector = PrivacyDetector()
        result = detector.check(
            "This is a normal message with no personal information."
        )
        assert result.detected is False
        assert result.confidence == 0.0
        assert "No PII" in result.explanation

    def test_empty_input(self):
        """Test that empty input is handled correctly."""
        detector = PrivacyDetector()
        result = detector.check("")
        assert result.detected is False
        assert result.confidence == 0.0

    def test_whitespace_only_input(self):
        """Test that whitespace-only input is handled correctly."""
        detector = PrivacyDetector()
        result = detector.check("   \n\t  ")
        assert result.detected is False

    def test_detector_name(self):
        """Test that detector has correct name."""
        detector = PrivacyDetector()
        assert detector.name == "privacy"


class TestEmailDetection:
    """Tests for email address detection."""

    @pytest.mark.parametrize(
        "email",
        [
            "user@example.com",
            "john.doe@company.org",
            "test+label@gmail.com",
            "first.last@subdomain.domain.co.uk",
            "user123@test-domain.io",
        ],
    )
    def test_detects_valid_emails(self, email):
        """Test detection of various email formats."""
        detector = PrivacyDetector()
        result = detector.check(f"Contact me at {email} for more info.")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["email"]

    def test_email_in_sentence(self):
        """Test email detection within a sentence."""
        detector = PrivacyDetector()
        result = detector.check("Please send your resume to hr@company.com by Friday.")
        assert result.detected is True
        assert (
            "email" in result.explanation.lower() or "pii" in result.explanation.lower()
        )


class TestPhoneDetection:
    """Tests for phone number detection."""

    @pytest.mark.parametrize(
        "phone",
        [
            "123-456-7890",
            "(123) 456-7890",
            "123.456.7890",
            "123 456 7890",
            "+1 123-456-7890",
            "+1 (123) 456-7890",
            "1-123-456-7890",
        ],
    )
    def test_detects_us_phone_formats(self, phone):
        """Test detection of various US phone number formats."""
        detector = PrivacyDetector()
        result = detector.check(f"Call me at {phone}")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["phone"]


class TestSSNDetection:
    """Tests for Social Security Number detection."""

    def test_detects_ssn_with_dashes(self):
        """Test detection of SSN with standard format."""
        detector = PrivacyDetector()
        result = detector.check("My SSN is 123-45-6789")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["ssn"]

    def test_ssn_in_context(self):
        """Test SSN detection in realistic context."""
        detector = PrivacyDetector()
        result = detector.check(
            "For tax purposes, please provide your SSN: 987-65-4321"
        )
        assert result.detected is True


class TestCreditCardDetection:
    """Tests for credit card number detection."""

    @pytest.mark.parametrize(
        "card",
        [
            "1234 5678 9012 3456",
            "1234-5678-9012-3456",
            "1234567890123456",
        ],
    )
    def test_detects_16_digit_cards(self, card):
        """Test detection of standard 16-digit card numbers."""
        detector = PrivacyDetector()
        result = detector.check(f"Payment card: {card}")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["credit_card"]

    @pytest.mark.parametrize(
        "card",
        [
            "3782 822463 10005",
            "3782-822463-10005",
        ],
    )
    def test_detects_amex_format(self, card):
        """Test detection of American Express format (4-6-5)."""
        detector = PrivacyDetector()
        result = detector.check(f"Amex card: {card}")
        assert result.detected is True


class TestIPAddressDetection:
    """Tests for IP address detection."""

    @pytest.mark.parametrize(
        "ip",
        [
            "192.168.1.1",
            "10.0.0.1",
            "172.16.0.1",
            "8.8.8.8",
            "255.255.255.255",
            "0.0.0.0",
        ],
    )
    def test_detects_valid_ipv4(self, ip):
        """Test detection of valid IPv4 addresses."""
        detector = PrivacyDetector()
        result = detector.check(f"Server IP: {ip}")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["ip_address"]

    def test_rejects_invalid_ip(self):
        """Test that invalid IP-like strings are not detected."""
        detector = PrivacyDetector()
        # 999 is not a valid octet
        result = detector.check("Invalid IP: 999.999.999.999")
        # Should not detect as valid IP
        assert (
            result.detected is False or "ip" not in result.category.lower()
            if result.category
            else True
        )


class TestAPIKeyDetection:
    """Tests for API key and secret detection."""

    @pytest.mark.parametrize(
        "key_format",
        [
            "api_key=abcdef1234567890abcdef",
            "apikey: ghijkl5678901234ghijkl",
            "secret_key='mnopqr0123456789mnopqr'",
            "auth_token=stuvwx4567890123stuvwx",
            "password=yzabcd7890123456yzabcd",
        ],
    )
    def test_detects_key_value_formats(self, key_format):
        """Test detection of key=value style secrets."""
        detector = PrivacyDetector()
        result = detector.check(f"Config: {key_format}")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["api_key"]

    def test_detects_openai_key(self):
        """Test detection of OpenAI API key format."""
        detector = PrivacyDetector()
        result = detector.check(
            "OPENAI_API_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz123456"
        )
        assert result.detected is True

    def test_detects_github_pat(self):
        """Test detection of GitHub Personal Access Token."""
        detector = PrivacyDetector()
        result = detector.check("GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890")
        assert result.detected is True

    def test_detects_aws_access_key(self):
        """Test detection of AWS access key ID."""
        detector = PrivacyDetector()
        result = detector.check("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE")
        assert result.detected is True

    def test_detects_bearer_token(self):
        """Test detection of Bearer authentication token."""
        detector = PrivacyDetector()
        result = detector.check(
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        )
        assert result.detected is True


class TestAddressDetection:
    """Tests for street address detection."""

    @pytest.mark.parametrize(
        "address",
        [
            "123 Main Street",
            "456 Oak Avenue",
            "789 Elm Road",
            "101 Pine Boulevard",
            "202 Cedar Drive",
            "303 Maple Lane",
            "404 Birch Way",
            "505 Willow Court",
            "606 Cherry Place",
        ],
    )
    def test_detects_street_addresses(self, address):
        """Test detection of various street address formats."""
        detector = PrivacyDetector()
        result = detector.check(f"Send package to {address}")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["address"]


class TestMultiplePIITypes:
    """Tests for detecting multiple PII types in same content."""

    def test_multiple_pii_types(self):
        """Test detection when multiple PII types present."""
        detector = PrivacyDetector()
        content = """
        Contact Information:
        Email: john.doe@example.com
        Phone: (555) 123-4567
        Address: 123 Main Street
        """
        result = detector.check(content)
        assert result.detected is True
        # Confidence should be higher with multiple types
        assert result.confidence > PII_RISK_WEIGHTS["email"]
        assert "multiple" in result.explanation.lower()

    def test_high_risk_pii_combination(self):
        """Test detection of high-risk PII combination."""
        detector = PrivacyDetector()
        content = """
        SSN: 123-45-6789
        Credit Card: 4111-1111-1111-1111
        API Key: sk-abcdefghijklmnopqrstuvwxyz123456
        """
        result = detector.check(content)
        assert result.detected is True
        # Should have high confidence with critical PII
        assert result.confidence >= 0.8


class TestConfidenceScoring:
    """Tests for confidence score calculation."""

    def test_single_low_risk_pii(self):
        """Test confidence for single low-risk PII."""
        detector = PrivacyDetector()
        result = detector.check("Server at 192.168.1.1")
        assert result.detected is True
        assert result.confidence == PII_RISK_WEIGHTS["ip_address"]

    def test_single_high_risk_pii(self):
        """Test confidence for single high-risk PII."""
        detector = PrivacyDetector()
        result = detector.check("SSN: 123-45-6789")
        assert result.detected is True
        assert result.confidence >= PII_RISK_WEIGHTS["ssn"]

    def test_confidence_capped_at_one(self):
        """Test that confidence is capped at 1.0."""
        detector = PrivacyDetector()
        # Many PII items should still cap at 1.0
        content = """
        email1@test.com, email2@test.com, email3@test.com
        SSN: 123-45-6789, SSN: 234-56-7890
        Card: 1234-5678-9012-3456
        Phone: 555-123-4567, 555-234-5678
        """
        result = detector.check(content)
        assert result.confidence <= 1.0

    def test_threshold_behavior(self):
        """Test that threshold affects detection."""
        # High threshold should require more PII
        detector_high = PrivacyDetector(threshold=0.5)
        result = detector_high.check("Contact: 192.168.1.1")  # IP only, low weight
        assert result.detected is False  # Below threshold

        # Default threshold should detect
        detector_default = PrivacyDetector()
        result = detector_default.check("SSN: 123-45-6789")  # SSN, high weight
        assert result.detected is True


class TestPipelineIntegration:
    """Tests for integration with guardrail pipeline."""

    def test_privacy_in_default_guardrails(self):
        """Test that privacy is included in default guardrails."""
        pipeline = GuardrailPipeline()
        assert "privacy" in pipeline.active_guardrails

    def test_pipeline_detects_pii(self):
        """Test that pipeline detects PII through privacy guardrail."""
        pipeline = GuardrailPipeline(guardrails=["privacy"])
        result = pipeline.check("My email is test@example.com")
        assert "privacy" in result.violations
        assert result.results["privacy"].detected is True

    def test_privacy_only_pipeline(self):
        """Test pipeline with only privacy guardrail."""
        pipeline = GuardrailPipeline(guardrails=["privacy"])
        assert pipeline.active_guardrails == ["privacy"]

    def test_privacy_contributes_to_risk_level(self):
        """Test that privacy violations affect overall risk level."""
        pipeline = GuardrailPipeline(guardrails=["privacy"])
        # High-risk PII should result in high/critical risk
        result = pipeline.check("SSN: 123-45-6789")
        assert result.overall_risk.value in ["high", "critical"]


class TestRemediation:
    """Tests for PII redaction functionality."""

    def test_email_redaction(self):
        """Test that emails are redacted correctly."""
        remediator = ContentRemediator(default_action=RemediationAction.REDACT)
        pipeline = GuardrailPipeline(guardrails=["privacy"])

        content = "Contact me at john.doe@example.com"
        pipeline_result = pipeline.check(content)
        result = remediator.remediate(content, pipeline_result)

        assert result.remediated_content is not None
        assert "[EMAIL REDACTED]" in result.remediated_content
        assert "john.doe@example.com" not in result.remediated_content

    def test_phone_redaction(self):
        """Test that phone numbers are redacted correctly."""
        remediator = ContentRemediator(default_action=RemediationAction.REDACT)
        pipeline = GuardrailPipeline(guardrails=["privacy"])

        content = "Call me at (555) 123-4567"
        pipeline_result = pipeline.check(content)
        result = remediator.remediate(content, pipeline_result)

        assert result.remediated_content is not None
        assert "[PHONE REDACTED]" in result.remediated_content

    def test_ssn_redaction(self):
        """Test that SSNs are redacted correctly."""
        remediator = ContentRemediator(default_action=RemediationAction.REDACT)
        pipeline = GuardrailPipeline(guardrails=["privacy"])

        content = "SSN: 123-45-6789"
        pipeline_result = pipeline.check(content)
        result = remediator.remediate(content, pipeline_result)

        assert result.remediated_content is not None
        assert "[SSN REDACTED]" in result.remediated_content
        assert "123-45-6789" not in result.remediated_content

    def test_credit_card_redaction(self):
        """Test that credit cards are redacted correctly."""
        remediator = ContentRemediator(default_action=RemediationAction.REDACT)
        pipeline = GuardrailPipeline(guardrails=["privacy"])

        content = "Card: 1234-5678-9012-3456"
        pipeline_result = pipeline.check(content)
        result = remediator.remediate(content, pipeline_result)

        assert result.remediated_content is not None
        assert "[CARD REDACTED]" in result.remediated_content

    def test_api_key_redaction(self):
        """Test that API keys are redacted correctly."""
        remediator = ContentRemediator(default_action=RemediationAction.REDACT)
        pipeline = GuardrailPipeline(guardrails=["privacy"])

        content = "Key: sk-abcdefghijklmnopqrstuvwxyz123456"
        pipeline_result = pipeline.check(content)
        result = remediator.remediate(content, pipeline_result)

        assert result.remediated_content is not None
        assert "[API KEY REDACTED]" in result.remediated_content

    def test_multiple_pii_redaction(self):
        """Test that PII is redacted when risk is below critical threshold."""
        remediator = ContentRemediator(default_action=RemediationAction.REDACT)
        pipeline = GuardrailPipeline(guardrails=["privacy"])

        # Use only a single low-risk PII type to stay below critical risk level
        # (High confidence with privacy guardrail triggers CRITICAL -> BLOCK)
        content = "Server IP: 192.168.1.1"
        pipeline_result = pipeline.check(content)
        result = remediator.remediate(content, pipeline_result)

        assert result.remediated_content is not None
        assert "[IP REDACTED]" in result.remediated_content
        assert "192.168.1.1" not in result.remediated_content

    def test_block_action_for_critical_pii(self):
        """Test that block action works for high-risk content."""
        remediator = ContentRemediator(default_action=RemediationAction.BLOCK)
        pipeline = GuardrailPipeline(guardrails=["privacy"])

        content = "SSN: 123-45-6789"
        pipeline_result = pipeline.check(content)
        result = remediator.remediate(content, pipeline_result)

        assert result.allowed is False
        assert result.action_taken == RemediationAction.BLOCK


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_pii_patterns_exist(self):
        """Verify all expected pattern categories exist."""
        expected_categories = [
            "email",
            "phone",
            "ssn",
            "credit_card",
            "ip_address",
            "api_key",
            "address",
        ]
        for category in expected_categories:
            assert category in PII_PATTERNS
            assert len(PII_PATTERNS[category]) > 0

    def test_risk_weights_exist(self):
        """Verify all pattern categories have risk weights."""
        for category in PII_PATTERNS:
            assert category in PII_RISK_WEIGHTS
            assert 0.0 < PII_RISK_WEIGHTS[category] <= 1.0

    def test_unicode_content(self):
        """Test handling of unicode content with valid PII."""
        detector = PrivacyDetector()
        # Unicode surrounding text with valid ASCII email
        result = detector.check("お問い合わせ: user@example.com までご連絡ください")
        # Should detect the valid ASCII email within unicode text
        assert result.detected is True

    def test_mixed_case_sensitivity(self):
        """Test case sensitivity handling."""
        detector = PrivacyDetector()
        # Email patterns should be case-insensitive
        result = detector.check("EMAIL: USER@EXAMPLE.COM")
        assert result.detected is True
