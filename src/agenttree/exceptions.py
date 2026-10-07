"""Framework errors shared by provider-powered decision strategies."""


class DecisionOutputError(ValueError):
    """Provider JSON does not satisfy a framework decision contract."""

    def __init__(self, message: str, *, failure_class: str = "schema_invalid",
                 reason_code: str = "schema_mismatch", field_name: str | None = None) -> None:
        super().__init__(message)
        if failure_class not in {"schema_invalid", "semantic_invalid", "invalid_reference"}:
            raise ValueError("Unsupported structured decision failure class")
        self.failure_class = failure_class
        self.reason_code = reason_code
        self.field_name = field_name


class DecisionParseError(DecisionOutputError):
    """Provider output is not an unambiguous JSON object."""
