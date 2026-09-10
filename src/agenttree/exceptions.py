"""Framework errors shared by provider-powered decision strategies."""


class DecisionOutputError(ValueError):
    """Provider JSON does not satisfy a framework decision contract."""


class DecisionParseError(DecisionOutputError):
    """Provider output is not an unambiguous JSON object."""
