from abc import ABC, abstractmethod
from core.schema import ActionProposal, VerificationResult, DatasetProfile

class BaseVerifier(ABC):
    """
    Abstract base class for verifiers.
    """
    @abstractmethod
    def verify(self, action: ActionProposal, profile: DatasetProfile) -> VerificationResult:
        pass