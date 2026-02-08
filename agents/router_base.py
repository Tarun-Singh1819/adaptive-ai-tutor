from abc import ABC, abstractmethod
from typing import Dict

class BaseRouter(ABC):
    """
    Base class for all routers.
    A router only decides which agent should handle the query.
    """

    @abstractmethod
    def route(self, state: Dict) -> str:
        """
        Given the current state, return one of:
        'overview', 'doubt', or 'research'
        """
        pass
