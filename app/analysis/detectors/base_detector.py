from abc import ABC, abstractmethod

class BaseDetector(ABC):
    """
    Base detector for detecting landmarks of different movements (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each subclass must implement the abstract method of returning a detector object to detect landmarks
    """

    # ------------------------------------------------------------------
    # --- START: Abstract properties to be implemented by subclasses ---
    # ------------------------------------------------------------------
    @abstractmethod
    def get_detector() -> object:
        """
        Must return a detector object that can detect landmarks from a video.
        """
        pass