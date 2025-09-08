from abc import ABC, abstractmethod

class BaseSignalAnalyzer(ABC):
    """
    Base signal analyzer for all signals (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each  subclass must implement these abstract methods for analyzing a signal
    """


    # ------------------------------------------------------------------
    # --- START: Abstract properties to be implemented by subclasses ---
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def analyze(self, raw_signal, normalization_factor, start_time, end_time) -> dict:
        """
        Should return a dictionary of features. Some recommended features would be 
            - lineplot
            - velocity plot
            - rawData
            - peaks
            - valleys
            - valleys_start
            - valleys_end
            - radarTable (dictionary)
                -MeanAmplitude
                - StdAmplitude  
                - MeanSpeed  
                - StdSpeed  
                - MeanRMSVelocity  
                - StdRMSVelocity  
                - MeanOpeningSpeed  
                - stdOpeningSpeed  
                - meanClosingSpeed  
                - stdClosingSpeed  
                - meanCycleDuration  
                - stdCycleDuration  
                - rangeCycleDuration  
                - rate  
                - amplitudeDecay  
                - velocityDecay  
                - rateDecay  
                - cvAmplitude  
                - cvCycleDuration  
                - cvSpeed  
                - cvRMSVelocity  
                - cvOpeningSpeed  
                - cvClosingSpeed
        """
        pass