from abc import *


class SignalBase(metaclass=ABCMeta):
    def __init__(self, input_sig):
        self.input_sig = input_sig

    @abstractmethod
    def get_cycle_len(self):
        pass

    @abstractmethod
    def get_cycle(self):
        pass

    @abstractmethod
    def get_systolic(self):
        pass

    @abstractmethod
    def get_diastolic(self):
        pass

    @abstractmethod
    def flat_detection(self):
        pass

    # @abstractmethod
    # def flip_detection(self):
    #     pass

    @abstractmethod
    def return_sig_status(self):
        pass

    @abstractmethod
    def signal_validation(self):
        pass

    @abstractmethod
    def plot(self):
        pass

    # @abstractmethod
    # def amp_chekcker(self):
    #     pass
    #
    # @abstractmethod
    # def pulse_pressure_checker(self):
    #     pass
    #
    # @abstractmethod
    # def under_damped_detection(self):
    #     pass
    #
    # @abstractmethod
    # def over_damped_detection(self):
    #     pass
