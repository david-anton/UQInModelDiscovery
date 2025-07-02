from pathlib import Path


class Error(Exception):
    pass


class FileNotFoundError(Error):
    def __init__(self, path_to_file: Path) -> None:
        self._message = f"The requested file {path_to_file} could not be found!"
        super().__init__(self._message)


class DirectoryNotFoundError(Error):
    def __init__(self, path_to_directory: Path) -> None:
        self._message = f"The directory {path_to_directory} could not be found"
        super().__init__(self._message)


class ProbabilityDistributionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class MainError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DataError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GPError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ParameterExtractionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class CombinedPriorError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ModelError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class OutputSelectorError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class LikelihoodError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PlotterError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ModelSelectionError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class StatisticsError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)
