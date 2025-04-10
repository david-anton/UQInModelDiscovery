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


class DataError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class DataSetError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GPError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class GPPriorError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ModelError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class LikelihoodError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class StressPlotterError(Error):
    def __init__(self, message: str) -> None:
        super().__init__(message)
