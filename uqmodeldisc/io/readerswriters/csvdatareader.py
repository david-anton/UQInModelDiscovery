from typing import Any, Optional

import numpy as np
import pandas as pd

from uqmodeldisc.customtypes import NPArray, PDDataFrame
from uqmodeldisc.io import ProjectDirectory
from uqmodeldisc.io.readerswriters.utility import (
    ensure_correct_file_ending,
    join_input_file_path,
)


class CSVDataReader:
    def __init__(self, project_directory: ProjectDirectory) -> None:
        self._project_directory = project_directory
        self._correct_file_ending = ".csv"

    def read_as_numpy_array(
        self,
        file_name: str,
        subdir_name: Optional[str] = None,
        header: Any = 0,
        seperator=",",
        read_from_output_dir: bool = False,
    ) -> NPArray:
        file_name = ensure_correct_file_ending(
            file_name=file_name, file_ending=self._correct_file_ending
        )
        input_file_path = join_input_file_path(
            file_name=file_name,
            subdir_name=subdir_name,
            project_directory=self._project_directory,
            read_from_output_dir=read_from_output_dir,
        )
        return pd.read_csv(input_file_path, header=header, sep=seperator).to_numpy(
            dtype=np.float64
        )

    def read_as_pandas_data_frame(
        self,
        file_name: str,
        subdir_name: Optional[str] = None,
        header: Any = 0,
        seperator=",",
        read_from_output_dir: bool = False,
    ) -> PDDataFrame:
        file_name = ensure_correct_file_ending(
            file_name=file_name, file_ending=self._correct_file_ending
        )
        input_file_path = join_input_file_path(
            file_name=file_name,
            subdir_name=subdir_name,
            project_directory=self._project_directory,
            read_from_output_dir=read_from_output_dir,
        )
        return pd.read_csv(input_file_path, header=header, sep=seperator)
