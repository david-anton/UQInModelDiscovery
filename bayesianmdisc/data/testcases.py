from typing import TypeAlias

from bayesianmdisc.customtypes import Tensor

TestCase: TypeAlias = Tensor
TestCases: TypeAlias = Tensor
AllowedTestCases = Tensor

test_case_identifier_uniaxial_tension = 0
test_case_identifier_equibiaxial_tension = 1
test_case_identifier_biaxial_tension = 2
test_case_identifier_pure_shear = 3
