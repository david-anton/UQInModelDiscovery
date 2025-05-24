from typing import TypeAlias

from bayesianmdisc.customtypes import Tensor

TestCase: TypeAlias = Tensor
TestCases: TypeAlias = Tensor
AllowedTestCases = Tensor

test_case_identifier_uniaxial_tension = 0
test_case_identifier_equibiaxial_tension = 1
test_case_identifier_biaxial_tension = 2
test_case_identifier_pure_shear = 3
test_case_identifier_simple_shear_12 = 4
test_case_identifier_simple_shear_21 = 5
test_case_identifier_simple_shear_13 = 6
test_case_identifier_simple_shear_31 = 7
test_case_identifier_simple_shear_23 = 8
test_case_identifier_simple_shear_32 = 9

test_case_label_uniaxial_tension = "UT"
test_case_label_equibiaxial_tension = "EBT"
test_case_label_biaxial_tension = "BT"
test_case_label_pure_shear = "PS"
test_case_label_simple_shear_12 = "SS12"
test_case_label_simple_shear_21 = "SS21"
test_case_label_simple_shear_13 = "SS13"
test_case_label_simple_shear_31 = "SS31"
test_case_label_simple_shear_23 = "SS23"
test_case_label_simple_shear_32 = "SS32"
