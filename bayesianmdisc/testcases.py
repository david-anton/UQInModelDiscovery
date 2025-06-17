from typing import TypeAlias

import torch

from bayesianmdisc.customtypes import Tensor

TestCase: TypeAlias = Tensor
TestCases: TypeAlias = Tensor
AllowedTestCases: TypeAlias = Tensor
TestCaseIdentifier: TypeAlias = int
TestCaseIdentifiers: TypeAlias = list[int]
TestCaseLabel: TypeAlias = str
TestCaseLabels = list[TestCaseLabel]

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


identifiers = [
    test_case_identifier_uniaxial_tension,
    test_case_identifier_equibiaxial_tension,
    test_case_identifier_biaxial_tension,
    test_case_identifier_pure_shear,
    test_case_identifier_simple_shear_12,
    test_case_identifier_simple_shear_21,
    test_case_identifier_simple_shear_13,
    test_case_identifier_simple_shear_31,
    test_case_identifier_simple_shear_23,
    test_case_identifier_simple_shear_32,
]
labels = [
    test_case_label_uniaxial_tension,
    test_case_label_equibiaxial_tension,
    test_case_label_biaxial_tension,
    test_case_label_pure_shear,
    test_case_label_simple_shear_12,
    test_case_label_simple_shear_21,
    test_case_label_simple_shear_13,
    test_case_label_simple_shear_31,
    test_case_label_simple_shear_23,
    test_case_label_simple_shear_32,
]

identifiers_to_labels_map = dict(zip(identifiers, labels))
labels_to_identifiers_map = dict(zip(labels, identifiers))


def map_test_case_identifiers_to_labels(identifiers: TestCases) -> TestCaseLabels:
    labels: TestCaseLabels = []

    for identifier in identifiers.tolist():
        labels += [identifiers_to_labels_map[identifier]]

    return labels


def map_test_case_labels_to_identifiers(labels: TestCaseLabels) -> TestCases:
    identifiers: list[TestCaseIdentifier] = []

    for label in labels:
        identifiers += [labels_to_identifiers_map[label]]

    return torch.tensor(identifiers)
