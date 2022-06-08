import unittest


def factorize(x):
    """ Factorize positive integer and return its factors.
        :type x: int,>=0
        :rtype: tuple[N],N>0
    """
    pass


class TestFactorize(unittest.TestCase):
    def test_wrong_types_raise_exception(self):
        cases = ('string', 1.5)
        for i, obj in enumerate(cases):
            with self.subTest(case=i):
                self.assertRaises(TypeError, factorize, obj)

    def test_negative(self):
        cases = (-1, -10, -100)
        for i, obj in enumerate(cases):
            with self.subTest(case=i):
                self.assertRaises(ValueError, factorize, obj)

    def test_zero_and_one_cases(self):
        cases = (0, 1)
        cases_result = ((0,), (1,))
        for i, obj in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(factorize(obj), cases_result[i])

    def test_simple_numbers(self):
        cases = (3, 13, 29)
        cases_result = ((3,), (13,), (29,))
        for i, obj in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(factorize(obj), cases_result[i])

    def test_two_simple_multipliers(self):
        cases = (6, 26, 121)
        cases_result = ((2, 3), (2, 13), (11, 11))
        for i, obj in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(factorize(obj), cases_result[i])

    def test_many_multipliers(self):
        cases = (1001, 9699690)
        cases_result = ((7, 11, 13), (2, 3, 5, 7, 11, 13, 17, 19))
        for i, obj in enumerate(cases):
            with self.subTest(case=i):
                self.assertEqual(factorize(obj), cases_result[i])
