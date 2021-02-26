
import os

path = os.path.abspath('./')
base = os.path.basename(path)
if base == 'model_test':
    fold = ""
else:
    fold = 'model_test/'

import unittest
import numpy as np

real_eps = np.finfo(np.float64).eps


class TestBase(unittest.TestCase):

    reference = list()
    values = list()
    folder = fold

    def load_ref_from_file(self):
        self.reference.clear()
        f = open(self.folder+self.filename, 'r')
        buf = f.readlines()

        for l in buf:
            self.reference.append(l[:-1])

        f.close()

    def save_ip(self, s):
        self.values.append(s)

    def check_lists_flt(self):
        self.outputs()
        self.load_ref_from_file()
        for v, r in zip(list(reversed(sorted(self.values))), list(reversed(sorted(self.reference)))):
            self.assertTrue(self.match_flt(v, r), msg=v+' != '+r+' !!!')

    def check_lists(self, cmax=1):
        self.outputs()
        self.load_ref_from_file()
        for v, r in zip(list(reversed(sorted(self.values))), list(reversed(sorted(self.reference)))):
            self.assertTrue(self.match_str(v, r, cmax), msg=v+' != '+r+' !!!')

    def write_reference_to_file(self, filename):
        f = open(filename, 'w')

        for l in reversed(sorted(self.reference)):
            f.write(l+'\n')

        f.close()

    def write_values_to_file(self, filename):
        self.outputs()
        f = open(filename, 'w')

        for l in reversed(sorted(self.values)):
            f.write(l + '\n')

        f.close()

    def outputs(self):
        pass

    @staticmethod
    def match_flt(s1, s2, eps=real_eps):

        s1p = s1.split('=')
        s2p = s2.split('=')

        v1 = float(s1p[1])
        v2 = float(s2p[1])

        return abs(v1 - v2) < eps

    @staticmethod
    def match_str(s1, s2, cmax=1):

        c = 0

        for c1, c2 in zip(s1, s2):
            if c1 != c2:
                c += 1

            if c > cmax:
                return False

        return True
