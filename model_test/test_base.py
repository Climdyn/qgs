
import unittest


class TestBase(unittest.TestCase):

    def load_ref_from_file(self):
        self.reference.clear()
        f = open(self.filename, 'r')
        buf = f.readlines()

        for l in buf:
            self.reference.append(l[:-1])

        f.close()

    def save_ip(self, s):
        self.values.append(s)

    def check_lists(self):
        self.outputs()
        self.load_ref_from_file()
        self.assertEqual(list(reversed(sorted(self.values))), list(reversed(sorted(self.reference))))

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
