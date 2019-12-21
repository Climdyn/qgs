
import numpy as np


class CooTensor(object):

    def __init__(self, ncoo=3, dtype=None):

        self.coo = np.zeros((1, ncoo), dtype=np.int32)

        if dtype is None:
            dtype=np.float64
        self.value = np.zeros(1, dtype=dtype)

    def add_entries(self, coo, value):

        self.coo = np.vstack((self.coo, coo))
        self.value = np.concatenate((self.value, value))

    def delete_entries(self, indices):
        self.coo = np.delete(self.coo, indices, axis=0)
        self.value = np.delete(self.value, indices, axis=0)

    def def_entries(self, coo_list, value_list):

        self.coo = np.array(coo_list)
        self.value = np.array(value_list)

    def __getitem__(self, index):

        return self.coo[index], self.value[index]

    def __len__(self):
        return len(self.value)


def from_coo_mat_list(coo_mat_list):

    out = CooTensor()

    for i, mat in enumerate(coo_mat_list):
        row = mat.row
        col = mat.col
        l = len(row)

        value = mat.data

        coo = np.hstack((np.full((l, 1), i, dtype=np.int32), row[..., np.newaxis], col[..., np.newaxis]))

        out.add_entries(coo, value)

    out.delete_entries(0)

    return out


def from_csr_mat_list(csr_mat_list):
    coo_mat_list = list()

    for mat in csr_mat_list:
        coo_mat_list.append(mat.tocoo())

    return from_coo_mat_list(coo_mat_list)
