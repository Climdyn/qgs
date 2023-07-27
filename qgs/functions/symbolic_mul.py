import sympy as sy


def _symbolic_tensordot(a, b, axes=2):
    """
    Compute tensor dot product along specified axes of two sympy symbolic arrays

    This is based on numpy.tensordot

    Parameters
    ----------
    a, b: sympy arrays
        Tensors to "dot"

    axes: int
        * integer_like
          If an int N, sum over the last N axes of `a` and the first N axes
          of `b` in order. The sizes of the corresponding axes must match.

    Returns
    -------
    output: sympy tensor
        The tensor dot product of the input

    """
    as_ = a.shape
    nda = len(as_)
    
    a_com = [nda+i for i in range(-axes, 0)]
    b_com = [nda+i for i in range(axes)]
    sum_cols = tuple(a_com + b_com)
    
    prod = sy.tensorproduct(a, b)
    
    return sy.tensorcontraction(prod, sum_cols)

def symbolic_sparse_mult2(dict, vec_a):
    #//TODO: Complete documentation
    res = sy.zeros((len(vec_a), len(vec_a)))

    for key in dict.keys():
        coo1, coo2, coo3 = key
        res[coo1, coo2] += vec_a[coo3] * dict[key]

    return res

def symbolic_sparse_mult3(dict, vec_a, vec_b):
    #//TODO: Complete documentation
    res = sy.zeros(len(vec_a))

    for key in dict.keys():
        coo1, coo2, coo3 = key
        res[coo1] += vec_a[coo2] * vec_b[coo3] * dict[key]

    return res

def symbolic_sparse_mult4(dict, vec_a, vec_b, vec_c):
    #//TODO: Complete documentation
    res = sy.zeros((len(vec_a), len(vec_a)))

    for key in dict.keys():
        coo1, coo2, coo3, coo4, coo5 = key
        res[coo1, coo2] += vec_a[coo3] * vec_b[coo4] * vec_c[coo5] * dict[key]

    return res

def symbolic_sparse_mult5(dict, vec_a, vec_b, vec_c, vec_d):
    #//TODO: Complete documentation
    res = sy.zeros(len(vec_a))

    for key in dict.keys():
        coo1, coo2, coo3, coo4, coo5 = key
        res[coo1] += vec_a[coo2] * vec_b[coo3] * vec_c[coo4] * vec_d[coo5] * dict[key]

    return res