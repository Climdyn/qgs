"""
    qgs atmospheric thermodynamic tensor module
    ===========================================

    This module computes and holds the tensors representing the atmospheric thermodynamic tendencies of the model's equations
    needed to compute the vertical wind velocity :math:`\\omega`.

    TODO: Add a list of the different tensor available

"""

import numpy as np
import sparse as sp

from qgs.tensors.qgtensor import QgsTensor

real_eps = np.finfo(np.float64).eps


class AtmoThermoTensor(QgsTensor):
    """Atmospheric thermodynamic tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        QgsTensor.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)

    def _compute_tensor_dicts(self):

        if self.params is None:
            return None

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return None

        aips = self.atmospheric_inner_products
        par = self.params
        atp = par.atemperature_params
        nvar = par.number_of_variables
        ndim = par.ndim

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        if self.params.dynamic_T:
            offset = 1
        else:
            offset = 0

        # constructing some derived matrices
        if aips is not None:
            a_theta = np.zeros((nvar[1], nvar[1]))
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[i, j] = aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

        #################

        if bips is not None:
            go = bips.stored
        else:
            go = True

        sparse_arrays_dict = dict()

        if aips.stored and go:

            # theta_a part
            for i in range(nvar[1]):
                t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                if par.Cpa is not None:
                    t[0, 0] += par.Cpa[i]

                if atp.hd is not None and atp.thetas is not None:
                    t[0, 0] += atp.thetas[i] * atp.hd

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists

                        val = a_theta[i, :] @ aips._g[:, jo, ko]
                        t[self._psi_a(j), self._theta_a(ko)] -= val

                for j in range(nvar[1]):
                    val = a_theta[i, :] @ aips._u[:, j]
                    if par.Lpa is not None:
                        t[self._theta_a(j), 0] -= val * atp.sc * par.Lpa
                    if par.LSBpa is not None:
                        t[self._theta_a(j), 0] -= val * par.LSBpa

                    if atp.hd is not None:
                        t[self._theta_a(j), 0] -= val * atp.hd

                if ocean:
                    for j in range(nvar[2]):
                        jo = j + offset  # skipping the theta 0 variable if it exists

                    if par.Lpa is not None:
                        for j in range(nvar[3]):
                            val = a_theta[i, :] @ aips._s[:, j]
                            t[self._deltaT_o(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_o(j), 0] += val * par.LSBpgo

                if ground_temp:
                    if par.Lpa is not None:
                        for j in range(nvar[2]):
                            val = a_theta[i, :] @ aips._s[:, j]
                            t[self._deltaT_g(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_g(j), 0] += val * par.LSBpgo

                sparse_arrays_dict[self._theta_a(i)] = t.to_coo()

        else:

            # theta_a part
            for i in range(nvar[1]):
                t = sp.zeros((ndim + 1, ndim + 1), dtype=np.float64, format='dok')

                if par.Cpa is not None:
                    t[0, 0] += par.Cpa[i]

                if atp.hd is not None and atp.thetas is not None:
                    t[0, 0] += atp.thetas[i] * atp.hd

                for j in range(nvar[0]):

                    jo = j + offset  # skipping the theta 0 variable if it exists

                    for k in range(nvar[0]):
                        ko = k + offset  # skipping the theta 0 variable if it exists

                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.g(jj, jo, ko)

                        t[self._psi_a(j), self._theta_a(ko)] -= val

                for j in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val += a_theta[i, jj] * aips.u(jj, j)

                    if par.Lpa is not None:
                        t[self._theta_a(j), 0] -= val * atp.sc * par.Lpa
                    if par.LSBpa is not None:
                        t[self._theta_a(j), 0] -= val * par.LSBpa

                    if atp.hd is not None:
                        t[self._theta_a(j), 0] -= val * atp.hd

                if ocean:
                    if par.Lpa is not None:
                        for j in range(nvar[3]):
                            val = 0
                            for jj in range(nvar[1]):
                                val += a_theta[i, jj] * aips.s(jj, j)
                            t[self._deltaT_o(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_o(j), 0] += val * par.LSBpgo

                if ground_temp:
                    if par.Lpa is not None:
                        for j in range(nvar[2]):
                            val = 0
                            for jj in range(nvar[1]):
                                val += a_theta[i, jj] * aips.s(jj, j)
                            t[self._deltaT_g(j), 0] += val * par.Lpa / 2
                            if par.LSBpgo is not None:
                                t[self._deltaT_g(j), 0] += val * par.LSBpgo

                sparse_arrays_dict[self._theta_a(i)] = t.to_coo()

        return sparse_arrays_dict


class AtmoThermoTensorDynamicT(AtmoThermoTensor):
    """Atmospheric thermodynamic dynamical temperature first order (linear) tendencies tensor class.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        AtmoThermoTensor.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)

    def _compute_tensor_dicts(self):

        if self.params is None:
            return None

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return None

        aips = self.atmospheric_inner_products

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products

        elif self.ground_inner_products is not None:
            bips = self.ground_inner_products

        if bips is not None:
            go = bips.stored
        else:
            go = True

        if aips.stored and go:

            sparse_arrays_full_dict = self._compute_stored_full_dict()

        else:

            sparse_arrays_full_dict = self._compute_non_stored_full_dict()

        return sparse_arrays_full_dict

    def _compute_stored_full_dict(self):
        par = self.params
        ap = par.atmospheric_params
        nvar = par.number_of_variables
        aips = self.atmospheric_inner_products

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        # constructing identity matrices
        if aips is not None:
            if aips is not None:
                a_theta = np.zeros((nvar[1], nvar[1]))
                for i in range(nvar[1]):
                    for j in range(nvar[1]):
                        a_theta[i, j] = aips.u(i, j)
                a_theta = np.linalg.inv(a_theta)
                a_theta = sp.COO(a_theta)

        #################

        sparse_arrays_full_dict = dict()
        # theta_a part
        for i in range(nvar[1]):
            sparse_arrays_full_dict[self._theta_a(i)] = list()

            if par.T4LSBpa is not None:
                val = sp.tensordot(a_theta[i], aips._z, axes=1)
                if val.nnz > 0:
                    sparse_arrays_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(- par.T4LSBpa * val, self._theta_a(0)))

            if ocean:
                if par.T4LSBpgo is not None:
                    val = sp.tensordot(a_theta[i], aips._v, axes=1)
                    if val.nnz > 0:
                        sparse_arrays_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(par.T4LSBpgo * val, self._deltaT_o(0)))

            if ground_temp:

                if par.T4LSBpgo is not None:
                    val = sp.tensordot(a_theta[i], aips._v, axes=1)
                    if val.nnz > 0:
                        sparse_arrays_full_dict[self._theta_a(i)].append(self._shift_tensor_coordinates(par.T4LSBpgo * val, self._deltaT_g(0)))

        return sparse_arrays_full_dict

    def _compute_non_stored_full_dict(self):
        par = self.params
        ap = par.atmospheric_params
        nvar = par.number_of_variables
        ndim = par.ndim
        aips = self.atmospheric_inner_products

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        # constructing some derived matrices
        if aips is not None:

            a_theta = np.zeros((nvar[1], nvar[1]))
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[i, j] = ap.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

        #################

        sparse_arrays_full_dict = dict()
        # theta_a part
        for i in range(nvar[1]):
            t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

            if par.T4LSBpa is not None:
                j = k = ell = 0
                for m in range(nvar[1]):
                    val = 0
                    for jj in range(nvar[1]):
                        val -= a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                    if m == 0:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4LSBpa * val
                    else:
                        t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = 4 * par.T4LSBpa * val

            if ocean:
                if par.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[3]):
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = par.T4LSBpgo * val
                        else:
                            t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = 4 * par.T4LSBpgo * val

            if ground_temp:
                if par.T4LSBpgo is not None:
                    j = k = ell = 0
                    for m in range(nvar[2]):
                        val = 0
                        for jj in range(nvar[1]):
                            val += a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                        if m == 0:
                            t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = par.T4LSBpgo * val
                        else:
                            t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = 4 * par.T4LSBpgo * val

            sparse_arrays_full_dict[self._theta_a(i)] = t_full.to_coo()

        return sparse_arrays_full_dict

    def compute_tensor(self):
        """Routine to compute the tensor."""
        # gathering
        par = self.params
        ndim = par.ndim

        sparse_arrays_dict = QgsTensor._compute_tensor_dicts(self)
        sparse_arrays_full_dict = self._compute_tensor_dicts()

        tensor = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='coo')
        if sparse_arrays_dict is not None:
            tensor = self._add_dict_to_tensor(sparse_arrays_dict, tensor)
        if sparse_arrays_full_dict is not None:
            tensor = self._add_dict_to_tensor(sparse_arrays_full_dict, tensor)
        self._set_tensor(tensor)


class AtmoThermoTensorT4(AtmoThermoTensorDynamicT):
    """Atmospheric thermodynamic :math:`T^4` tendencies tensor class. Implies dynamical zeroth-order temperature.

    Parameters
    ----------
    params: None or QgParams, optional
        The models parameters to configure the tensor. `None` to initialize an empty tensor. Default to `None`.
    atmospheric_inner_products: None or AtmosphericInnerProducts, optional
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts, optional
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts, optional
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.

    Attributes
    ----------
    params: None or QgParams
        The models parameters used to configure the tensor. `None` for an empty tensor.
    atmospheric_inner_products: None or AtmosphericInnerProducts
        The inner products of the atmospheric basis functions on which the model's PDE atmospheric equations are projected.
        If `None`, disable the atmospheric tendencies. Default to `None`.
    oceanic_inner_products: None or OceanicInnerProducts
        The inner products of the oceanic basis functions on which the model's PDE oceanic equations are projected.
        If `None`, disable the oceanic tendencies. Default to `None`.
    ground_inner_products: None or GroundInnerProducts
        The inner products of the ground basis functions on which the model's PDE ground equations are projected.
        If `None`, disable the ground tendencies. Default to `None`.
    tensor: sparse.COO(float)
        The tensor :math:`\\mathcal{T}_{i,j,k}` :math:`i`-th components.
    jacobian_tensor: sparse.COO(float)
        The jacobian tensor :math:`\\mathcal{T}_{i,j,k} + \\mathcal{T}_{i,k,j}` :math:`i`-th components.
    """

    def __init__(self, params=None, atmospheric_inner_products=None, oceanic_inner_products=None, ground_inner_products=None):

        AtmoThermoTensorDynamicT.__init__(self, params, atmospheric_inner_products, oceanic_inner_products, ground_inner_products)

    def _compute_non_stored_full_dict(self):

        if self.params is None:
            return None

        if self.atmospheric_inner_products is None and self.oceanic_inner_products is None \
                and self.ground_inner_products is None:
            return None

        aips = self.atmospheric_inner_products
        par = self.params
        ap = par.atmospheric_params
        nvar = par.number_of_variables
        ndim = par.ndim

        bips = None
        if self.oceanic_inner_products is not None:
            bips = self.oceanic_inner_products
            ocean = True
        else:
            ocean = False

        if self.ground_inner_products is not None:
            bips = self.ground_inner_products
            ground_temp = True
        else:
            ground_temp = False

        # constructing some derived matrices
        if aips is not None:

            a_theta = np.zeros((nvar[1], nvar[1]))
            for i in range(nvar[1]):
                for j in range(nvar[1]):
                    a_theta[i, j] = ap.sig0 * aips.a(i, j) - aips.u(i, j)
            a_theta = np.linalg.inv(a_theta)
            a_theta = sp.COO(a_theta)

        #################

        sparse_arrays_full_dict = dict()

        # theta_a part
        for i in range(nvar[1]):
            t_full = sp.zeros((ndim + 1, ndim + 1, ndim + 1, ndim + 1), dtype=np.float64, format='dok')

            if par.T4LSBpa is not None:
                for j in range(nvar[1]):
                    for k in range(nvar[1]):
                        for ell in range(nvar[1]):
                            for m in range(nvar[1]):
                                val = 0
                                for jj in range(nvar[1]):
                                    val -= a_theta[i, jj] * aips.z(jj, j, k, ell, m)
                                t_full[self._theta_a(j), self._theta_a(k), self._theta_a(ell), self._theta_a(m)] = par.T4LSBpa * val

            if ocean:
                if par.T4LSBpgo is not None:
                    for j in range(nvar[3]):
                        for k in range(nvar[3]):
                            for ell in range(nvar[3]):
                                for m in range(nvar[3]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                        val += a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                                    t_full[self._deltaT_o(j), self._deltaT_o(k), self._deltaT_o(ell), self._deltaT_o(m)] = par.T4LSBpgo * val

            if ground_temp:
                if par.T4LSBpgo is not None:
                    for j in range(nvar[2]):
                        for k in range(nvar[2]):
                            for ell in range(nvar[2]):
                                for m in range(nvar[2]):
                                    val = 0
                                    for jj in range(nvar[1]):
                                        val += a_theta[i, jj] * aips.v(jj, j, k, ell, m)
                                    t_full[self._deltaT_g(j), self._deltaT_g(k), self._deltaT_g(ell), self._deltaT_g(m)] = par.T4LSBpgo * val

            sparse_arrays_full_dict[self._theta_a(i)] = t_full.to_coo()

        return sparse_arrays_full_dict


def _kronecker_delta(i, j):

    if i == j:
        return 1

    else:
        return 0


if __name__ == '__main__':
    from qgs.params.params import QgParams
    from qgs.inner_products.analytic import AtmosphericAnalyticInnerProducts, OceanicAnalyticInnerProducts
    from qgs.inner_products.symbolic import AtmosphericSymbolicInnerProducts, OceanicSymbolicInnerProducts

    # Analytic test

    params = QgParams({'rr': 287.e0, 'sb': 5.6e-8})
    params.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})
    params.set_atmospheric_channel_fourier_modes(2, 2)
    params.set_oceanic_basin_fourier_modes(2, 4)
    aip = AtmosphericAnalyticInnerProducts(params)
    oip = OceanicAnalyticInnerProducts(params)
    aip.connect_to_ocean(oip)
    agotensor = AtmoThermoTensor(params, aip, oip)

    # Symbolic dynamic T test

    # params_t = QgParams({'rr': 287.e0, 'sb': 5.6e-8}, dynamic_T=True)
    # params_t.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})
    # params_t.set_atmospheric_channel_fourier_modes(2, 2, mode='symbolic')
    # params_t.set_oceanic_basin_fourier_modes(2, 4, mode='symbolic')
    #
    # aip = AtmosphericSymbolicInnerProducts(params_t, quadrature=True)  # , stored=False)
    # oip = OceanicSymbolicInnerProducts(params_t, quadrature=True)  # , stored=False)
    # agotensor_t = AtmoThermoTensorDynamicT(params_t, aip, oip)
    #
    # # Symbolic dynamic T4 test
    #
    # params_t4 = QgParams({'rr': 287.e0, 'sb': 5.6e-8}, T4=True)
    # params_t4.set_params({'kd': 0.04, 'kdp': 0.04, 'n': 1.5})
    # params_t4.set_atmospheric_channel_fourier_modes(2, 2, mode='symbolic')
    # params_t4.set_oceanic_basin_fourier_modes(2, 4, mode='symbolic')
    #
    # aip = AtmosphericSymbolicInnerProducts(params_t4, quadrature=True)  # , stored=False)
    # oip = OceanicSymbolicInnerProducts(params_t4, quadrature=True)  # , stored=False)
    #
    # # aip.save_to_file("aip.ip")
    # # oip.save_to_file("oip.ip")
    # #
    # # aip.load_from_file("aip.ip")
    # # oip.load_from_file("oip.ip")
    #
    # agotensor_t4 = AtmoThermoTensorT4(params_t4, aip, oip)
