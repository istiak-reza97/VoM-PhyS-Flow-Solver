import numpy as np
from time import time
import pandas as pd
import scipy.sparse.linalg as spla
import scipy.sparse as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
from joblib import Parallel, delayed


equal = False

# Parameters
dz = 1e-3
dx = dy = 1e-3

#e = 6.4e-05 # metres   # finite radius of influence
e = 0.001 * 5
print('e = ', e)
E = e

a = 1e-1  # 1e-6       # Perfusion proportionaltiy factor

Ka = 1e-3  # 1e-12     # Permeability of arterial compartment tissue
Kv = 1e-3

myu = 1e-3        # Viscosity of blood

gamma_a = 1  # 1e-5 #1e-14  # This gamma is used for the terminal boundary condition equation
gamma_v = 1  # 1e-5 # 1e-14

P_ba = 10000  # Pa
P_bv = 100  # Pa

Lambda_a = Ka/myu
Lambda_v = Kv/myu

old = True

a_element = pd.read_csv('a_ele.csv')
v_element = pd.read_csv('v_ele.csv')

if(old == True):
    a_outlets = pd.read_csv('a_out.csv')
    v_outlets = pd.read_csv('v_out.csv')


if(old == True):
    ca_t = 'constants/Ca_3D_dx_dy_0.001_e_' + str(e) + '.npy'
    cv_t = 'constants/Cv_3D_dx_dy_0.001_e_' + str(e) + '.npy'

    title_nbr_a = 'nbrhd_matrices/'+str(e)+'/nbrhd_3D_a_dx_dy_0.001_e_' + \
        str(e) + '_new.npy'
    title_nbr_v = 'nbrhd_matrices/'+str(e)+'/nbrhd_3D_v_dx_dy_0.001_e_' + \
        str(e) + '_new.npy'

Ca = np.load(ca_t)
Cv = np.load(cv_t)


nbrhd_a = np.load(title_nbr_a, allow_pickle=True).tolist()
nbrhd_v = np.load(title_nbr_v, allow_pickle=True).tolist()


dom = np.load('testDom.npy')


ny, nx, nz = np.shape(dom)
c_dom = np.load('c_testDom.npy')


Ntvx = np.max(c_dom[:, :, :]) + 1

# Number of elements in the arterial blood vessel network
nPa = len(a_element)
nNa = int(max((max(a_element.iloc[:, 1]), max(a_element.iloc[:, 2])))) + 1

# Number of elements in the venal blood vessel network
nPv = len(v_element)
nNv = int(max((max(v_element.iloc[:, 1]), max(v_element.iloc[:, 2])))) + 1


nbr_a = nbrhd_a
nbr_v = nbrhd_v

arterial_db = a_element
venous_db = v_element

arterial_out = a_outlets.to_numpy()
venous_out = v_outlets.to_numpy()

perfusion = a*dx*dy*dz
dVol = dx*dy*dz


a_inlet = 0
v_inlet = 0

# CALCULATE THE NUMBER OF UNKNONWS #
un_a = len(a_element) + 1
un_v = len(v_element) + 1

un_t = np.max(c_dom) + 1

N_unk = un_a + un_v + 2*un_t


def solve(A, B):
    fm_start = time.time()
    X = spla.spsolve(A, B)
    fm_stop = time.time()
    print(fm_stop - fm_start)
    return (X)


def SoI(center, point, dx, dy, dz):
    x0, y0, z0 = center
    x1, y1, z1 = point

    s = np.sqrt(((x0-x1)*dx)**2 + ((y0-y1)*dy)**2 + ((z0-z1)*dz)**2)

    return(s)


def eta(x, e, C, nC, equal):
    if(x/e < 1):
        if equal == True:
            eta_x = 1/nC
        else:
            eta_x = C*np.exp(1/((abs(x/e))**2 - 1))  # /(e**3) # !!!! NOT DIVIDED BY VOLUME
    else:
        eta_x = 0
    # print(x,eta_x,C)
    return eta_x


tissue_tag = 1
air_tag = 0
a_tag = 2
v_tag = 3

indices_of_zeros = np.argwhere(dom == tissue_tag)
stencil = [(1, 0, 0), (-1, 0, 0), (0, 1, 0),
           (0, -1, 0), (0, 0, 1), (0, 0, -1)]
txStencil = [(dx, dy, dz), (dx, dy, dz), (dy, dz, dx),
             (dy, dz, dx), (dz, dx, dy), (dz, dx, dy)]


def massBalance(ds, Lambda):
    tx = 2*ds[1]*ds[2]*(1/(ds[0]/Lambda + ds[0]/Lambda))
    return tx


def neighbor(st, pt, dom):
    if(dom[pt[0]+st[0], pt[1]+st[1], pt[2] + st[2]] == tissue_tag):
        return True
    else:
        return False


def process_voxel_arterial(iii, dom, c_dom, stencil, txStencil, Lambda_a, perfusion, nbr_a, arterial_db, arterial_out, myu, e, Ca, dx, dy, dz, equal):
    """
    Processes a single voxel for the arterial compartment equations.
    Returns the row, col, and data contributions for this voxel.
    """
    row = []
    col = []
    data = []
    x, y, z = iii
    ijk_sum = 0

    # Loop over stencil
    for kkk, jjj in enumerate(stencil):
        tissueCondition = neighbor(jjj, iii, dom)  # Pass dom here
        if tissueCondition:
            tx = massBalance(txStencil[kkk], Lambda_a)
            ijk_sum += tx

            # Append to arrays
            row.append(c_dom[x, y, z])
            col.append(c_dom[x + jjj[0], y + jjj[1], z + jjj[2]])
            data.append(-tx)

    # Append to arrays
    row.append(c_dom[x, y, z])
    col.append(c_dom[x, y, z])
    data.append(ijk_sum + perfusion)

    row.append(c_dom[x, y, z])
    col.append(un_t + c_dom[x, y, z])
    data.append(-perfusion)

    # Vectorized Operations
    coord = [x, y, z]
    nbrIndex = [i for i, outer_list in enumerate(
        nbr_a) if any(np.array_equal(coord, np.array(nested_list)) for nested_list in outer_list)]
    if len(nbrIndex) != 0:
        for b in nbrIndex:
            ele = arterial_db.loc[arterial_db.iloc[:, 2] == arterial_out[b, 0]]
            r, l = ele.iloc[0, 4], ele.iloc[0, 3]
            k1 = np.pi * r**4 / (8 * myu * l)
            x0, y0, z0 = arterial_out[b, 1:4]
            s = SoI([x0, y0, z0], [x, y, z], dx, dy, dz)  # Pass dx, dy, dz
            n_ex = eta(s, e, Ca[b], len(nbr_a[b]), equal)  # Pass equal
            G = k1 * n_ex * dVol

            # Append to arrays
            row.extend([c_dom[x, y, z], c_dom[x, y, z]])
            col.extend([2 * un_t + ele.iloc[0, 1], 2 * un_t + ele.iloc[0, 2]])
            data.extend([-G, G])

    return row, col, data


def process_voxel_venous(iii, dom, c_dom, stencil, txStencil, Lambda_v, perfusion, nbr_v, venous_db, venous_out, myu, e, Cv, dx, dy, dz, equal):
    """
    Processes a single voxel for the venous compartment equations.
    Returns the row, col, and data contributions for this voxel.
    """
    row = []
    col = []
    data = []

    x, y, z = iii
    ijk_sum = 0

    # Loop over stencil
    for kkk, jjj in enumerate(stencil):
        tissueCondition = neighbor(jjj, iii, dom)  # Pass dom here
        if tissueCondition:
            tx = massBalance(txStencil[kkk], Lambda_v)
            ijk_sum += tx

            # Append to arrays
            row.append(un_t + c_dom[x, y, z])
            col.append(un_t + c_dom[x + jjj[0], y + jjj[1], z + jjj[2]])
            data.append(-tx)

    # Append to arrays
    row.append(un_t + c_dom[x, y, z])
    col.append(un_t + c_dom[x, y, z])
    data.append(ijk_sum + perfusion)

    row.append(un_t + c_dom[x, y, z])
    col.append(c_dom[x, y, z])
    data.append(-perfusion)

    # Vectorized Operations
    coord = [x, y, z]
    nbrIndex = [i for i, outer_list in enumerate(
        nbr_v) if any(np.array_equal(coord, np.array(nested_list)) for nested_list in outer_list)]
    if len(nbrIndex) != 0:
        for b in nbrIndex:
            ele = venous_db.loc[venous_db.iloc[:, 2] == venous_out[b, 0]]
            r, l = ele.iloc[0, 4], ele.iloc[0, 3]
            k1 = np.pi * r**4 / (8 * myu * l)
            x0, y0, z0 = venous_out[b, 1:4]
            s = SoI([x0, y0, z0], [x, y, z], dx, dy, dz)  # Pass dx, dy, dz
            n_ex = eta(s, e, Cv[b], len(nbr_v[b]), equal)  # Pass equal
            G = k1 * n_ex * dVol

            # Append to arrays
            row.extend([un_t + c_dom[x, y, z], un_t + c_dom[x, y, z]])
            col.extend([2 * un_t + un_a + ele.iloc[0, 1],
                       2 * un_t + un_a + ele.iloc[0, 2]])
            data.extend([-G, G])

    return row, col, data


def arterialCompartmentEquations_parallel():
    row = []
    col = []
    data = []

    pool = mp.Pool(mp.cpu_count())  # Use all available cores

    # Prepare arguments for process_voxel
    args = [(iii, dom, c_dom, stencil, txStencil, Lambda_a, perfusion, nbr_a, arterial_db,
             arterial_out, myu, e, Ca, dx, dy, dz, equal) for iii in indices_of_zeros]

    # Run process_voxel in parallel
    results = pool.starmap(process_voxel_arterial, args)

    pool.close()
    pool.join()

    # Combine results
    for r, c, d in results:
        row.extend(r)
        col.extend(c)
        data.extend(d)

    return np.array(row), np.array(col), np.array(data, dtype=float)


def venousCompartmentEquations_parallel():
    row = []
    col = []
    data = []

    pool = mp.Pool(mp.cpu_count())  # Use all available cores

    # Prepare arguments for process_voxel
    args = [(iii, dom, c_dom, stencil, txStencil, Lambda_v, perfusion, nbr_v, venous_db,
             venous_out, myu, e, Cv, dx, dy, dz, equal) for iii in indices_of_zeros]

    # Run process_voxel in parallel
    results = pool.starmap(process_voxel_venous, args)

    pool.close()
    pool.join()

    # Combine results
    for r, c, d in results:
        row.extend(r)
        col.extend(c)
        data.extend(d)

    return np.array(row), np.array(col), np.array(data, dtype=float)


def process_vascular_element(i, offset, offset2, vascInlet, vasc_db, vasc_out, nbr_vasc, Cvasc, gamma_vasc, myu, e, dx, dy, dz, equal):
    """Processes a single vascular element."""
    row = []
    col = []
    data = []
    vascOutdb = pd.DataFrame(vasc_out)
    eqn_n = offset + i
    ij_sum = 0
    if i != vascInlet and i not in vasc_out[:, 0]:
        ele = vasc_db.loc[vasc_db.iloc[:, 1] == i]
        for j in range(len(ele)):
            r = ele.iloc[j, 4]
            L = ele.iloc[j, 3]
            phi = np.pi * (r)**4 / (8 * myu * L)
            ij_sum = ij_sum + phi
            row.append(eqn_n)
            col.append(offset + ele.iloc[j, 2])
            data.append(-phi)
        ele = vasc_db.loc[vasc_db.iloc[:, 2] == i]
        for j in range(len(ele)):
            r = ele.iloc[j, 4]
            L = ele.iloc[j, 3]
            phi = np.pi * (r)**4 / (8 * myu * L)
            ij_sum = ij_sum + phi
            row.append(eqn_n)
            col.append(offset + ele.iloc[j, 1])
            data.append(-phi)
        row.append(eqn_n)
        col.append(offset + i)
        data.append(ij_sum)

    if i in vascOutdb.iloc[:, 0].values:
        i_index = vascOutdb[vascOutdb.iloc[:, 0] == i].index[0]
        ele = vasc_db.loc[vasc_db.iloc[:, 2] == i]
        r = ele.iloc[0, 4]
        L = ele.iloc[0, 3]
        k = np.pi * (r)**4 / (8 * myu * L)
        row.append(eqn_n)
        col.append(offset + ele.iloc[0, 1])
        data.append(-k)
        row.append(eqn_n)
        col.append(eqn_n)
        data.append((k + gamma_vasc / myu))
        nexSum = 0
        nbr_vasc_i = nbr_vasc[i_index]
        Cvasc_i = Cvasc[i_index]

        for m in range(len(nbr_vasc_i)):
            x, y, z = nbr_vasc_i[m]
            x0, y0, z0 = (vascOutdb.iloc[i_index, 1:4]).tolist()
            s = SoI([x0, y0, z0], [x, y, z], dx, dy, dz)  # Pass dx, dy, dz
            n_ex = eta(s, e, Cvasc_i, len(nbr_vasc_i), equal)  # Pass equal
            nexSum += n_ex
            chi = n_ex * gamma_vasc / myu  # *dx*dy*dz

            row.append(eqn_n)
            col.append(offset2 + c_dom[x, y, z])
            data.append(-chi)
        print(nexSum)

    return row, col, data


def segmentedVasculature_parallel(offset, offset2, vascInlet, vasc_db, vasc_out, nbr_vasc, Cvasc, gamma_vasc):
    row = []
    col = []
    data = []

    element_indices = range(len(vasc_db) + 1)

    pool = mp.Pool(mp.cpu_count())
    args = [(i, offset, offset2, vascInlet, vasc_db, vasc_out, nbr_vasc, Cvasc, gamma_vasc, myu, e, dx, dy, dz, equal)
            for i in element_indices]
    results = pool.starmap(process_vascular_element, args)
    pool.close()
    pool.join()

    for r, c, d in results:
        row.extend(r)
        col.extend(c)
        data.extend(d)

    return np.array(row), np.array(col), np.array(data, dtype=float)


if __name__ == '__main__':
    totalTime = time()
    r1, c1, d1 = arterialCompartmentEquations_parallel()
    r2, c2, d2 = venousCompartmentEquations_parallel()
    r3, c3, d3 = segmentedVasculature_parallel(
        2 * un_t, 0, a_inlet, arterial_db, arterial_out, nbr_a, Ca, gamma_a)
    r4, c4, d4 = segmentedVasculature_parallel(
        2 * un_t + un_a, un_t, v_inlet, venous_db, venous_out, nbr_v, Cv, gamma_v)
    print('Total time taken for matrix generation = ',
          (time() - totalTime) / 60, ' mins')

    row = np.append(r1, r2)
    col = np.append(c1, c2)
    data = np.append(d1, d2)

    row = np.append(row, r3)
    col = np.append(col, c3)
    data = np.append(data, d3)

    row = np.append(row, r4)
    col = np.append(col, c4)
    data = np.append(data, d4)

    row = np.append(row, 2 * un_t + 0)
    col = np.append(col, 2 * un_t + 0)
    data = np.append(data, 1)

    row = np.append(row, 2 * un_t + un_a)
    col = np.append(col, 2 * un_t + un_a)
    data = np.append(data, 1)

    A = sp.csc_matrix((np.array(data), (np.array(row), np.array(col))),
                      shape=(N_unk, N_unk))  # .todense()
    Adense = A.todense()

    plt.figure(figsize=(10, 10), dpi=300)
    plt.spy(A)
    plt.show()

    b = np.zeros((N_unk), dtype=int)
    b[2 * un_t] = P_ba
    b[2 * un_t + un_a] = P_bv

    X = spla.spsolve(A, b)

    Z = A.dot(np.ones(N_unk))

    Z2 = A.dot(X)

    PArt = X[2 * un_t:2 * un_t + un_a]
    PVrt = X[2 * un_t + un_a:]

    Q_in = (PArt[0] - PArt[1]) * np.pi * \
        a_element.iloc[0, 4]**4 / (8 * myu * a_element.iloc[0, 3])
    Q_out = (PVrt[0] - PVrt[1]) * np.pi * \
        v_element.iloc[0, 4]**4 / (8 * myu * v_element.iloc[0, 3])

    print('mass conservation error = ', Q_in + Q_out)
