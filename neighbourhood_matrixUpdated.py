import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import os 
from math import ceil, floor

#dx = dy = 1e-3
#dz = 1e-3
dx = dy =6.4e-5    # metres
dz = 0.333e-3     # metres


e = 0.001 * 10
old = False


a_outlets = pd.read_csv('arteries_outlet_coordinates_3D_shifted.csv')
v_outlets = pd.read_csv('veins_outlet_coordinates_3D_shifted.csv')

dom = np.load('tongue_3D.npy')

ny, nx, nz = np.shape(dom)


nbr_a = []
nbr_v = []

for n in range(len(a_outlets)):
    nbr_a.append([])

for n in range(len(v_outlets)):
    nbr_v.append([])
    

for iii in range(len(a_outlets)):
    x0,y0,z0 = a_outlets.iloc[iii,1:].values
    x_min = max(0, x0 - floor(e/dx))
    x_max = min(nx, x0 + ceil(e/dx))
    y_min = max(0, y0 - floor(e/dy))
    y_max = min(ny, y0 + ceil(e/dy))
    z_min = max(0, z0 - floor(e/dz))
    z_max = min(nz, z0 + ceil(e/dz))
    
    for zzz in range(z_min, z_max):
        for yyy in range(y_min, y_max):
            for xxx in range(x_min, x_max):
                #print(iii,dom[yyy,xxx,zzz] )
                if(dom[yyy,xxx,zzz] == 1):    ##x,y are interchanged in the frog tongue
                    s = np.sqrt( ((xxx-x0)*dx)**2 + ((yyy-y0)*dy)**2 + ((zzz-z0)*dz)**2)
                    if s/e < 1:
                        #print(s, s/e, yyy,xxx,zzz)
                        nbr_a[iii].append([yyy,xxx,zzz])
                    

print('1')
    

for iii in range(len(v_outlets)):
    x0,y0,z0 = v_outlets.iloc[iii,1:].values
    x_min = max(0, x0 - floor(e/dx))
    x_max = min(nx, x0 + ceil(e/dx))
    y_min = max(0, y0 - floor(e/dy))
    y_max = min(ny, y0 + ceil(e/dy))
    z_min = max(0, z0 - floor(e/dz))
    z_max = min(nz, z0 + ceil(e/dz))
    
    for zzz in range(z_min, z_max):
        for yyy in range(y_min, y_max):
            for xxx in range(x_min, x_max):
                if(dom[yyy,xxx,zzz] == 1):
                    s = np.sqrt( ((xxx-x0)*dx)**2 + ((yyy-y0)*dy)**2 + ((zzz-z0)*dz)**2)
                    if s/e < 1:
                        nbr_v[iii].append([yyy,xxx,zzz])

print('2')
                   
path = 'nbrhd_matrices/' + str(e)
try:
    os.mkdir(path)
except OSError as error:
    print(error)
    
artery_nbrhd = np.array(nbr_a, dtype = object)
title_a = 'nbrhd_matrices/' + str(e)+'/nbrhd_3D_a_dx_dy_'+str(dx)+'_e_' + str(e) + '_new.npy'
np.save(title_a,artery_nbrhd)

veins_nbrhd = np.array(nbr_v, dtype = object)
title_v = 'nbrhd_matrices/' + str(e)+'/nbrhd_3D_v_dx_dy_'+str(dx)+'_e_' + str(e) + '_new.npy'
np.save(title_v,veins_nbrhd)

print('finished')
