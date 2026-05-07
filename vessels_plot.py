import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = np.load("av_ele_ref.npy")
dom = np.load('tongue_3D.npy')
c_dom = np.load('c_dom.npy') 

a_element = pd.read_csv('arteries_element_database.csv')
v_element = pd.read_csv('veins_element_database.csv')

art = dom[:,:,1]
ven = dom[:,:,3]

nx, ny, nz = np.shape(dom)

for kk in range(ny):
    for jj in range(nx):
        if (art[jj,kk] == 0):
            d[jj,kk,0]= -2
        if (art[jj,kk] == 1):
            d[jj,kk,0] = -1
        
        if(ven[jj,kk] == 0):
            d[jj,kk,1] = -2
        if(ven[jj,kk] == 1):
            d[jj,kk,1] = -1
            
         
un_a = 121
un_v = 236
un_t = 555723

            

#X = pd.read_csv('Solved Pressure.csv', index_col=None).to_numpy()
X = np.load('solutions/0.007/flow_solution_new_X0_6.4e-05.npy')
P_art = X[2 * un_t:2*un_t + un_a]
P_ven = X[2*un_t + un_a :]

P_art_ele = []
P_ven_ele = []

for ii in range(len(a_element)):
    n1, n2 = a_element.iloc[ii,1:3].values
    P_avg = (P_art[n1] + P_art[n2])/2.0 
    P_art_ele.append(P_avg)

for ii in range(len(v_element)):
    n1, n2 = v_element.iloc[ii,1:3].values
    P_avg = (P_ven[n1] + P_ven[n2])/2.0
    P_ven_ele.append(P_avg)

P_art_dom = np.zeros((nx,ny), dtype = float)
P_ven_dom = np.zeros((nx, ny), dtype = float)
for jj in range(ny):
    for ii in range(nx):
        if d[ii,jj,0] > -1:
            P_art_dom[ii,jj] = P_art_ele[d[ii,jj,0]]
        if d[ii,jj,1] > -1:
            P_ven_dom[ii,jj] = P_ven_ele[d[ii,jj,1]]
            
X1 = X[:183255]
X2 = X[555722+381178:]
#P_art_comp = np.zeros((nx,ny), dtype = float)

for jj in range(ny):
    for ii in range(nx):
        if c_dom[ii,jj,1] > 0:
            P_art_dom[ii,jj] = X1[c_dom[ii,jj,1]]
        if c_dom[ii,jj,3] > 0:
            P_ven_dom[ii,jj] = X2[c_dom[ii,jj,3]-381178]

P_art_dom[P_art_dom == 0] = np.nan
P_ven_dom[P_ven_dom == 0] = np.nan

plt.figure(figsize = (12,6), dpi = 600)
#plt.contourf(P_art_comp, cmap='viridis') #, vmin=10500, vmax=11000
plt.subplot(1, 2, 1)
# plt.contourf(P_art_dom, cmap='viridis_r') #, cmap='plasma'
# plt.colorbar(label='Pressure, Pa', fontsize = 18)
# plt.axis('off')
# plt.gca().invert_yaxis()
#plt.title('Pressure art')
im = plt.contourf(P_art_dom, cmap='viridis_r', vmax=10600)
cbar = plt.colorbar(im) #.set_label('Pressure (Pa)', fontsize=18)
cbar.set_label('Pressure (Pa)', fontsize=18)
cbar.ax.tick_params(labelsize=15, width=2)   #, length=4
plt.axis('off')
plt.gca().invert_yaxis()
plt.title('(a)',y = -0.1, fontsize=18)


plt.subplot(1, 2, 2)
#plt.contourf(P_ven_dom, cmap='viridis_r') #, vmin=10500, vmax=11000)
im2 = plt.contourf(P_ven_dom, cmap='viridis_r')
cbar = plt.colorbar(im2)  #.set_label('Pressure (Pa)', fontsize=18)
cbar.set_label('Pressure (Pa)', fontsize=18)
cbar.ax.tick_params(labelsize=15, width=2)
#plt.title('Pressure ven')
plt.gca().invert_yaxis()
#plt.colorbar(label='Pressure, Pa', fontsize = 18)
plt.axis('off')
plt.title('(b)',y = -0.1, fontsize=18)

plt.tight_layout()
plt.savefig("e=0.007.png", dpi=600)
plt.show()



