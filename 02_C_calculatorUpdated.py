import numpy as np
import pandas as pd
#import neighbourhood_matrixUpdated as nbm
from concurrent.futures import ProcessPoolExecutor

old = False

a_element = pd.read_csv('arteries_element_database.csv')
v_element = pd.read_csv('veins_element_database.csv')
# a_element = pd.read_csv('a_ele_trial.csv')
# v_element = pd.read_csv('v_ele.csv')



a_outlets = pd.read_csv('arteries_outlet_coordinates_3D_shifted.csv')
v_outlets = pd.read_csv('veins_outlet_coordinates_3D_shifted.csv')
# a_outlets = pd.read_csv('a_out_trial.csv')
# v_outlets = pd.read_csv('v_out.csv')





#e =  6.4e-05
e = 0.001 * 10
dx = dy =6.4e-5    # metres
dz = 0.333e-3     # metres

if(old == True):
    title_nbr_a = 'nbrhd_matrices/'+str(e)+'/nbrhd_old_a_dx_dy_'+str(dx)+'_e_' + str(e) + '.npy'
    title_nbr_v = 'nbrhd_matrices/'+str(e)+'/nbrhd_old_v_dx_dy_'+str(dx)+'_e_' + str(e) + '.npy'
    
if(old == False):
    title_nbr_a = 'nbrhd_matrices/'+str(e)+'/nbrhd_3D_a_dx_dy_'+str(dx)+'_e_' + str(e) + '_new.npy'
    title_nbr_v = 'nbrhd_matrices/'+str(e)+'/nbrhd_3D_v_dx_dy_'+str(dx)+'_e_' + str(e) + '_new.npy'

nbrhd_a = np.load(title_nbr_a, allow_pickle=True).tolist()
nbrhd_v = np.load(title_nbr_v, allow_pickle=True).tolist()

def artery():
    global dx
    global dy
    global dz
    global e
    global a_outlets
    global nbrhd_a
    
    C_a = []
    
    for i in range(len(a_outlets)):
        c = []
        C_a.append(c)
    
    for i in range(len(a_outlets)):
        x0,y0,z0 = a_outlets.iloc[i,1:].values
        sum_exp = 0.0
        for j in range(len(nbrhd_a[i])):
            y,x,z = nbrhd_a[i][j]
            #print([x,y,z], [x0,y0,z0])
            s = np.sqrt( ((x-x0)*dx)**2 + ((y-y0)*dy)**2 + ((z-z0)*dz)**2) 
            if(s/e < 1):
                #print(s, s/e)
                exp = np.exp(1/(abs(s/e)**2 - 1))#/(e**3) #;print(s,s/e)
                sum_exp = sum_exp + exp ; #print(exp)
        C_a[i].append(1/sum_exp)
    
    if(old == True):
        ca_title = 'constants/Ca_old_dx_dy_'+str(dx) +'_e_' + str(e) + '.npy' 
    if(old == False):
        ca_title = 'constants/Ca_3D_dx_dy_'+str(dx) +'_e_' + str(e) + '.npy' 
    print(C_a)
    np.save(ca_title,np.array(C_a))
    

def veins():
    global dx
    global dy
    global dz
    global e
    global v_outlets
    global nbrhd_v
    
    C_v = []
    
    for j in range(len(v_outlets)):
        c = []
        C_v.append(c)
        
    for i in range(len(v_outlets)):
        x0,y0,z0 = v_outlets.iloc[i,1:].values
        sum_exp = 0.0
        for j in range(len(nbrhd_v[i])):
            y,x,z = nbrhd_v[i][j]
            s = np.sqrt( ((x-x0)*dx)**2 + ((y-y0)*dy)**2 + ((z-z0)*dz)**2) 
            # print(s,e)
            if(s/e < 1):
                exp = np.exp(1/(abs(s/e)**2 - 1))#/(e**3) ;#print(s/e)
                sum_exp = sum_exp + exp ; #print(exp)
        C_v[i].append(1/sum_exp)
    
    if(old == True):
        cv_title = 'constants/Cv_old_dx_dy_'+str(dx) +'_e_' + str(e) + '.npy' 
    if(old == False):
        cv_title = 'constants/Cv_3D_dx_dy_'+str(dx) +'_e_' + str(e) + '.npy' 
    np.save(cv_title,np.array(C_v))
    #print(C_v)
    
artery()
veins()

#def main():
#    with ProcessPoolExecutor(max_workers=3) as executor_p:
#        executor_p.submit(artery)
#        executor_p.submit(veins)
#        
#if __name__=='__main__':
#    main()


print('finished')
