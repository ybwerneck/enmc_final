import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import Poisson

####MODELO######
vw = 0.44671545
phi = 0.32142857142 #porosidade
mi_w = 0.001 #viscosidade da água 20 graus 
mi_g = 0.00018 #viscosidade do ar 20 graus (ar)
MRF = 1.0
q = 0.2998155 #velocidade de Darcy (calculada)
#krw = 10*(-9)

def PermEff(S):
        lamb = 2 #mobilidade total*
        krg = 10**(-10) #permeabilidade efetiva (gás)
        krw = 10**(-9) #permeabilidade efetiva (água)
        Swc = 0.0 #Saturação da água*
        Sgr = 0.0 #Saturação do gás*
        Swe = (S-Swc)/(1-S-Sgr)  #equação 7
        k_w = krw*(Swe**lamb) #equação 8
        k_g = krg*((1-Swe)**(3+(2/lamb))) #equação 9
        if(S!=0):
            print(S,k_w,k_g)
        return k_w,k_g


def f(Sw):
   krw1,krg1=PermEff(abs(Sw))
   lw=  (krw1 / mi_w)
   lg= (krg1 / (mi_g * MRF))
   lt= lw + lg
   res = lw/lt
   #print(res)
   return res
   
   #return res

def init(L,dl):
    nl=int(L/dl)
    r= np.zeros((nl,nl))
    return r


def Solve2d(L: float, dl: float, t: float, dt: float, Sw0: float, times: list) -> list:
    nt, nl = int(t/dt), int(L/dl)

    rw = (vw*dt)/(dl*phi)
    W, U = Poisson.poisson(dl, L)  # Assuming you have Poisson's solver
    W, U = W/np.linalg.norm(W), U/np.linalg.norm(U)

    Sw = np.zeros((nt+1, nl, nl))
    Sw[0] = init(L, dl)

    Sw_list = []

    Sw[0, nl-3:nl, 0:3] = 0.99
    for k in range(1, nt+1):
        print("done ",k/nt * 100)

        if k*dt in times:
            Sw_list.append(np.copy(Sw[k-1]))

        for i in range(1, nl-1):
            for j in range(1, nl-1):
                C = 1/(dl) 
                v = C*U[i, j]
                w = C*W[i, j]

                if v <= 0:
                    Fy = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i+1, j]))
                else:
                    Fy = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i-1, j]))
                if w <= 0:
                    Fx = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i, j+1]))
                else:
                    Fx = (1/dl)*(f(Sw[k-1, i, j])-f(Sw[k-1, i, j-1]))

                Sw[k, i, j] = Sw[k-1, i, j] + (dt/phi)*(np.abs(v)*Fy + np.abs(w)*Fx)
                if Sw[k, i, j] > 1:
                    return Sw_list

        Sw[k, nl-3:nl, 0:3] = 0.99

    return Sw_list


L = 1
dl = 0.01
t = 10
dt = 0.01
Sw0 = 0
times_of_interest = [1,2,3,4,5,6,7,8,9]  # Adjust this list with the specific times you want

solutions = Solve2d(L, dl, t, dt, Sw0, times_of_interest)

directory = 'agua'  # Replace with your subfolder path
for idx, time in enumerate(times_of_interest):
    ref_csv_filename = f'concentration_{idx+2}.csv'
    ref_csv_path = os.path.join(directory, ref_csv_filename)
    ref_matrix = np.loadtxt(ref_csv_path, delimiter=',')

    calculated_matrix = solutions[idx]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(calculated_matrix, cmap='viridis', origin='lower', vmin=0, vmax=np.max(calculated_matrix))
    plt.colorbar(label='Sw')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Calculated Sw at time ' + str(time))

    plt.subplot(1, 2, 2)
    plt.imshow(np.rot90(ref_matrix), cmap='viridis', origin='lower', vmin=0, vmax=1)
    plt.colorbar(label='Sw')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reference Sw at time ' + str(time))

    plt.tight_layout()
    plt.show()
    plt.savefig(directory+"/ajuste_at_"+str(time)+".png")


