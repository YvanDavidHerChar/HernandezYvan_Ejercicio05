import numpy as np
import matplotlib.pyplot as plt


cadenaCaracteres = 'scccc'
numc= 0 
nums = 0
for i in range(len(cadenaCaracteres)):
    if cadenaCaracteres[i] == 's':
        nums += 1
    elif cadenaCaracteres[i] == 'c':
        numc += 1
H = np.linspace(1E-5,1-(1E-5),100)


verosimilitud = (H**numc)*(1-H)**nums

Evidencia = np.trapz(verosimilitud,H)

posterior = verosimilitud/Evidencia

print(np.trapz(posterior,H))

L = np.log(posterior)

L_mov = L[1:]

derv_L = (L_mov - L[0:99])/(H[2]-H[1])

zero = np.where(np.abs(derv_L)==np.min(np.abs(derv_L)))

derv_L_mov = derv_L[1:]

seg_derv_L = (derv_L_mov - derv_L[0:98])/(H[2]-H[1])

sigma = (-seg_derv_L[zero])**(-1/2)

print(sigma)

h_zero = H[zero]

def gausiana(x,s,z):
    s = s**2
    f = np.exp(-1/2*(x-z)**2/s)
    f = f/(2*np.pi*s)**(1/2)
    return f
gaus = gausiana(H,sigma,h_zero)

plt.plot(H,gaus, '--')
plt.plot(H,posterior)
plt.title(r"H = {:.2f} $\pm$ {:.2f}".format(float(h_zero) , float(sigma)))
plt.xlabel('H')
plt.ylabel('P(H|{obs})')
plt.savefig("coins.png", bbox_inches='tight')