import numpy as np
import math
import PIL.Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

number_of_dots = 24

def f(vector):
    return [x / vector[2] for x in vector]

def init():
    
    x = [
        np.array([335, 75, 1]), #x1
        np.array([555, 55, 1]), #x2
        np.array([720, 172, 1]), #x3
        np.array([544, 194, 1]), #x4
        np.array([333, 296, 1]), #x5 
        np.array([0, 0, 1]), #x6 nevidljiva
        np.array([714, 401, 1]), #x7
        np.array([539, 431, 1]), #x8
        np.array([264, 343, 1]), #x9
        np.array([0, 0, 1]), #x10 nevidljiva
        np.array([778, 368, 1]), #x11
        np.array([316, 412, 1]), #x12
        np.array([268, 591, 1]), #x13 
        np.array([0, 0, 1]), #x14 nevidljiva
        np.array([771, 617, 1]), #x15
        np.array([317, 669, 1]), #x16 
        np.array([94, 631, 1]), #x17
        np.array([0, 0, 1]), #x18 nevidljiva
        np.array([927, 603, 1]), #x19
        np.array([703, 783, 1]), #x20
        np.array([96, 826, 1]), #x21
        np.array([0, 0, 1]), #x22 nevidljiva
        np.array([921, 788, 1]), #x23
        np.array([698, 991, 1]), #x24
    ]
    
    
    y = [
        np.array([396, 77, 1]), #y1
        np.array([564, 80, 1]), #y2
        np.array([568, 198, 1]), #y3
        np.array([373, 198, 1]), #y4
        np.array([0, 0, 1]), #y5 nevidljiva
        np.array([0, 0, 1]), #y6 nevidljiva
        np.array([568, 423, 1]), #y7
        np.array([379, 422, 1]), #y8
        np.array([284, 315, 1]), #y9
        np.array([237, 376, 1]), #y10 
        np.array([690, 403, 1]), #y11
        np.array([717, 335, 1]), #y12
        np.array([0, 0, 1]), #y13 nevidljiva
        np.array([719, 555, 1]), #y14 
        np.array([688, 639, 1]), #y15
        np.array([251, 617, 1]), #y16 
        np.array([127, 550, 1]), #y17
        np.array([0, 0, 1]), #y18 nevidljiva
        np.array([863, 654, 1]), #y19
        np.array([463, 783, 1]), #y20
        np.array([133, 721, 1]), #y21
        np.array([0, 0, 1]), #y22 nevidljiva
        np.array([858, 838, 1]), #y23
        np.array([466, 975, 1]), #y24
    ]
    
    
    #skrivene tacke
    
    x[5] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x[0], x[4])), f(np.cross(x[6], x[2])))), x[1])),
              f(np.cross(f(np.cross(f(np.cross(x[0], x[3])), f(np.cross(x[2], x[1])))), x[6]))))
    x[5] = np.round(x[5])
    
    x[9] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x[8], x[11])), f(np.cross(x[15], x[12])))), x[10])),
              f(np.cross(f(np.cross(f(np.cross(x[11], x[10])), f(np.cross(x[14], x[15])))), x[8]))))
    x[9] = np.round(x[9])
    
    x[13] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x[11], x[10])), f(np.cross(x[15], x[14])))), x[12])),
              f(np.cross(f(np.cross(f(np.cross(x[8], x[11])), f(np.cross(x[15], x[12])))), x[14]))))
    x[13] = np.round(x[13])
    
    x[17] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x[16], x[19])), f(np.cross(x[20], x[23])))), x[18])),
              f(np.cross(f(np.cross(f(np.cross(x[19], x[18])), f(np.cross(x[22], x[23])))), x[16]))))
    x[17] = np.round(x[17])
   
    x[21] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(x[19], x[18])), f(np.cross(x[22], x[23])))), x[20])),
              f(np.cross(f(np.cross(f(np.cross(x[16], x[19])), f(np.cross(x[23], x[20])))), x[22]))))
    x[21] = np.round(x[21])
    
    
    y[4] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y[3], y[7])), f(np.cross(y[6], y[2])))), y[0])),
              f(np.cross(f(np.cross(f(np.cross(y[0], y[3])), f(np.cross(y[2], y[1])))), y[7]))))
    y[4] = np.round(y[4])
    
    y[5] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y[0], y[4])), f(np.cross(y[6], y[2])))), y[1])),
              f(np.cross(f(np.cross(f(np.cross(y[0], y[3])), f(np.cross(y[2], y[1])))), y[6]))))
    y[5] = np.round(y[5])
    
    #kriticna
    y[12] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y[8], y[9])), f(np.cross(y[10], y[11])))), y[15])),
              f(np.cross(f(np.cross(f(np.cross(y[8], y[11])), f(np.cross(y[10], y[9])))), y[15]))))
    y[12] = np.round(y[12])
    
    y[17] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y[16], y[19])), f(np.cross(y[20], y[23])))), y[18])),
              f(np.cross(f(np.cross(f(np.cross(y[19], y[18])), f(np.cross(y[22], y[23])))), y[16]))))
    y[17] = np.round(y[17])
    
    #i ova je malo kriticna
    y[21] = f(np.cross(f(np.cross(f(np.cross(f(np.cross(y[19], y[18])), f(np.cross(y[22], y[23])))), y[20])),
              f(np.cross(f(np.cross(f(np.cross(y[16], y[19])), f(np.cross(y[23], y[20])))), y[22]))))
    y[21] = np.round(y[21])
     
    return x, y

def fundamental_matrix(x, y):
    jed8 = np.zeros((8, 9))
    
    for i in range(4):
        jed8[i] = [x[i][0] * y[i][0],
                   x[i][1] * y[i][0],
                   x[i][2] * y[i][0],
                   x[i][0] * y[i][1],
                   x[i][1] * y[i][1],
                   x[i][2] * y[i][1],
                   x[i][0] * y[i][2],
                   x[i][1] * y[i][2],
                   x[i][2] * y[i][2]]
    for i in range(8,12):
        jed8[i-4] = [x[i][0] * y[i][0],
                   x[i][1] * y[i][0],
                   x[i][2] * y[i][0],
                   x[i][0] * y[i][1],
                   x[i][1] * y[i][1],
                   x[i][2] * y[i][1],
                   x[i][0] * y[i][2],
                   x[i][1] * y[i][2],
                   x[i][2] * y[i][2]]
        
    U, D, V = np.linalg.svd(jed8)
    last = V[:][-1]
    f = np.reshape(last, (3, 3))

    return f

def epipoles(FF):
    u, d, v = np.linalg.svd(FF)
    e1 = v[2][:]
    e1 = (1/e1[2])*e1
    
    e2 = u.T[2][:]
    e2 = (1/e2[2])*e2
    
    return e1, e2

def jednacine(xx, yy, T1, T2):
    return np.array([xx[1]*T1[2]-xx[2]*T1[1],
                    -xx[0]*T1[2]+xx[2]*T1[0],
                    yy[1]*T2[2]-yy[2]*T2[1],
                    -yy[0]*T2[2]+yy[2]*T2[0]])

def UAfine(XX):
    XX = XX /XX[3]
    XX = np.array([XX[0], XX[1], XX[2]])
    return XX

def TriD(xx, yy, T1, T2):
    U, D, V = np.linalg.svd(jednacine(xx, yy, T1, T2))
    Vp = V[3]
    Vafino = UAfine(Vp)
    return Vafino

def main():
    x, y = init()
    
    
    FF = fundamental_matrix(x, y)
    
    print("fundamentalna matrica:")
    print(FF)
    
    print("determinanta FF:")
    print(np.linalg.det(FF))
    
    e1, e2 = epipoles(FF)
    print("epipolovi FF:")
    print("e1:", e1, "\ne2:", e2)
    
    FFX = FF 
    U, D, V = np.linalg.svd(FFX)
    D[2] = 0
    DD = [[D[0], 0, 0], [0, D[1], 0], [0, 0, 0]]
    FF1 = U @ DD @ V
    
    print("fundamentalna matrica FF1:")
    print(FF1)
    print("determinanta FF1:")
    print(np.linalg.det(FF1))
    
    e1, e2 = epipoles(FF1)
    print("epipolovi FF1:")
    print("e1:", e1, "\ne2:", e2)
    
    e2_matrix = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    print("E2 matrica:")
    print(e2_matrix)
    
    T1 = np.array([
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0]
    ])
    print("T1:")
    print(T1)
    
    T2_TMP = e2_matrix @ FF1 
    T2 = np.array([
        [T2_TMP[0][0], T2_TMP[0][1], T2_TMP[0][2], e2[0]],
        [T2_TMP[1][0], T2_TMP[1][1], T2_TMP[1][2], e2[1]],
        [T2_TMP[2][0], T2_TMP[2][1], T2_TMP[2][2], e2[2]]
    ])
    print("T2:")
    print(T2)
    
    C1 = [0, 0, 0, 1]
    U, D, V = np.linalg.svd(T2)
    C2 = V[:][-1]
    
    print("koordinate prve kamere C1:")
    print(C1)
    print("koordinate druge kamere C2:")
    print(C2)
    
    U, D, V = np.linalg.svd(jednacine(x[0], y[0], T1, T2))
    print(V[3])
    print("U:")
    print(U)
    print("D:")
    print(D)
    print("V:")
    print(V)
    
    rekonstruisane = []
    for i in range(number_of_dots):
        tmp = TriD(x[i], y[i], T1, T2)
        rekonstruisane.append(tmp)
    dig = np.eye(3)
    dig[2][2] = 400
    rekonstruisane400 = np.zeros((number_of_dots, 3))
    
    for i in range(number_of_dots):
        rekonstruisane400[i] = dig.dot(rekonstruisane[i])
        print(rekonstruisane400[i])
    '''    
    print(rekonstruisane)
    X = rekonstruisane400
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:4, 0], X[:4, 1], X[:4, 2], color="blue")
    ax.scatter(X[4:8, 0], X[4:8, 1], X[4:8, 2], color="blue")
    ax.scatter(X[8:12, 0], X[8:12, 1], X[8:12, 2], color="red")
    ax.scatter(X[12:16, 0], X[12:16, 1], X[12:16, 2], color="red")
    plt.gca().invert_yaxis()
    plt.show()
    '''
    
    #image = PIL.Image.open("desnaNew.jpg")
    #image.show()
    
if __name__ == "__main__":
    main()
