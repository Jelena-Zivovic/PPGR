import numpy as np
import numpy.linalg
import math
import copy
from tkinter import *
from tkinter import filedialog
import sys
import PIL.Image


def naivniAlgoritam(originali, slike):

    #matrica sistema originalnih tacaka
    matricaSistema1 = np.array([
        [originali[0][0], originali[1][0], originali[2][0]],
        [originali[0][1], originali[1][1], originali[2][1]],
        [originali[0][2], originali[1][2], originali[2][2]]
    ])

    #koordinate cetvrte tacke
    rezultat1 = np.array([originali[3][0], originali[3][1], originali[3][2]])
    
    #resavanje sistema jednacina
    resenjeSistema1 = np.linalg.solve(matricaSistema1, rezultat1)

    #izdvajanje sva 3 resenja
    alfa1 = resenjeSistema1[0]
    beta1 = resenjeSistema1[1]
    gama1 = resenjeSistema1[2]

    #formiranje odgovarajuce matrice preslikavanja uz pomoc dobijenih koeficijenata
    kolona11 = np.array([alfa1*originali[0][0], alfa1*originali[0][1], alfa1*originali[0][2]])
    kolona12 = np.array([beta1*originali[1][0], beta1*originali[1][1], beta1*originali[1][2]])
    kolona13 = np.array([gama1*originali[2][0], gama1*originali[2][1], gama1*originali[2][2]])

    p1 = np.column_stack([kolona11, kolona12, kolona13])

    #racunanje inverza prve matrice preslikavanja
    p1 = np.linalg.inv(p1)

    #matrica sistema tacaka koje predstavljaju slike originalnih tacaka
    matricaSistema2 = np.array([
        [slike[0][0], slike[1][0], slike[2][0]],
        [slike[0][1], slike[1][1], slike[2][1]],
        [slike[0][2], slike[1][2], slike[2][2]]
    ])

    rezultat2 = np.array([slike[3][0], slike[3][1], slike[3][2]])

    #resavanja drugog sistema
    resenjeSistema2 = np.linalg.solve(matricaSistema2, rezultat2)

    alfa2 = resenjeSistema2[0]
    beta2 = resenjeSistema2[1]
    gama2 = resenjeSistema2[2]

    #formiranje odgovarajuce matrice preslikavanja uz pomoc dobijenih koeficijenata
    kolona21 = np.array([alfa2*slike[0][0], alfa2*slike[0][1], alfa2*slike[0][2]])
    kolona22 = np.array([beta2*slike[1][0], beta2*slike[1][1], beta2*slike[1][2]])
    kolona23 = np.array([gama2*slike[2][0], gama2*slike[2][1], gama2*slike[2][2]])

    p2 = np.column_stack([kolona21, kolona22, kolona23])

    #racunanje konacne matrice preslikavanja
    p = np.dot(p2, p1)

    return p

def DLT(originali, slike):
    x1 = float(originali[0][0])
    x2 = float(originali[0][1])
    x3 = float(originali[0][2])

    x1p = float(slike[0][0])
    x2p = float(slike[0][1])
    x3p = float(slike[0][2])

    #formiranje prva dva reda matrice A
    A = np.array([
        [0, 0, 0, (-1)*x3p*x1, (-1)*x3p*x2, (-1)*x3p*x3, x2p*x1, x2p*x2, x2p*x3],
        [x3p*x1, x3p*x2, x3p*x3, 0, 0, 0, (-1)*x1p*x1, (-1)*x1p*x2, (-1)*x1p*x3]
    ])

    #formiranje ostatka matrice A
    for i in range(1, len(originali)):
        x1 = originali[i][0]
        x2 = originali[i][1]
        x3 = originali[i][2]

        x1p = slike[i][0]
        x2p = slike[i][1]
        x3p = slike[i][2]     

        r1 = np.array([0, 0, 0, -x3p*x1, -x3p*x2, -x3p*x3, x2p*x1, x2p*x2, x2p*x3])
        r2 = np.array([x3p*x1, x3p*x2, x3p*x3, 0, 0, 0, -x1p*x1, -x1p*x2, -x1p*x3])

        A = np.vstack((A, r1))
        A = np.vstack((A, r2))

    #SVD dekompozicija matrice, dobijaju se 3 matrice: U, D i V
    U, D, V = np.linalg.svd(A)
 
    #matrica preslikavanja je poslednja vrsta matrice V
    P = V[-1].reshape(3, 3)
 
    return P

#normalizacija tacaka
def normalize(tacke):
    x = 0.0
    y = 0.0

    udaljenost = 0.0

    for i in range(len(tacke)):
        x = x + float(tacke[i][0]) / float(tacke[i][2])
        y = y + float(tacke[i][1]) / float(tacke[i][2])

    #x i y su afine koordinate tezista tacaka
    x = x / float(len(tacke))
    y = y / float(len(tacke))

    for i in range(len(tacke)):
        #svaka 
        tmp1 = tacke[i][0]/tacke[i][2] - x
        tmp2 = tacke[i][1]/tacke[i][2] - y

        udaljenost = udaljenost + math.sqrt(tmp1**2 + tmp2**2)

    udaljenost = udaljenost / float(len(tacke))

    k = math.sqrt(2) / udaljenost 

    S = np.array([[k, 0, -k*x], [0, k, -k*y], [0, 0, 1]])

    return S

def dlt_normalize(originali, slike):

    #matrice T i Tp su matrice normalizacije originalnih tacaka i njihovih slika
    T = normalize(originali)
    Tp = normalize(slike)

    #kopije originalnih tacaka i njihovih slika
    originali_copy = copy.deepcopy(originali)
    slike_copy = copy.deepcopy(slike)

    for i in range(len(originali)):
        #normalizacija originalnih tacaka
        [x, y, z] = np.dot(T, [originali[i][0], originali[i][1], originali[i][2]])

        originali_copy[i][0] = float(x) 
        originali_copy[i][1] = float(y)
        originali_copy[i][2] = float(z)

    for i in range(len(slike)):
        #normalizacija slika
        [x, y, z] = np.dot(Tp, [float(slike[i][0]), float(slike[i][1]), float(slike[i][2])])

        slike_copy[i][0] = float(x) 
        slike_copy[i][1] = float(y)
        slike_copy[i][2] = float(z)

    #primena DLT algoritma na normalizovane originalne tacke i njihove slike
    Pp = DLT(originali_copy, slike_copy)

    #odgovarajuca matrica preslikavanja
    P = np.dot(np.linalg.inv(Tp), Pp)
    P = np.dot(P, T)

    return P


root = Tk()

frame1 = Frame(root)
frame1.pack()

frame2 = Frame(root)
frame2.pack()

frame3 = Frame(root)
frame3.pack()

frame4 = Frame(root)
frame4.pack()

def prikaziNaivni():
    global frame1
    labelNaivniUnesi = Label(frame1, text="Unesite homogene koordinate originalnih tacaka kao i njihovih slika: ")
    labelNaivniUnesi.grid(row=1)

    originaliEntry = []
    slikeEntry = []

    for i in range(4):
        eOriginal = Entry(frame1)
        originaliEntry.append(eOriginal)
        eOriginal.grid(row=i+2, column=0)
        

        eSlika = Entry(frame1)
        slikeEntry.append(eSlika)
        eSlika.grid(row=i+2, column=1)
        

    def izracunajNaivni():
        originali = []
        slike = []

        for i in range(4):
            x = float(originaliEntry[i].get().strip().split(' ')[0])
            y = float(originaliEntry[i].get().strip().split(' ')[1])
            z = float(originaliEntry[i].get().strip().split(' ')[2])

            originali.append([x, y, z])

            xp = float(slikeEntry[i].get().strip().split(' ')[0])
            yp = float(slikeEntry[i].get().strip().split(' ')[1])
            zp = float(slikeEntry[i].get().strip().split(' ')[2])

            slike.append([xp, yp, zp])

        P = naivniAlgoritam(originali, slike)

        labelIspis = Label(frame1, text="Odgovarajuca matrica preslikavanja:")
        labelIspis.grid(row=8)

        labelMatrica = Label(frame1, text=str(P))
        labelMatrica.grid(row=9)

    buttonOk = Button(frame1, text="OK", command=izracunajNaivni)
    buttonOk.grid(row=7)
 

def prikaziDLT():
    global frame2

    labelBrojTacaka = Label(frame2, text="Unesite broj tacaka:")
    labelBrojTacaka.grid(row=2)

    eBrojTacaka = Entry(frame2)
    eBrojTacaka.grid(row=3)

    originali = []
    slike = []

    def unetBrojTacaka():
        brojTacaka = int(eBrojTacaka.get().strip())

        originaliEntry = []
        slikeEntry = []

        labelUnesi = Label(frame2, text="Unesite homogene koordinate originalnih tacaka i njihovih slika:")
        labelUnesi.grid(row=5)

        for i in range(brojTacaka):
            eOriginal = Entry(frame2)
            originaliEntry.append(eOriginal)
            eOriginal.grid(row=i+6, column=0)
            

            eSlika = Entry(frame2)
            slikeEntry.append(eSlika)
            eSlika.grid(row=i+6, column=1)

        def izracunajDLT():
           

            for i in range(brojTacaka):
                x = float(originaliEntry[i].get().strip().split(' ')[0])
                y = float(originaliEntry[i].get().strip().split(' ')[1])
                z = float(originaliEntry[i].get().strip().split(' ')[2])

                originali.append([x, y, z])

                xp = float(slikeEntry[i].get().strip().split(' ')[0])
                yp = float(slikeEntry[i].get().strip().split(' ')[1])
                zp = float(slikeEntry[i].get().strip().split(' ')[2])

                slike.append([xp, yp, zp])

            P = DLT(originali, slike)

            labelIspis = Label(frame2, text="Odgovarajuca matrica preslikavanja:")
            labelIspis.grid(row=brojTacaka+7)

            labelMatrica = Label(frame2, text=str(P))
            labelMatrica.grid(row=brojTacaka+8)

        
        buttonOK = Button(frame2, text="OK", command=izracunajDLT)
        buttonOK.grid(row=brojTacaka+6, column = 0)

        def poredjenjeSaNaivnim():

            originaliNaivni = []
            slikeNaivni = []

            for i in range(4):
                originaliNaivni.append(originali[i])
                slikeNaivni.append(slike[i])

            pNaivni = naivniAlgoritam(originaliNaivni, slikeNaivni)


            pDLT = DLT(originali, slike)

            pDLT = (pDLT / pDLT[0, 0]) * pNaivni[0,0]

            labelNaivni = Label(frame2, text="Matrica dobijena naivnim:")
            labelNaivni.grid(row=brojTacaka+9, column=0)

            labelMat1 = Label(frame2, text=str(pNaivni))
            labelMat1.grid(row=brojTacaka+10, column=0)

            labelDlt = Label(frame2, text="Matrica dobijena DLT-om:")
            labelDlt.grid(row=brojTacaka+9, column=1)

            labelMat2 = Label(frame2, text=str(pDLT))
            labelMat2.grid(row=brojTacaka+10, column=1)
            

        buttonPoredjenje = Button(frame2, text="Poredjenje sa naivnim", command=poredjenjeSaNaivnim)
        buttonPoredjenje.grid(row=brojTacaka+6, column=1)

    
    buttonOK1 = Button(frame2, text="OK", command=unetBrojTacaka)
    buttonOK1.grid(row=4)

def prikaziModifikovaniDLT():
    global frame3

    labelBrojTacaka = Label(frame3, text="Unesite broj tacaka:")
    labelBrojTacaka.grid(row=2)

    eBrojTacaka = Entry(frame3)
    eBrojTacaka.grid(row=3)

    originali = []
    slike = []

    def unetBrojTacaka():
        brojTacaka = int(eBrojTacaka.get().strip())

        originaliEntry = []
        slikeEntry = []

        labelUnesi = Label(frame3, text="Unesite homogene koordinate originalnih tacaka i njihovih slika:")
        labelUnesi.grid(row=5)

        for i in range(brojTacaka):
            eOriginal = Entry(frame3)
            originaliEntry.append(eOriginal)
            eOriginal.grid(row=i+6, column=0)
            

            eSlika = Entry(frame3)
            slikeEntry.append(eSlika)
            eSlika.grid(row=i+6, column=1)


        def izracunajDLT():
           

            for i in range(brojTacaka):
                x = float(originaliEntry[i].get().strip().split(' ')[0])
                y = float(originaliEntry[i].get().strip().split(' ')[1])
                z = float(originaliEntry[i].get().strip().split(' ')[2])

                originali.append([x, y, z])

                xp = float(slikeEntry[i].get().strip().split(' ')[0])
                yp = float(slikeEntry[i].get().strip().split(' ')[1])
                zp = float(slikeEntry[i].get().strip().split(' ')[2])

                slike.append([xp, yp, zp])

            P = dlt_normalize(originali, slike)

            labelIspis = Label(frame3, text="Odgovarajuca matrica preslikavanja:")
            labelIspis.grid(row=brojTacaka+7)

            labelMatrica = Label(frame3, text=str(P))
            labelMatrica.grid(row=brojTacaka+8)


        def uporediModifikovani():

            C1 = np.array([[0, 1, 2], [-1, 0, 3], [0, 0, 1]])
            C2 = np.array([[1, -1, 5], [1, 1, -2], [0, 0, 1]])
            
            novi_originali = []
            nove_slike = []

            for i in range(len(originali)):
                novi_originali.append(np.dot(C1, originali[i]))
                nove_slike.append(np.dot(C2, slike[i]))


            Pdlt = DLT(originali, slike)
            PdltPreslikano = DLT(novi_originali, nove_slike)

            Pkonacna = np.dot(np.linalg.inv(C2), PdltPreslikano)
            Pkonacna = np.dot(Pkonacna, C1)

            Pkonacna = (Pkonacna / Pkonacna[0, 0]) * Pdlt[0, 0]


            labelIspis1 = Label(frame3, text="Prva:")
            labelIspis1.grid(row=brojTacaka+9, column=0)

            labelMat1 = Label(frame3, text=str(Pdlt))
            labelMat1.grid(row=brojTacaka+10, column=0)

            labelIspis2 = Label(frame3, text="Druga:")
            labelIspis2.grid(row=brojTacaka+9, column=1)

            labelMat2 = Label(frame3, text=str(Pkonacna))
            labelMat2.grid(row=brojTacaka+10, column=1)

            Pndlt = dlt_normalize(originali, slike)
            PndltPreslikano = dlt_normalize(novi_originali, nove_slike)
            
            PndltKonacka = np.dot(np.linalg.inv(C2), PndltPreslikano)
            PndltKonacka = np.dot(PndltKonacka, C1)

            labelIspis3 = Label(frame3, text="Prva")
            labelIspis3.grid(row=brojTacaka+11, column=0)

            labelMat3 = Label(frame3, text=str(Pndlt))
            labelMat3.grid(row=brojTacaka+12, column=0)

            labelIspis4 = Label(frame3, text="Druga:")
            labelIspis4.grid(row=brojTacaka+11, column=1)

            labelMat4 = Label(frame3, text=str(PndltKonacka))
            labelMat4.grid(row=brojTacaka+12, column=1)

            

        
        buttonUporedi = Button(frame3, text="Poredjenje sa DLT-om", command=uporediModifikovani)
        buttonUporedi.grid(row=brojTacaka+6, column=1)

        buttonOK = Button(frame3, text="OK", command=izracunajDLT)
        buttonOK.grid(row=brojTacaka+6, column = 0)

    
    buttonOK1 = Button(frame3, text="OK", command=unetBrojTacaka)
    buttonOK1.grid(row=4)
    


def ucitaj():
    #odabir slike
    filename = filedialog.askopenfilename()

    global image 

    #otvaranje slike koja je odabrana
    image = PIL.Image.open(filename)
    #prikaz slike koja je otvorena
    image.show()

    #dimenzije ucitane slike
    global dimensions
    dimensions = image.size

    global frame4

    labelUnesi = Label(frame4, text="Unesite 4 koordinate na slici:")
    labelUnesi.grid(row=1)

    tackeE = []

    for i in range(4):
        e = Entry(frame4)
        tackeE.append(e)
        tackeE[i].grid(row=i+2)

    
    def ispraviSliku():
        tacke = []

        #odabrane tacke na slici
        for i in range(4):
            x = float(tackeE[i].get().strip().split(' ')[0])
            y = float(tackeE[i].get().strip().split(' ')[1])

            tacke.append([x, y, 1])

        #izracunavanje koordinata pravougaonika u cija temena treba da se preslikaju odabrane tacke na slici
        slike = []

        
        #p1 i p2 su pomocne promenljive koje predstavljaju duzinu ili sirinu odabranog cetvorougla
        #tj. trougla koji definisu 4 odabrane tacke na slici
        p1 = math.sqrt((tacke[0][0]-tacke[3][0])**2 + (tacke[0][1]-tacke[3][1])**2)
        p2 = math.sqrt((tacke[1][0] - tacke[2][0])**2 + (tacke[1][1] - tacke[2][1])**2)

        #sirina pravougaonika je aritmeticka sredina sirina originalnog cetvorougla
        sirina = round((p1+p2)/2)

        p1 = math.sqrt((tacke[0][0]-tacke[1][0])**2 + (tacke[0][1]+tacke[1][1])**2)
        p2 = math.sqrt((tacke[3][0]-tacke[2][0])**2 + (tacke[3][1] - tacke[2][1])**2)

        duzina = round((p1+p2)/2)

        s1 = dimensions[1] - sirina
        s1 = round(s1/2)

        s2 = dimensions[0] - duzina
        s2 = round(s2/2)

        slike.append([s2, s1, 1])
        slike.append([dimensions[0]-s2, s1, 1])
        slike.append([dimensions[0]-s2, dimensions[1]-s1, 1])
        slike.append([s2, dimensions[1]-s1, 1])

        #odgovarajuca matrica preslikavanja se dobija uz pomoc modifikovanog DLT algoritma
        P = dlt_normalize(tacke, slike)

        #racunanje inverza dobijenog preslikavanja
        P_inv = np.linalg.inv(P)

        #izdvajanje piksela ucitane slike
        old_pixels = image.load()

        #otvaranje nove slike koja je na pocetku crna
        image_new = PIL.Image.new("RGB", dimensions, "#000000")
        new_pixels = image_new.load()

        #prolazenje kroz sve piksele 
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                #na svaki piksel (i, j) se racuna piksel koji se sa originalne slike preslikava u njega
                x_coor, y_coor, z_coor = np.dot(P_inv, [i, j, 1])

                x = round(x_coor/z_coor)
                y = round(y_coor/z_coor)
                
                #ako koordinate tog piksela "iskacu" van originalne slike, piksel (i, j) ostaje crn
                #inace se piksel (i, j) postavlja na vrednost dobijenog piksela (x, y)
                if (x < 0 or x >= dimensions[0]):
                    continue
                elif (y < 0 or y >= dimensions[1]):
                    continue
                else:
                    new_pixels[i, j] = old_pixels[x, y]
                    

        #cuvanje i prikazivanje slike sa ispravljenom projektivnom distorzijom
        image_new.save("nova.bmp")
        image_new.show()

    buttonOk = Button(frame4, text="OK", command=ispraviSliku)
    buttonOk.grid(row=6)

buttonNaivni = Button(frame1, text="a) Naivni", command=prikaziNaivni)
buttonNaivni.grid(row=0)

buttonDLT = Button(frame2, text="b) DLT", command=prikaziDLT)
buttonDLT.grid(row=0)

buttonModifikovaniDlt = Button(frame3, text="c) Modifikovani DLT" ,command=prikaziModifikovaniDLT)
buttonModifikovaniDlt.grid(row=0)

buttonUcitaj = Button(frame4, text="d) Ucitaj sliku", command=ucitaj)
buttonUcitaj.grid(row=0)

mainloop()