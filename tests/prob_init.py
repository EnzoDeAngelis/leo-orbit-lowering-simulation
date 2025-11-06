import math

R = 6371     # Raggio terrestre
mu = 398600  # Mu Terra

# Quota di riferimento
h = 499.7

# Raggio adimensionale
r = (h+R)/R

v = math.sqrt(mu/(r*R))  # Velocità dimensionale alla quota h
vc1 = math.sqrt(mu/R)    # Valore per adimensionalizzare la velocità
v_nd = v/vc1             # Velocità adimensionale alla quota h

print(f"Il raggio adimensionale vale r = {r}")
print(f"La velocità adimensionale alla quota h = {h} km da inserire in prob.yaml è v = {v_nd}")

# Da h = 500 km a 499.7
# Guess : -0.05 -0.1 0.0 0.3 -0.3 0.0 0.997 2.30 3.80 5.0
# New g : -3.2176e-01 -1.4947e-02 0.0 3.2028e-02 -3.9607e-01 0.0 9.9954e-01 1.9795e+00 3.3740e+00 5.4075e+00 (1.3352e+02, 1.3352e+02)

### I guess non cambiano perchè ottengo la convergenza in 1 iterazione
# Da h = 499.7 a 499.4
# Iter : 2
# New g : -3.2176e-01 -1.4947e-02 0.0 3.2028e-02 -3.9607e-01 0.0 9.9954e-01 1.9795e+00 3.3740e+00 5.4075e+00 (1.3352e+02, 1.3352e+02)

### Converge in una sola iterazione, continuo ad abbassare la quota per vedere fin dove questi guess siano validi
# Da h = 499.4 a 499.1
# Iter : 1
# Guess : -3.2176e-01 -1.4947e-02 0.0 3.2028e-02 -3.9607e-01 0.0 9.9954e-01 1.9795e+00 3.3740e+00 5.4075e+00

### Uso i soliti guess ma non per t1,t2,tf che adatto alla SF. Ottengo convergenza e nuovi costati
# Da h = 499.1 a 498.8
# Iter : 3161
# Guess : -3.2176e-01 -1.4947e-02 0.0 3.2028e-02 -3.9607e-01 0.0 9.9954e-01 1.9495e+00 3.3840e+00 5.5875e+00
# New g : -3.5303e-01 -7.1372e-05  0.0000e+00  1.6356e-04 -3.9549e-01  0.0000e+00  9.9999e-01  2.0579e+00  3.5059e+00  5.5768e+00 ( 6.3817e-02, 6.3817e-02

### Riesco a convergere con i guess iniziali delle iterazioni precedenti ma non con i new guess dell'iterazione precedente nonostante parta
### da un errore di 1e-02 contro i 1e+02 dell'altra
# Da h = 498.8 a 498.5
# Iter : 3461
# Guess : -3.2176e-01 -1.4947e-02 0.0 3.2028e-02 -3.9607e-01 0.0 9.9954e-01 1.9495e+00 3.3840e+00 5.5875e+00
# New g : -3.5076e-01 -1.1593e-03  0.0000e+00  2.6570e-03 -3.9562e-01  0.0000e+00  9.9999e-01  2.0535e+00  3.5024e+00  5.5787e+00 ( 1.0358e+01, 1.0358e+01)

### Ottengo una buona convergenza con i new guess dell'iterazione precedente partendo da 1e+01 e cambiando di poco t1 e t2
# Da h = 498.5 a 498.2
# Iter : 1461
# Guess : -3.5076e-01 -1.1593e-03  0.0000e+00  2.6570e-03 -3.9562e-01  0.0000e+00  9.9999e-01  2.0235e+00  3.5524e+00  5.5787e+00
# New g: -3.5166e-01 -7.3699e-04  0.0000e+00  1.6911e-03 -3.9557e-01  0.0000e+00  9.9999e-01  2.0356e+00  3.5359e+00  5.5731e+00 ( 6.5841e+00, 6.5841e+00)

### Stessa tecnica di prima, continuo a convergere
# Da h = 498.2 a 497.9
# Iter : 2524
# Guess : -3.5166e-01 -7.3699e-04  0.0000e+00  1.6911e-03 -3.9557e-01  0.0000e+00  9.9999e-01  1.9956e+00  3.5959e+00  5.5731e+00
# New g : -3.5269e-01 -2.6341e-04  0.0000e+00  6.0320e-04 -3.9551e-01  0.0000e+00  9.9999e-01  2.0354e+00  3.5366e+00  5.5744e+00 ( 2.3540e+00, 2.3540e+00)

### Ho provato a modicare t1 e t2 per adattare la SF ma non ha funzionato. Se uso i new guess invariati converge     
# Da h = 497.9 a 497.6
# Iter : 725
# Guess : -3.5269e-01 -2.6341e-04  0.0000e+00  6.0320e-04 -3.9551e-01  0.0000e+00  9.9999e-01  2.0354e+00  3.5366e+00  5.5744e+00
# New g : -3.5271e-01 -2.5479e-04  0.0000e+00  5.8397e-04 -3.9551e-01  0.0000e+00  9.9999e-01  2.0310e+00  3.5292e+00  5.5702e+00 ( 2.2760e+00, 2.2760e+00)

### Con i new g converge in 1 iterazione. Se provo a modificare t1 e t2 non converge
# Da h = 497.6 a 497.3
# Iter : 1
# Guess : -3.5271e-01 -2.5479e-04  0.0000e+00  5.8397e-04 -3.9551e-01  0.0000e+00  9.9999e-01  2.0310e+00  3.5292e+00  5.5702e+00
# New g : //                  //

### Converge in 1 iterazione
# Da h = 497.3 a 497
# Iter : 1
# Guess : 
# New g : -3.5271e-01 -2.5479e-04  0.0000e+00  5.8397e-04 -3.9551e-01  0.0000e+00  9.9999e-01  2.0300e+00  3.5200e+00  5.5700e+00 ( 2.2763e+00, 2.2763e+00)
 
### Converge in 1 iterazione
# Da h = 497 a 496.7
# Iter : 1
# Guess : 

### Uso i primissimi guess e converge. Con i new guess precedenti converge in 1 iterazione
# Da h = 496.7 a 496.4
# Iter : 2821
# Guess : -0.05 -0.1 0.0 0.3 -0.3 0.0 0.997 2.30 3.80 5.0
# New g : -3.1456e-01 -1.8751e-02  0.0000e+00  4.2613e-02 -3.9857e-01  0.0000e+00  9.9994e-01  2.0805e+00  3.4675e+00  5.3755e+00 ( 1.6762e+02, 1.6762e+02)

### Ottimo risultato! Uso i new guess e converge in maniera relativamente veloce
# Da 496.4 a 496.1   
# Iter : 1084
# Guess : -3.1456e-01 -1.8751e-02  0.0000e+00  4.2613e-02 -3.9857e-01  0.0000e+00  9.9994e-01  2.0805e+00  3.4675e+00  5.3755e+00
# New g : -3.1686e-01 -1.7686e-02  0.0000e+00  4.0130e-02 -3.9852e-01  0.0000e+00  9.9994e-01  2.0782e+00  3.4609e+00  5.3801e+00 ( 1.5804e+02, 1.5804e+02)

### Converge in 1 iterazione con i new guess precedenti. Se invece uso i guess usati nello step precedente ottengo questa convergenza.
# Da 496.1 a 495.8
# Iter : 1076
# Guess : -3.1456e-01 -1.8751e-02  0.0000e+00  4.2613e-02 -3.9857e-01  0.0000e+00  9.9994e-01  2.0805e+00  3.4675e+00  5.3755e+00
# New g : -3.1670e-01 -1.7761e-02  0.0000e+00  4.0303e-02 -3.9853e-01  0.0000e+00  9.9994e-01  2.0783e+00  3.4608e+00  5.3794e+00 ( 1.5873e+02, 1.5873e+02)










###### Convergenze varie di lowering a diverse quote, non servono per il caso studio
# Da h = 400 km a 399.7
# Iter : 5882
# Guess : -3.2176e-01 -1.4947e-02 0.0 3.2028e-02 -3.9607e-01 0.0 9.9954e-01 1.9795e+00 3.3740e+00 5.4075e+00
# New g : -3.6092e-01 -1.8624e-05  0.0000e+00  4.4022e-05 -3.9549e-01  0.0000e+00  9.9999e-01  2.1002e+00  3.4276e+00  5.5449e+00 ( 1.7041e-02, 1.7041e-02)

# Da h = 491 km a 490.7
# Iter = 4861
# Guess : -3.2176e-01 -1.4947e-02  0.0000e+00 3.2028e-02 -3.9607e-01  0.0000e+00  9.9954e-01  1.8700e+00  3.4500e+00  5.57e+00 (1.3352e+02, 1.3352e+02)
# New g : -3.5368e-01 -6.0726e-05  0.0000e+00 1.4037e-04 -3.9549e-01  0.0000e+00  9.9999e-01  2.0612e+00  3.4998e+00  5.5743e+00 (5.4408e-02, 5.4408e-02)




############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################


# Convergenze ottenute con la nuova gui

# Ottengo convergenza utilizzando i new guess della primissima convergenza con la vecchia gui. Per ottenerla parto a tempo libero ma dopo poche iterazioni
# devo bloccarlo. C'è sempre lo stesso problema di Hf troppo alto, che ora spesso di risolvere bloccando il tempo.
# Da h = 500 a 499.7 
# Iter : 4676
# Guess : -3.2176e-01 -1.4947e-02 0.0 3.2028e-02 -3.9607e-01 0.0 9.9954e-01 1.9795e+00 3.3740e+00 5.4075e+00
# New g: -6.2523e-01 -9.9423e-07  0.0000e+00 -8.3881e-02 -7.9387e-01  0.0000e+00  9.9999e-01  2.0998e+00  3.2974e+00  5.4004e+00 ( 9.9492e-07, 9.9423e-07)