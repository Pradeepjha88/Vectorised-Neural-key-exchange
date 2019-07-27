from NeuralMachine import Machine
import numpy as np
import time
import sys
import hmac
import pickle
from StringIO import StringIO
import matplotlib.pyplot as plt
import hashlib

k = int(raw_input("Enter the number of hidden neurons: "))
n = int(raw_input("Enter the number of Input neurons: "))
l = int(raw_input("Enter the Range of Weight: "))

update_rules = ['hebbian', 'anti_hebbian', 'random_walk']
ruleNum = int(raw_input("The rule used: \n\t0 - Hebbian \n\t1 - Anti Hebbian \n\t2 - Random Walk\n"))
update_rule = update_rules[ruleNum if (0 <= ruleNum <= 2) else 0]

print("Creating machines : k=" + str(k) + ", n=" + str(n) + ", l=" + str(n))
print("Using " + update_rule + " update rule.")
Alice = Machine(k, n, l)
Bob = Machine(k, n, l)
Eve = Machine(k, n, l)

def random():
    return np.random.randint(-l, l + 1, [k, n])

def sync_score(m1, m2):
    return 1.0 - np.average(1.0 * np.abs(m1.W - m2.W)/(2 * l))


sync = False
nb_updates = 0
nb_updates_eve = 0
start_time = time.time()
sync_history = []
serial = False
while(not sync):
    X = random()
    tauA = Alice(X)
    tauB = Bob(X)
    tauE = Eve(X)
    Alice.W = Alice.update(tauB, update_rule,serial)
    Bob.W = Bob.update(tauA, update_rule, serial)
    nb_updates += 1
    score = 100 * sync_score(Alice, Bob)
    sync_history.append(score)
    if(tauA == tauB == tauE):
        Eve.update(tauA,update_rule,serial)
        nb_updates_eve+=1
    print('\r' + "Synchronization = " + str(int(score)) + "%   /  Updates = " + str(nb_updates))
    if score == 100:
        sync = True

if(sync_score(Eve,Alice)>=1):
    print("Eve has found the Synchronised Key, Abort transmission.")
    sys.exit(0)



end_time = time.time()
time_taken = end_time - start_time
ABC = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_0123456789'
AliceKey=''
key_size =int(len(ABC)/(Alice.l*2+1))
key_length = Alice.k*Alice.n /key_size
for i in range(1,key_length):
    k=1
    for j in range(((i-1)*key_size),(i*key_size-1)):
        k=k+Alice.W.flatten()[j]+Alice.l
    AliceKey = AliceKey + ABC[k]



print ('\nMachines have been synchronized.')
print ('Time taken = ' + str(time_taken)+ " seconds.")
print ('Updates = ' + str(nb_updates) + ".")
print("--------------------------------------------------------------------------------")
print ('Output Key =' + str(AliceKey))
print("--------------------------------------------------------------------------------")

plt.plot(sync_history)
plt.show()

def Encrypt(message):
    return hmac.new(str(AliceKey), message, hashlib.sha512).hexdigest()


class SimpleObject(object):
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return self.name

out_s = StringIO()
message = raw_input("Enter the Message to be Encrypted: ")
o = SimpleObject(message)
pickled_data = pickle.dumps(o)
digest = Encrypt(pickled_data)
header = '%s %s' % (digest, len(pickled_data))
print '\nWRITING:', header
out_s.write(header + '\n')
out_s.write(pickled_data)

in_s = StringIO(out_s.getvalue())
incoming_digest = ''
incoming_length = 0
incoming_pickled_data = ''
#read Data
while True:
    first_line = in_s.readline()
    if not first_line:
        break
    incoming_digest, incoming_length = first_line.split(' ')
    incoming_length = int(incoming_length)
    print '\nREAD:', incoming_digest, incoming_length
    incoming_pickled_data = in_s.read(incoming_length)

actual_digest = Encrypt(incoming_pickled_data)
print 'ACTUAL:', actual_digest

if incoming_digest != actual_digest:
    print 'WARNING: Data corruption'
else:
    obj = pickle.loads(incoming_pickled_data)
    print 'OK:', obj