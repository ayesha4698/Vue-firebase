
import random
# or2147483647
#   10000000
# -2k-1 to 2k-1-1


def gen32Bit():
    a = random.randint(-10000000, 10000000)
    hash = random.getrandbits(32)
    return bin(hash)

def opOR(a, b):
    return a | b

def opAnd(a, b, opcode, sa):
    return a & b



def testMain ():
    f = open("myfile.txt", "w")





hash = random.getrandbits(32)
a = gen32Bit()
print(bin(hash))


