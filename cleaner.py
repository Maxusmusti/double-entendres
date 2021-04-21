import re

"""
Main script for reformatting sentences.txt to work with neural network/reformatting TWSS tweets
"""

#my_set = set(open('double_entendres.txt'))

"""
with open('double_entendres.txt', 'r') as file:
    data = file.read()

theSet = set(data.split('\n\n'))

file1 = open("new.txt", "w")

for de in theSet:
    file1.write(de + "\n\n")
file1.close()
"""

with open('normal.txt', 'r') as file:
    data = file.read()

data = re.sub('\[.*\]\n', '', data)

theSet = set(data.split('\n\n'))

print(len(theSet))

file1 = open("sentences_2.txt", "w")

for de in theSet:
    de = de.replace("\n", ";")
    file1.write(de + "\t0\n")
file1.close()

#print(theSet)
