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

with open('de_pure.txt', 'r') as file:
    data = file.read()

theSet = set(data.split('\n\n'))

print(len(theSet))

file1 = open("sentences.txt", "w")

for de in theSet:
    de = de.replace("\n", ";")
    file1.write(de + "\t1\n")
file1.close()

#print(theSet)