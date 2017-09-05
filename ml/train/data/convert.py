import sys

for line in open(sys.argv[1]):
    tokens = line.strip().split(' ')
    label = '0' if int(tokens[0]) < 0.5 else '1'
    print label + ' ' + ' '.join(tokens[1:])   
