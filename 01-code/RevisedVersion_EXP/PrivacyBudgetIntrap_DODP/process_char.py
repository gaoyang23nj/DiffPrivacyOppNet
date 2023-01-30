import os

Inputfiles = []
filenames = os.listdir('./')
for filename in filenames:
    suffix = filename.split('.')[-1]
    label = filename.split('.')[0].split('_')[-1]
    print(filename, suffix, label)
    if suffix == 'csv' and label != 'new':
        Inputfiles.append(filename)

print('*'*50)
print(Inputfiles)

for filename in Inputfiles:
    outputfilename = filename.split('.')[0] + '_new.' + filename.split('.')[1]
    f = open(filename, 'r+')
    fo = open(outputfilename, 'w+')
    lines = f.readlines()
    label = False
    for line in lines:
        tunples = line.split(',')
        if 'no_noise' in tunples:
            label = True
            newline = line.replace('no_noise', '-1')
            fo.write(newline)
            print(tunples)
        else:
            fo.write(line)
    f.close()
    fo.close()
