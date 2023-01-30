
Inputfiles = ["res_oppnet_varyttl_qiaobeidata_20230120022551.csv",
              "res_oppnet_varyttl_qiaobeidata_20230120145701.csv"]

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
