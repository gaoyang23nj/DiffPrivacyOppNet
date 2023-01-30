
Inputfiles = ["res_oppnet_varyeps_Pinter_qiaobeidata_20230121093336.csv",
              "res_oppnet_varyeps_Pinter_qiaobeidata_20230122082001.csv",
              "res_oppnet_varyeps_Pinter_qiaobeidata_20230123092811.csv",
              "res_oppnet_varyeps_Pinter_qiaobeidata_20230127195502.csv"]

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
