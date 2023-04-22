filename = 'log_console.txt'
new_filename = 'new_log_console.txt'
f = open(filename,'r')
new_f = open(new_filename,'w+')
lines = f.readlines()
for line in lines:
    if 'delivered' in line:
        new_f.write(line)
new_f.close()
f.close()

filename = 'Console_Output.txt'
new_filename = 'new_Console_Output.txt'
f = open(filename,'r')
new_f = open(new_filename,'w+')
lines = f.readlines()
for line in lines:
    if 'delivered' in line:
        new_f.write(line)
new_f.close()
f.close()
