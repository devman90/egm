*************
c1 1 0 CSTRAP
r1 2 1 RSTRAP
l1 2 0 LSTRAP
c2 3 2 CBODY
r2 4 3 RDELAY
c3 4 3 CDELAY
vin 5 4 pwl(0 0 0 4kv 150ns 4kv)
l2 5 6 0.14e-6
r3 6 0 2

.save @r3[i]

.tran 50ps 150ns

.control
run
*plot @r3[i]
set filetype = ascii
write 'OUTPUT'
quit
.endc
.end
