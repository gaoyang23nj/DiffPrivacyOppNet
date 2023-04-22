# Program arguments in the ONE for this simulation
-b 1 new_settings.txt

# 4 classes (files) are created to support the one-way transmission scenario of the BSS and the routing method TTPM.
OneWayConnectionEvent.java,

OneToEachMessageGenerator.java,

OneFromEachMessageGenerator.java

TTPMSpdUpDjkRouter.java

# File Location
The parameters (P and {p}) of TTPM are located in "\01-code\Main\PandpforOne".
(Need to conduct 'MainSimulator_Params_Pandp.py' in the '01-code' folder to collect these parameters beforehand)
TTPMSpdUpDjkRouter.java relies on these parameters.

The contact events are in "\05-the-one-master\the-one-master\NanjingData\data_for_ONE.txt"

The message generation events are in "\05-the-one-master\the-one-master\NanjingData\MSG_XXX.txt", e.g., MsgGen_3600 (1 message per hour).

