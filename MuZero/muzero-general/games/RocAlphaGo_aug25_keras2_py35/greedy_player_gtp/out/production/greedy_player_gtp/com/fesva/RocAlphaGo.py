#
#import call_python_version from call_python

import sys
sys.path.insert(0, '../../../../../') # /greedy_player_gtp')
from greedy_player_gtp import *
from call_python import  *
# Run Py2py3
call_python_version('3.7', greedy_player_gtp, '*',  '') # Module, Function, ArgumentList):
# Run RocAlphaGo