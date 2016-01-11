import warnings
warnings.filterwarnings("ignore")

from openmdao.api import Component, Group, Problem, IndepVarComp

import numpy as np
import datetime
import os

fpath = os.path.dirname(os.path.realpath(__file__))

from solar import *


top = Problem()
top.root = Group()

n = 10

# top.root.add("batt", Batteries(n), promotes=["*"])
# top.root.batt.fd_options['form'] = 'complex_step'
# dvar = (
#     ('P_generated', np.random.randn(n)**2, {'units' : 'W'}),
#     ('array_power', 100, {'units' : 'W'}),
#     ('P_consumption', np.random.randn(n)**2, {'units' : 'W'}),
#     ('power_capacity', 50.0, {'units' : 'W*h'}),
# )

top.root.add("batt", BatteryConstraints(n), promotes=["*"])
top.root.batt.fd_options['form'] = 'complex_step'
dvar = (
    ('SOC', np.random.rand(n)),
)

# top.root.add("loads", Loads(n), promotes=["*"])
# dvar = (
#     ('P_generated', np.random.randn(n) * 4, {'units' : 'W'}),
#     ('ambient_temperatures', np.random.randn(n)**2, {'units' : 'W'}),
#     ('irradiance', np.random.randn(n)*20, {'units' : 'W/m**2'}),
#     ('array_power', 1000.0, {'units' : 'W'}),
#     ('P_constant', 4.0, {'units' : 'W'}),
#     ('P_direct', 3.0, {'units' : 'W'}),
#     ('P_daytime', 2.0, {'units' : 'W'}),
#     ('P_nighttime', 1.0, {'units' : 'W'}),
#     ('power_capacity', 50.0, {'units' : 'W*h'}),
# )


top.root.add("dvar", IndepVarComp(dvar), promotes=["*"])

top.setup()

top.run()

top.check_partial_derivatives()

