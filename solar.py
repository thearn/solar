import warnings
warnings.filterwarnings("ignore")

from openmdao.api import Component, Group, Problem, IndepVarComp

import numpy as np
import datetime
import os
from ks import KSfunction

fpath = os.path.dirname(os.path.realpath(__file__))

class PowerSource(Component):
    """Parses NREL data and provides associated transient outputs"""

    def __init__(self, fns=None):
        super(PowerSource, self).__init__()

        # length of time series
        self.n = 8760

        # create array of corresponding dates (for plotting)
        next_year = datetime.datetime.now().year + 1
        start = datetime.datetime(next_year, 1, 1)
        h = datetime.timedelta(hours=1)
        self.dates = np.array([start + i*h for i in range(self.n)])

        # load and scale NREL data from a 4 kW to 1 W system 
        # (near linear response in generated DC power)
        self.data = np.load(fpath + "/data/example.npy")
        self.data[:,-2:] = self.data[:,-2:] / 4000.0

        self.add_param("array_power", 1.0, units="W")

        # Variables that will be outputted
        self.add_output("ambient_temperature", np.zeros(self.n), units="degC")
        self.add_output("hour", np.zeros(self.n), units="h")
        self.add_output("day", np.zeros(self.n), units="d")
        self.add_output("month", np.zeros(self.n), units="mo")
        self.add_output("P_generated", np.zeros(self.n), units="W")
        self.add_output("wind", np.zeros(self.n), units="m/s")
        self.add_output("irradiance", np.zeros(self.n), units="W/m**2")

    def solve_nonlinear(self, p, u, r):
        # scale collectable DC power to actual system size
        u['P_generated'] = self.data[:, -2] * p['array_power']

        # parse and output other data values directly
        u['month'] = self.data[:,0]
        u['day'] = self.data[:,1]
        u['hour'] = self.data[:,2]
        u['ambient_temperature'] = self.data[:,5]
        u['wind'] = self.data[:,6]
        u['irradiance'] = self.data[:,4]

    def linearize(self, p, u, r):
        J = {}
        J['P_generated', 'array_power'] = self.data[:, -2]
        return J

class Loads(Component):
    """A very basic PV solar load component. Has constant power draws, 
    and direct loading."""

    def __init__(self, n):
        super(Loads, self).__init__()
        self.n = n

        self.add_param("P_constant", 0.0, units="W")
        self.add_param("P_direct", 0.0, units="W")
        self.add_param("P_daytime", 0.0, units="W")
        self.add_param("P_nighttime", 0.0, units="W") 

        self.add_param("P_generated", np.zeros(self.n), units="W")
        self.add_param("ambient_temperature", np.zeros(self.n), units="degF")
        self.add_param("hour", np.zeros(self.n), units="h")
        self.add_param("irradiance", np.zeros(self.n), units="W/m**2")
        self.add_param("wind", np.zeros(self.n), units="mi/h")
        
        self.add_output("P_consumption", np.zeros(self.n), units="W")

    def solve_nonlinear(self, p, u, r):
        self.J = {}
        # constant background consumption
        u['P_consumption'] = np.zeros(self.n)
        u['P_consumption'] += p['P_constant']
        
        self.J['P_consumption', 'P_constant'] = np.ones(self.n)

        # daytime - based on PV
        idx = np.where(p['P_generated'] >= 0.01)
        u['P_consumption'][idx] += p['P_daytime']
        
        self.J['P_consumption', 'P_daytime'] = np.zeros(self.n)
        self.J['P_consumption', 'P_daytime'][idx] = 1.0

        # nightime - based on irradiance
        idx = np.where(p['irradiance'] < 10.0)
        u['P_consumption'][idx] += p['P_nighttime']
        
        self.J['P_consumption', 'P_nighttime'] = np.zeros(self.n)
        self.J['P_consumption', 'P_nighttime'][idx] = 1.0


        # direct load - based on available PV power
        idx = np.where((p['P_generated'] >= p['P_direct']))
        u['P_consumption'][idx] += p['P_direct']
        
        self.J['P_consumption', 'P_direct'] = np.zeros(self.n)
        self.J['P_consumption', 'P_direct'][idx] = 1.0

    def linearize(self, p, u, r):
        return self.J

class Batteries(Component):
    """Battery model, computes state of charge (SOC) over time"""
    
    def __init__(self, n):
        super(Batteries, self).__init__()
        self.n = n

        # inputs: battery power capacity, and PV generated power and load
        # consumptions over time
        self.add_param("power_capacity", 0.0, units="W*h")
        self.add_param("P_generated", np.zeros(self.n), units="W")
        self.add_param("P_consumption", np.zeros(self.n), units="W")

        # output: resulting state of charge
        self.add_output("SOC", np.ones(self.n), units="unitless")

        if not os.path.exists(fpath + "/data/ds_dp.npy"):
            # takes awhile...
            self.ds_dp = np.zeros((n, n))
            for i in xrange(n):
                for k in range(i+1): 
                    k_ = min([k, 1023])
                    self.ds_dp[i,i-k -n+1] = 1. - 1./2.0**(k_)
            self.ds_dp[:,0] = 1.0
            np.save(fpath + "/data/ds_dp", self.ds_dp)
        else:
            self.ds_dp = np.load(fpath + "/data/ds_dp.npy")


    def solve_nonlinear(self, p, u, r):
        u['SOC'] = np.ones(self.n)

        self.J = {}

        self.J['SOC', 'power_capacity'] = np.zeros((self.n, 1))
        self.J['SOC', 'P_generated'] = np.zeros((self.n, self.n))
        self.J['SOC', 'P_consumption'] = np.zeros((self.n, self.n))

        # initial state of charge at beginning of time series: assume 100%
        SOC = 1.0

        # Integrate SOC for each time point (hour-by-hour)
        for i in range(self.n):
            old_SOC = SOC

            # available energy in battery [Wh]
            available = SOC * p['power_capacity']
            # PV energy collectable during this hour [Wh]
            generated = p['P_generated'][i] # * 1 hour
            # Energy consumed by loads during this hour [Wh]
            consumed = p['P_consumption'][i] # * 1 hour

            # Power balance [Wh]
            diff = available + generated - consumed

            # Base SOC calculation: Wh / Wh -> percentage of rated power capacity
            SOC = (diff) / p['power_capacity'] 

            # Bound between 0 and 100 %
            if SOC > 1.0:
                SOC = 1.0
            elif SOC < 0:
                SOC = 0.0

            # trapezoid rule: integral(W * dt, t=hour_i..hour_i+1) = 1 * (W_i+1 - W_i)/2 [Wh]
            u['SOC'][i] = (SOC + u['SOC'][i-1])/2.0 

            self.J['SOC', 'power_capacity'][i] = (1 - u['SOC'][i]) / p['power_capacity']


        self.J['SOC', 'P_generated'] = self.ds_dp/p['power_capacity']
        self.J['SOC', 'P_consumption'] = -self.ds_dp/p['power_capacity']


    def linearize(self, p, u, r):
        return self.J

class Costs(Component):
    """Basic cost model"""

    def __init__(self):
        super(Costs, self).__init__()
        # inputs
        self.add_param("power_capacity", 50.0, units="W*h")
        self.add_param("array_power", 100.0, units="W")

        # output
        self.add_output("cost", 0.0)

    def solve_nonlinear(self, p, u, r):
        # cost estimate is $1.33 per panel watt, and $0.20 per battery Wh
        u['cost'] = 1.33 * p['array_power'] + 0.2 * p['power_capacity']
        #print u['cost'], p['array_power'] ,p['power_capacity']

    def linearize(self, p, u, r):
        J = {}

        J['cost', 'power_capacity'] = 0.2

        J['cost', 'array_power'] = 1.33

        return J


class BatteryConstraints(Component):

    def __init__(self, n):
        super(BatteryConstraints, self).__init__()
        self.n = n

        self.add_param("SOC", np.ones(self.n), units="unitless")
        self.add_output("Cons1", 1.0)

        self.KS = KSfunction()
        self.rho = 50
        self.SOC_min = 0.8

    def solve_nonlinear(self, params, unknowns, resids):
        SOC = params['SOC']
        #print SOC.min()
        unknowns['Cons1'] = self.KS.compute(self.SOC_min - SOC, self.rho)
    def linearize(self, p, u, r):
        J = {}

        J['Cons1', 'SOC'] = - self.KS.derivatives()[0].reshape((1, self.n))

        return J

class Basic(Group):
    """
    Simple solar PV model. Collects all components, and establishes data 
    relationships.
    """
    def __init__(self, fns=None):
        super(Basic, self).__init__()
        
        # most likely top-level params for design
        params = (
            ('array_power', 100.0, {'units' : 'W'}),
            ('power_capacity', 50.0, {'units' : 'W*h'}),
        )
        self.add('des_vars', IndepVarComp(params), promotes=["*"])

        # add NREL data parsing component
        self.add("data", PowerSource(fns=fns), promotes=["*"])
        n = self.data.n

        # Load component
        self.add("loads", Loads(n), promotes=["*"]) 

        # Battery component
        self.add("batteries", Batteries(n), promotes=["*"])

        self.add("costs", Costs(), promotes=["*"])
        self.add("batt_con", BatteryConstraints(n), promotes=["*"])


if __name__ == "__main__":
    import pylab
    from openmdao.api import ScipyOptimizer

    top = Problem()
    top.root = Basic()
    
    # top.driver = ScipyOptimizer()
    # top.driver.options['optimizer'] = 'SLSQP'

    # top.driver.add_desvar('array_power', lower=1, upper=500)
    # top.driver.add_desvar('power_capacity', lower=1, upper=2000)
    # top.driver.add_objective('cost')
    # top.driver.add_constraint('Cons1', upper=0.0)
    # top.driver.iprint = 1

    top.setup(check=True)

    # load specification
    top['P_constant'] = 3.0
    top['P_daytime'] = 0.0
    top['P_nighttime'] = 4.0
    top['P_direct'] = 25.0

    # design variables
    top['array_power'] = 200 # Watts
    top['power_capacity'] = 900 # Watt-hours

    # print order
    print [s.name for s in top.root.subsystems(local=True)]
    print 

    # connections 
    top.root.list_connections()
    top.run()

    print 
    print "dSOC/darray_power"
    v = top.calc_gradient(['SOC'],['array_power'])
    print v
    print v.max(), v.min()

    print "dConstr/darray_power"
    v = top.calc_gradient(['Cons1'],['array_power'])
    print v
    print v.max(), v.min()

    print "dcost/darray_power"
    v = top.calc_gradient(['cost'],['array_power'])
    print v
    print v.max(), v.min()

    print "dSOC/dpower_capacity"
    v = top.calc_gradient(['SOC'],['power_capacity'])
    print v
    print v.max(), v.min()

    print "dConstr/dpower_capacity"
    v = top.calc_gradient(['Cons1'],['power_capacity'])
    print v
    print v.max(), v.min()

    print "dcost/dpower_capacity"
    v = top.calc_gradient(['cost'],['power_capacity'])
    print v
    print v.max(), v.min()

    dates = top.root.data.dates
    soc = top['SOC']
    generated = top['P_generated']
    used = top['P_consumption']


    pylab.plot(dates, soc)
    pylab.show()
