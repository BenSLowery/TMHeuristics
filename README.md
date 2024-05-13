# Inventory Management system 
An inventory management Discrete Event System, and heuristics, for a retailer operating under online and offline demand channels, many stores, and partial lost-sales.


## Structure
* Discrete Event System
  * DiscreteEventSystem.py - A discrete event system with an optional order cap. Assumes a base-stock policy, possibility of positive lead time, and a logger in Pandas to keep a track of events. 
  * DiscreteEventSystemSpeedy.py - A quicker version of above. Good for evaluating policies and running a policy thousands of times for evaluation.
  * DiscreteEventSystemZLT.py - A discrete event system assuming zero replenishement lead time. Allows for for flexible ordering policies (although a base-stock/OUT is optimal).
* Heuristics
  * ZeroLeadTime
    * .
  * PositiveLeadTime
    * .

## Discrete Event System
Note DiscreteEventSystem and DiscreteEventSystemSpeedy work in the same way except in the speedy version, the `log` variable does not exist and you have to pre-specify the desired online demand allocation. 

The whereabouts of the inventory at any point is stores in a pipeline vector. For example, if a store has a lead time of `3`, it's pipeline vector may look like `[10,2,3,0]`. Here, we are saying there is 10 units currently at the store, `2` due to arrive in one period, and `3` due to arrive in two periods. The final element is what will be ordered in this period, and will be replaced by an order quantity once a decision has been made.

Ordering is done as a Echelon base-stock policy. Every period, we aim to make the sum of all items in the vector equal to a base-stock number. For example, using the pipeline vector `[10,2,3,0]`, if we want a base-stock level at the store of 18, we need to order 3 units, so the pipeline vector becomes `[10,2,3,3]`.

Below is an example of how to run the simulation and the log.

```python
# Import packages
import numpy as np
import DiscreteEventSystem.DiscreteEventSystem as DES
import scipy.stats as sp

np.random.seed(42)

### PARAMETERS ###
T=36 # Length of simulation
n=5 # Number of stores
lt_w = 1 # Warehouse Lead time
lt_s = 1 # Store Lead time
co = [1,2] # Holding cost, first argument warehouse, second store. For now assumes same across all stores
cu = [18,18] # Shortage penalty cost, same across stores and warehouse
sim_len = 2 # Run two simulations, one to test the reset function, and another to print the log

### Set Echelon Base Stock Levels ###
wh_local = 57 # Local Warehouse
st = 22
wh_ech = wh_local + st*n # Echelon level 
cap = 10 # order cap



# Generate demand, assumes 5 stores, no demand from online channel, and a demand of Poisson(10) from the offine channel for two simulation worth
demand = [[[np.random.poisson(int(i))] + [np.random.poisson(int(j)) for k in range(n)] for i,j in np.array([[0,10] for s in range(T)])] for i in range(sim_len)]

# Create a new simulation instance
test = DES.DES(T, n, lt_s, lt_w, 10000, 10000, 10000, 0.8, 0, co[0], co[1], cu[0], cu[1],  wh_local, st)
  
  
test.set_order_q([wh_ech for s in range(T)], [[st for s in range(T)] for k in range(n)], [[cap for s in range(36)] for k in range(n)])

# Run first instance
test.run(demand[0])
test.reset()
# Run second instance and record the log
test.run(demand[1])

# Print the log
test.log
```

#### Understanding the Log
The log comes with the following headers and is outputted as a pandas DataFrame:
* **Period**: The time index of the simulation
* **StartingInvWarehouse**: Pipeline Vector of the starting inventory at the warehouse as well as what is due to arrive, the final item is empty as we haven't made a stocking decision yet.
* **StartingInventoryStore**: Same as above but with the store.
* **Order**: A vector for the orders made, first item is from supplier to warehouse, and the following items are store orders.
* **Allocation_Req**: Boolean on if an allocation policy needed to be used to distribute stock. i.e. if we don't have enough stock to distribute.
* **PostOrderWarehouse**: `StartingInvWarehouse` but with the orders placed for the last item of the list, and first item depleted for inventory sent to stores.
* **PostOrderStore**: `StartingInventoryStore` but with the orders placed and added as the last item of the list.
* **Demand**: Generated demand, first item online demand, rest of the items store demands.
* **DFW_Fulfillment**: How much stock at each store was directed through DFW.
* **DFW_total**: Sum of above.
* **PostDemandWarehouse**: Pipeline vector after stock depletion from online demand.
* **PostDemandStore**: Pipeline vector after stock depletion from store demand and DFW rerouting.
* **EndingInventoryWarehouse**: Pipeline movements at the warehouse (so all items move forward in the list by one place).
* **EndingInventoryStore**: Same as above but for stores.
* **PeriodCost**: Cost of the period by multiplying shortage by the penalty cost or DFW cost and leftover stock by the holding cost.

#### Zero Lead Time simulation
Also included is a zero lead time simulation environment (`DiscreteEventSystemZLT.py`). In a practical setting, instant replenishment is unrealistic, however there are nice theoretical properties such as optimiality of the base-stock policy. It also provides a nice test bed to evaluate simple policies, which usually translate well to positive lead time cases (see for example a base-stock policy in zero lead time, is equivalent to the echelon base-stock in positive lead time). 

An example to use this is given below. It works relatively the same way as the positive equivalent with a few differences.

```python
import numpy as np
import DiscreteEventSystem.DiscreteEventSystemZLT as zlt_sim
import scipy.stats as sp

### PARAMETERS ##
T=36 # Length of simulation
sim_len = 2 # Number of simulations
n = 5 # Note it works out the number of stores from the other parameter rather than needing an explicit input
co = [1] + [2 for i in range(n)] # Holding, warehouse then stores. This and cu can have different parameters for each store unlike the lead time version.
cu = [18 for i in range(n+1)] # Penalty warehouse, then stores
dfw = [0 for i in range(n)] # DFW cost
salvage = 0 # How much items can be salvaged for at end of horizon
init = [0 for i in range(n+1)] # Initial inventory (doesn't matter if ordering policy is Base-stock)
p = 0.8 # Probability of DFW
order_type = 'OUT' # OUT = Base-stock policy, alternatives are FQ=Fixed Order Quantity, OPT=optimal policy which is redundant now base-stock is provably optimal.
log = True

# Set order Quantity
wh = [64 for t in range(T)]
st = [[11 for i in range(n)] for t in range(T)]

# Generate demand (Poisson(10))
demand = np.array([[[np.random.poisson(10) for i in range(n+1)] for t in range(T)] for s in range(sim_len)])

# Create instance and set orders
test_zlt = zlt_sim.zlt_inventory_simulation(T, cu, co, dfw, salvage, init, p, 'OUT', log)
test_zlt.set_order_q([[w] + s for w,s in zip(wh,st)])

# Run first and reset
test_zlt.run(demand[0])
test_zlt.reset()

# Run second and print log
test_zlt.run(demand[1])
test_zlt.log
```
## Heuristics

### Zero Lead Time
The baseline method is similar to stock cover in the current system, keeping it at a constant level. We propose two heuristics to tackle the problem and improve costs. A local control and central control method.

For zero lead time, stock arrives instantly at each echelon and its provably optimal to use a base-stock policy. 

#### Baseline (Search over feasible space, same decision all time periods.)
For stationary values, we perform an exhaustive search over base-stock values at warehouse ($y_t^{(w)}$) and store ($y_t^{(s)}$). Assumes all stores order the same, so only have to optimise over two parameters. Use simulation to evaluate results. For non-stationary and asymmetric this acts as a benchmark. 


#### Local Control (Optimal Search, then Aggregate)
This method solves a simple stochastic dynamic program for inidividual stores, then aggregates. Example:

```python
import numpy as np
import Heuristics.ZeroLeadTime.LocalControl as lc

n = 5 
cu = [18 for i in range(n+1)]
co = [1] + [3 for  i in range(n)]
cdfw = [0 for i in range(n)]
p = 0.8

# Generate the demand parameters
# Assume 2 stores have a demand of Pois(10) and 3 stores with Pois(5), no online demand. For 5 periods. Save as a numpy array
demand_lambdas = np.array([[0] + [10 for i in range(2)] + [3 for i in range(5)] for t in range(5)])

# Will return the base-stock levels for the warehouse and stores across time.
lc.local_control_heuristic(n, cu, co, cdfw, p, demand_lambdas)
```
The output is a numpy array iwhos rows are time index's, and columns the base-stock level for the warehouse, followed by the `n` stores.

#### Central Control (Search over feasible space, then proportionally adjust)


### Positive Lead Time
#### Baseline



#### Order Cap
Much like the previous one, the Order cap is just found through a search over parameters in three dimensions. There are some logical ways to set bounds but it is essentially searching different combinations, and finding the best one. Given the problem is easily parellelised, multiple searches can be carried out at the same time. Some pointers on search bounds:
* The order cap rarely varies much from the mean of the expected demand (especially for Poisson).
* As the holding cost at the store increases, the bounds at the store can be reduced, and warehouse can be increased. And vice versa.

Example code for searching for an order cap:

```Python
import 


```