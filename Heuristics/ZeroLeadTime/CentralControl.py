# Parellised version of the Central Control Heuristic 
import numpy as np
import itertools
import DiscreteEventSystem.DiscreteEventSystemZLT as zlt_sim
import multiprocessing as mp
import scipy.stats as sp

def constant_base_stock(n,demand_params,T, underage, overage, dfw, p, init, sim_len, store_range, warehouse_range, num_cores):
    """
        Params:
            n: number of stores
            demand_params: demand
            T: number of periods
            underage: underage cost
            overage: overage cost
            dfw: DFW cost
            p: probability of DFW
            init: initial inventory
            sim_len: number of simulations to run
            store_range: range of store base stock levels to search through
            warehouse_range: range of warehouse base stock levels to search through (this is just purely online + dfw estimation)   
    """

    

    # Actions
    # Sensible ranges:
    #     For store ranges, start from 0 and go up to the a bit over the newsvendor order.
    #     For warehouse ranges, start at the Online demand order mean level, and go up to the expected number of witholding at stores x number of stores
    store_base_stock_options = store_range
    warehouse_base_stock_options = warehouse_range # These are just taking into account online demand and potential DFW, we add onto the warehouse order level n*store_order_level

    # Model params
    salvage = 0
    # Go through each combination of actions:
    st_wh_actions = itertools.product(store_base_stock_options,warehouse_base_stock_options)
    # Run a simulation on each action set


    # Pre-generate the demand
    np.random.seed(42)
    demand = np.array([[[np.random.poisson(demand_params[t][i]) for i in range(n+1)]for t in range(T)] for i in range(sim_len)])

    cl = mp.Pool(num_cores)
    # Save parellised costs
    all_tot_costs_p = cl.starmap(evaluate_parameter_set_cbs,list(zip(reversed(list(st_wh_actions)),itertools.repeat(demand),itertools.repeat([n, T, underage, overage, dfw, salvage, init, p, sim_len]))))
    cl.close()

    all_tot_costs  = {}
    for best, orders, cost in all_tot_costs_p:
        all_tot_costs['excess_warehouse_{}_store_{}'.format(best[0], best[1])]=[cost, orders]

    best = sorted(all_tot_costs.items(), key=lambda x:x[1])[0]
    return best, all_tot_costs

def evaluate_parameter_set_cbs(st_wh_combo, demand, parameters):
    """
        Parmater order: [n, T, underage, overage, dfw, salvage, init, p, sim_len]
    """
    st = st_wh_combo[0]
    wh = st_wh_combo[1]
    n,T,underage,overage,dfw,salvage,init,p,sim_len = parameters

    # Set a common random seed
    np.random.seed(37)

    # Log action costs
    action_cost_tot = []

    order_up_to = []
    order_up_to = [[wh + st*n] + [st for i in range(n)] for t in range(T)]
    # Run instance
    test_zlt = zlt_sim.zlt_inventory_simulation(T, underage, overage, dfw, salvage, init, p, 'OUT', False)
    test_zlt.set_order_q(order_up_to)

    # Run 1000 simulations
    for sim in range(sim_len):
        test_zlt.run(demand[sim])
        action_cost_tot.append(test_zlt.period_cost.sum())
        test_zlt.reset()
    return [[wh, st], order_up_to, np.mean(action_cost_tot)]

# Deals with change in demand
def adapted_constant_base_stock_demand_change(n,demand_params,T, underage, overage, dfw, p, init, sim_len, store_range, warehouse_range, num_cores):
    """
        Params:
            n: number of stores
            T: number of periods
            underage: underage cost
            overage: overage cost
            dfw: DFW cost
            p: probability of DFW
            demand_params: list of demand parameters for each store
            init: initial inventory
            sim_len: number of simulations to run
            store_range: range of store base stock levels to search through
            warehouse_range: range of warehouse base stock levels to search through (this is just purely online + dfw estimation)   
    """
    
    # Action ranges
    store_base_stock_options = store_range
    warehouse_base_stock_options = warehouse_range # These are just taking into account potential DFW, we add onto the warehouse order level n*store_order_level
    
    # Calculate most common parameter across time and stores
    # Collapse into one list and then find mode
    store_demands_collapsed = [d for d_t in demand_params for d in d_t[1:]]
    mode_store = max(store_demands_collapsed, key=store_demands_collapsed.count)
    
    # Make sure the minimum store action is at least the mode of the store
    #store_base_stock_options[0] = int(mode_store)

    st_wh_actions = itertools.product(store_base_stock_options,warehouse_base_stock_options)
    salvage = 0

    # Set up the simulation
    np.random.seed(42)
    demand = np.array([[[np.random.poisson(demand_params[t][i]) for i in range(n+1)] for t in range(T)] for i in range(sim_len)])

    cl = mp.Pool(num_cores)
    # Save parellised costs
    all_tot_costs_p = cl.starmap(evaluate_parameter_set_acbs_demand,list(zip(reversed(list(st_wh_actions)),itertools.repeat(demand),itertools.repeat(demand_params),itertools.repeat(mode_store),itertools.repeat([n, T, underage, overage, dfw, salvage, init, p, sim_len]))))
    cl.close()
    
    all_tot_costs  = {}
    for mode_orders, orders, cost in all_tot_costs_p:
        all_tot_costs['warehouse_dfw_{}_store_{}'.format(mode_orders[0], mode_orders[1])]=[cost, orders]

    best = sorted(all_tot_costs.items(), key=lambda x:x[1][0])[0]
    return best, all_tot_costs
    

def evaluate_parameter_set_acbs_demand(st_wh_combo, demand, demand_params, mode_store, parameters):
    """
        Paramater order: [n, T, demand_params, underage, overage, dfw, salvage, init, p, sim_len]

    """
    # Set a common random seed
    np.random.seed(37)
    st = st_wh_combo[0]
    wh = st_wh_combo[1] # Add online order for warehouse (currently just considers demand)
    
    n,T, underage,overage,dfw,salvage,init,p,sim_len = parameters
    safety_stock_perc = ((st-mode_store)/mode_store)

    # Log action costs
    action_cost_tot = []

    st_out = []
    for t in range(T):
        store_out_t = []
        for store in range(n):
            if demand_params[t][store+1] == mode_store:
                store_out_t.append(st)
            else:
                store_out_t.append(np.round(demand_params[t][store+1]+demand_params[t][store+1]*safety_stock_perc))
        st_out.append(store_out_t)

    wh_online_demand = [sp.poisson(demand_params[t][0]).ppf(underage[0]/(underage[0]+overage[0])) for t in range(T)]

    order_up_to = [[wh+wh_online_demand[t] + np.sum(st_out[t])] + st_out[t] for t in range(T)]
        
    # Run instance
    test_zlt = zlt_sim.zlt_inventory_simulation(T, underage, overage, dfw, salvage, init, p, 'OUT', False)
    test_zlt.set_order_q(order_up_to)

    # Run 1000 simulations
    for sim in range(sim_len):
        test_zlt.run(demand[sim])
        action_cost_tot.append(test_zlt.period_cost.sum())
        test_zlt.reset()
    return [[wh, st],order_up_to, np.mean(action_cost_tot)]

# Deals with change in holding cost
def adapted_constant_base_stock_holding_change(n, demand_params, T, underage, overage, dfw, p, init, sim_len, store_range, warehouse_range, num_cores):

    # Action ranges
    store_base_stock_options = store_range
    warehouse_base_stock_options = warehouse_range

    # Calculate the most common parameter 
    mode_overage = max(overage[1:], key=overage[1:].count)

    # Make sure the minimum store
    st_wh_actions = itertools.product(store_base_stock_options, warehouse_base_stock_options)
    salvage = 0

    # Set up the simulation
    np.random.seed(42)
    demand = np.array([[[np.random.poisson(demand_params[t][i]) for i in range(n+1)] for t in range(T)] for i in range(sim_len)])

    cl = mp.Pool(num_cores)
    # Save parellised costs
    all_tot_costs_p = cl.starmap(evaluate_parameter_set_acbs_overage,list(zip(reversed(list(st_wh_actions)),itertools.repeat(demand),itertools.repeat(demand_params),itertools.repeat(mode_overage),itertools.repeat([n, T, underage, overage, dfw, salvage, init, p, sim_len]))))
    cl.close()
    
    all_tot_costs  = {}
    for mode_orders, orders, cost in all_tot_costs_p:
        all_tot_costs['warehouse_dfw_{}_store_{}'.format(mode_orders[0], mode_orders[1])]=[cost, orders]

    best = sorted(all_tot_costs.items(), key=lambda x:x[1][0])[0]
    return best, all_tot_costs

def evaluate_parameter_set_acbs_overage(st_wh_combo, demand, demand_params, mode_overage, parameters):
    """
        Paramater order: [n, T, demand_params, underage, overage, dfw, salvage, init, p, sim_len]

    """
    # Set a common random seed
    np.random.seed(37)
    st = st_wh_combo[0]
    wh = st_wh_combo[1] # Add online order for warehouse (currently just considers demand)
    
    n,T, underage,overage,dfw,salvage,init,p,sim_len = parameters
    safety_stock_perc = ((st-mode_store)/mode_store)

    # Log action costs
    action_cost_tot = []

    st_out_one_time_period = []
    # Adjust the different parameter values
    for c_o in overage[1:]:
        if c_o == mode_overage:
            st_out_one_time_period.append(st)
        else:
            CF_mode = underage[1]/(underage[1]+mode_overage)
            CF =  underage[1]/(underage[1]+c_o)
            if c_o <= mode_overage:
               
                st_out_one_time_period.append(np.floor(st-st*(CF_mode-CF)))
                
            elif c_o >= mode_overage:
                st_out_one_time_period.append(np.ceil(st+st*(CF-CF_mode)))
    st_out = [st_out_one_time_period for t in range(T)]

    wh_online_demand = [sp.poisson(demand_params[t][0]).ppf(underage[0]/(underage[0]+overage[0])) for t in range(T)]

    order_up_to = [[wh+wh_online_demand[t] + np.sum(st_out[t])] + st_out[t] for t in range(T)]
        
    # Run instance
    test_zlt = zlt_sim.zlt_inventory_simulation(T, underage, overage, dfw, salvage, init, p, 'OUT', False)
    test_zlt.set_order_q(order_up_to)

    # Run 1000 simulations
    for sim in range(sim_len):
        test_zlt.run(demand[sim])
        action_cost_tot.append(test_zlt.period_cost.sum())
        test_zlt.reset()
    return [[wh, st],order_up_to, np.mean(action_cost_tot)]