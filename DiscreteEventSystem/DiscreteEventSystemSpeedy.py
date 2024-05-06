##
# DiscreteEventSystemSpeedy.py
# A quicker version of DiscreteEventSystem
# Cuts some corners and removes output options like the log
# But... does make it quicker
# This is good for iterating through lots of policies.
##

import numpy as np





class DES:

    def __init__(self, periods, N, store_lead_time, warehouse_lead_time, production_capacity, warehouse_capacity, store_capacity, p, cost_dfw, holding_warehouse, holding_store, shortage_warehouse, shortage_store, warehouse_initial, store_initial):
        self.T = periods
        self.N = N
        self.l_s = store_lead_time
        self.l_w = warehouse_lead_time
        self.cap_prod = production_capacity
        self.cap_w = warehouse_capacity
        self.cap_s = store_capacity
        self.p = p
        self.c_dfw = cost_dfw
        self.co_w = holding_warehouse
        self.co_s = holding_store
        self.cu_w = shortage_warehouse
        self.cu_s = shortage_store
        self.init_warehouse = warehouse_initial
        self.init_store = store_initial

        

        # Set up the inventory levels
        self.x_warehouse = np.zeros((self.T+1, self.l_w+1))
        self.x_store = np.zeros((self.N, self.T+1, self.l_s+1))

        # Populate the initial inventory evenely across the inventory pipeline (excluding the L position which is for orders)
        if (self.l_w > 0):
            for l in range(self.l_w):
                self.x_warehouse[0][l] = int(self.init_warehouse/(self.l_w+1))
        else:
            self.x_warehouse[0][0] = self.init_warehouse

        for store in range(self.N):
            if (self.l_s > 0):
                for l in range(self.l_s):
                    self.x_store[store][0][l] = int(self.init_store/(self.l_s+1))
            else:
                self.x_store[store][0][0] = self.init_store

    

    def calc_store_orders(self,t):
        desired_orders = [] # Store "desired" orders for each store
        for store in range(self.N):
            # Applies the order quantity
            # Workout current inventory pipeline
            inv_pipeline_store = np.sum(self.x_store[store][t])
            # Base-stock policy
            desired_q = max(self.st_out[store][t]-inv_pipeline_store,0)
            # Apply the storage capacity
            desired_q = min(desired_q, self.cap_s-inv_pipeline_store)
            
            # Apply order cap
            desired_q = min(desired_q, self.r[store][t])

            desired_orders.append(desired_q)
        
        # Check we have enough in thw warehouse to fulfil the order
        total_orders = np.sum(desired_orders)
        if total_orders <= self.x_warehouse[t][0]:
            return desired_orders
        else:
            allocated_orders = self.allocate_stock(desired_orders,self.x_warehouse[t][0], t)
            return allocated_orders
    
    def calc_warehouse_orders(self,t):
        # Calculate the echelon order up-to-level
        # i.e. sum inventory position across the entire network
        inv_pipeline_all = sum(self.x_warehouse[t])+sum([sum(self.x_store[store][t]) for store in range(self.N)])

        warehouse_q = max(self.ech_out[t]-inv_pipeline_all,0)

        # Apply the production capacity constraint
        warehouse_q = min(warehouse_q, self.cap_prod)

        # Apply the warehouse capacity
        warehouse_q = min(warehouse_q, self.cap_w-inv_pipeline_all)

        return warehouse_q
    


    def allocate_stock(self, desired_orders, available_stock, t):
        # Add at the start of the allocation the online desired orders
        desired_orders = [self.online_demand_alloc[t]]  + desired_orders

        actual_allocation = [0 for i in range(len(desired_orders))] 

        # Go through how much we have available and assign one by one
        for i in range(int(available_stock)):
            # Get index of largest shortfall
            idx = np.argmax(desired_orders)
            actual_allocation[idx] += 1
            desired_orders[idx] -= 1
        return actual_allocation[1:]

    def set_order_q(self, echelon_base_stock_level, base_stock_level_st, order_cap=None):
        # In the echelon inventory we must have that ECH > sum of Store Base Stocks
        # The order cap is set less than the store base stock level.
        for t in range(self.T):
            if (np.sum([base_stock_level_st[store][t] for store in range(self.N)]) > echelon_base_stock_level[t]):
                raise ValueError('Sum of store base stocks must be less than echelon base stock level')
        self.ech_out = echelon_base_stock_level
        self.st_out = base_stock_level_st
        if (order_cap):
            self.r = order_cap
        else:
            # If no order cap is givng, not having an order cap is equivalent to setting the order cap at the base-stock since that's
            # the largest it can be
            self.r = base_stock_level_st
    
    def reset(self):
        """Reset the inventory simulation to its initial state."""
        self.x_warehouse = np.zeros((self.T+1, self.l_w+1))
        self.x_store = np.zeros((self.N, self.T+1, self.l_s+1))
        if (self.l_w > 0):
            for l in range(self.l_w):
                self.x_warehouse[0][l] = int(self.init_warehouse/(self.l_w+1))
        else:
            self.x_warehouse[0][0] = self.init_warehouse

        for store in range(self.N):
            if (self.l_s > 0):
                for l in range(self.l_s):
                    self.x_store[store][0][l] = int(self.init_store/(self.l_s+1))
            else:
                self.x_store[store][0][0] = self.init_store
                
        self.period_cost = np.zeros(self.T+1)
        
    def run(self, demand, online_demand):
        
        self.demand = demand
        self.online_demand_alloc = online_demand

        # Save period costs
        self.period_cost = np.zeros([self.T+1, self.N+1])

        for t in range(self.T):
            # Append starting inventory to log

            # Step 1. Get orders into the system 
            # We do the warehouse first (as this makes zero lead time at the warehouse easier to calculate)
            
            # Update the warehouse pipeline vector
            Q_wh = self.calc_warehouse_orders(t)
            self.x_warehouse[t][self.l_w] += Q_wh
            

            # For this, we assume a base-stock policy for stores.
            Q_stores = self.calc_store_orders(t)
           

            # Update the store pipeline vectors
            for store in range(self.N):
                self.x_store[store][t][self.l_s] += Q_stores[store]
              
            self.x_warehouse[t][0] -= np.sum(Q_stores)
            
            # Experince demand and DFW fulfilment
            dfw_fulfillment = []
            # Warehouse
            self.x_warehouse[t][0] -= self.demand[t][0]
            
            # Stores
            for store in range(self.N):
                self.x_store[store][t][0] -= self.demand[t][store+1]

                # Is DFW available?
                if self.x_store[store][t][0] < 0:
                    
                    dfw_request = np.random.binomial(np.abs(self.x_store[store][t][0]), self.p)
                    
                    # Here we can only give DFW from what's available in the warehouse
                    dfw_request = min(dfw_request, max(self.x_warehouse[t][0],0))
                    dfw_fulfillment.append(dfw_request)
                    
                    # Take DFW from warehouse and lesser the stockout at store
                    self.x_warehouse[t][0] -= dfw_request
                    self.x_store[store][t][0] += dfw_request
                else:
                    dfw_fulfillment.append(0) 
            
            # Calculate the period costs
                    
            # Warehouse
            self.period_cost[t][0] = np.abs(self.x_warehouse[t][0])*self.cu_w if self.x_warehouse[t][0]<=0 else self.x_warehouse[t][0]*self.co_w
           
            # Store
            for store in range(self.N):
                self.period_cost[t][store+1] = np.abs(self.x_store[store][t][0])*self.cu_s if self.x_store[store][t][0]<=0 else self.x_store[store][t][0]*self.co_s
                 # Add dfw
                self.period_cost[t][store+1] += self.c_dfw*dfw_fulfillment[store]

            # Carry inventory to next period
            # Warehouse
            if (self.l_w > 0):
                self.x_warehouse[t+1][0] = self.x_warehouse[t][0] + self.x_warehouse[t][1]
                for pos in range(1,self.l_w):
                    self.x_warehouse[t+1][pos] = self.x_warehouse[t][pos+1]
                self.x_warehouse[t+1][self.l_w] = 0
            else:
                self.x_warehouse[t+1][0] = self.x_warehouse[t][0]

            # Store 
            if(self.l_s > 0):
                for store in range(self.N):
                    self.x_store[store][t+1][0] = max(self.x_store[store][t][0],0) + self.x_store[store][t][1]
                    for pos in range(1,self.l_s):
                        self.x_store[store][t+1][pos] = self.x_store[store][t][pos+1]
                    self.x_store[store][t+1][self.l_s] = 0
            else:
                for store in range(self.N):
                    self.x_store[store][t+1][0] = max(self.x_store[store][t][0],0)



                
            

