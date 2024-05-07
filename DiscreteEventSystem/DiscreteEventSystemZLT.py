import numpy as np
import pandas as pd

class zlt_inventory_simulation:

    def __init__(self, periods, underage, overage, dfw_cost, salvage, initial_inventory, prob_dfw, order_rule,log=False):
        self.T = periods
        self.cu = underage
        self.co = overage
        self.c = salvage
        self.c_dfw = dfw_cost
        self.p = prob_dfw
        self.order_rule = order_rule
        self.init_inv = initial_inventory
        self.N = len(initial_inventory)-1
        
        self.log_data = log
        if(self.log_data):
            # Create empty dictionary to store information which will eventually become a dataframe
            self.log = {}
        
        # Set up inventory levels
        self.x = np.zeros((self.T+1, self.N+1))
        self.x[0] = initial_inventory
        
        if (self.log_data):
            # Create empty dictionary to store information which will eventually become a dataframe
            self.log = {'StartingInv': [], 'Order': [],'Allocation_Req':[], 'PostOrder': [], 'Demand': [], 'DFW_Fulfillment': [], 'DFW_total': [], 'EndingInventory': [],'PeriodCost': []}
    
    def set_order_q(self, order):
        if self.order_rule == 'FQ': # If a fixed order quantity
            self.orders = lambda t :  [order[t][i] for i in range(self.N+1)]
        elif self.order_rule == 'OUT': # If an order-up-to level, we have a special case for the warehouse where we need to subtract the stroe inventory less than the order level from the warehouse OuT
            self.orders = lambda t: [order[t][0]-sum([min(order[t][i],self.x[t][i])for i in range(1,self.N+1)])-self.x[t][0]]+[max(order[t][i]-self.x[t][i],0) for i in range(1,self.N+1)]
        elif self.order_rule == 'OPT': # If an optimal policy, we have a different order rule based on the current inventory
            # Rename columns to lower case incase we entered them incorrectly with camel case
            order.rename(str.lower, axis='columns')
            self.orders = lambda t: [max(int(order[(order['period'] == t+1) & (order['inventory'] == tuple(map(int, self.x[t])))]['order-up-to'].values[0][i])-self.x[t][i],0) for i in range(self.N+1)] # very very silly method, should improve
    
    def reset(self):
        """Reset the inventory simulation to its initial state."""
        self.x = np.zeros((self.T+1, self.N+1))
        self.x[0] = self.init_inv
        self.period_cost = np.zeros(self.T+1)
        if (self.log_data):
            self.log = {'StartingInv': [], 'Order': [],'Allocation_Req':[], 'PostOrder': [], 'Demand': [], 'DFW_Fulfillment': [], 'DFW_total': [], 'EndingInventory': [],'PeriodCost': []}
    
     
    def allocate_stock(self, t,Q, allocation_policy='max_req'):
        """
             Allow more allocation policies
        """
        x = self.x[t].copy()
        
        # Check if we need to allocate in the first place
        x_wh = x[0]+Q[0]-np.sum(Q[1:])
        
        # This case we don't need to allocate stock
        if(x_wh>=0):
            x[0] += Q[0]-np.sum(Q[1:])
            x[1:] += Q[1:]
            if(self.log_data):
                self.log['Allocation_Req'].append(False)
            return x
        
        # If not then we enact an allocation policy
        # TODO: lost allocation charges (?)
        if allocation_policy == 'max_req':
            if(self.log_data):
                self.log['Allocation_Req'].append(True)
            # Get available stock
            avail = x[0]+Q[0]
            store_q = Q[1:].copy() # keep track of how much we can allocate
            
            # Allocate one by one to each store with highest need
            while avail>0:
                store_q[np.argmax(store_q)]-=1
                avail-=1
            
            # Allocate deliveries and shipments
            x[0] = 0 # Warehouse will obviously have no stock now.
            x[1:] += np.subtract(Q[1:],store_q) # Store stock
            
        return x 
    
    def run(self, demand):
        
            
        # demand
        self.demand = demand
        
        # period costs
        self.period_cost = np.zeros([self.T+1, self.N+1])
        
        for t in range(self.T):
            if (self.log_data):
                si = self.x[t].copy()
                self.log['StartingInv'].append(si)
            
            # Step 1. Find stocking decision for the period
            Q = self.orders(t)
            
            # Step 2. Echelons recieve stock and warehouse loses stock
            
            # Set an allocation procedure to make sure max amount of stock is sent.
            self.x[t] = self.allocate_stock(t,Q)
            
            if (self.log_data):
                po = self.x[t].copy()
                self.log['PostOrder'].append(po)
                
            # Step 3. Demand in each channel realised
            self.x[t]-=self.demand[t]
            
            # Step 4. Calculate DFW fulfilment
            dfw_fulfillment = np.zeros(self.N)
            
            for idx, s in enumerate(self.x[t][1:]):
                if s<0:
                    ### TODO: move to the "better" way of generating random numbers in numpy ###
                    dfw_fulfillment[idx]=np.random.binomial(np.abs(s), self.p)
                    self.x[t][idx+1]+=dfw_fulfillment[idx]
                else:
                    dfw_fulfillment[idx] = 0
            
            # Take off DFW fulfilment from warehouse
            self.x[t][0] -= np.sum(dfw_fulfillment)
            
            # Step 5. Calculate period costs
            
            # Warehouse cost
            self.period_cost[t][0] = np.abs(self.x[t][0]*self.cu[0]) if self.x[t][0]<=0 else self.x[t][0]*self.co[0]
            
            # Store cost
            self.period_cost[t][1:] = self.c_dfw*dfw_fulfillment  # DFW cost
            for s in range(1,self.N+1):
                self.period_cost[t][s] += np.abs(self.x[t][s]*self.cu[s]) if self.x[t][s]<=0 else self.x[t][s]*self.co[s]
            
            # Step 6. Carry Inventory to next period
            self.x[t+1][0] = self.x[t][0]
            self.x[t+1][1:] = np.maximum(self.x[t][1:],0)
            
            if(self.log_data):
                self.log['Order'].append(Q)
                self.log['Demand'].append(self.demand[t])
                self.log['DFW_Fulfillment'].append(dfw_fulfillment)
                self.log['DFW_total'].append(np.sum(dfw_fulfillment))
                self.log['EndingInventory'].append(self.x[t]) # i.e. record inventory after pipeline movements
                self.log['PeriodCost'].append(np.sum(self.period_cost[t]))
        
        # Salvage remaining inventory:
        self.period_cost[self.T] = -self.c*self.x[self.T]
        if(self.log_data):
            self.log['StartingInv'].append(self.x[self.T])
            self.log['Order'].append(0)
            self.log['PostOrder'].append(0)
            self.log['Demand'].append(0)
            self.log['Allocation_Req'].append(0)
            self.log['DFW_Fulfillment'].append(0)
            self.log['DFW_total'].append(0)
            self.log['EndingInventory'].append(0)
            self.log['PeriodCost'].append(np.sum(self.period_cost[self.T]))

        # Export log to dataframe
        if(self.log_data):
            self.log = pd.DataFrame(self.log)
            self.log.index +=1
            self.log.index.name = 'Period'