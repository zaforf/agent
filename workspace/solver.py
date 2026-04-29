import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os

params = {
    "WLSACCESSID": 'f78e1d62-d325-454b-8fbe-11a1f7316748',
    "WLSSECRET": '9d4fe479-8352-4805-ac38-430a0a359347',
    "LICENSEID": 2812682,
}
env = gp.Env(params=params)

# ==========================================
# 1. DATA LOADING
# ==========================================
def time_to_min(time_str):
    if pd.isna(time_str): return 0
    if isinstance(time_str, str):
        try:
            h, m = map(int, time_str.split(':')[:2])
            return h * 60 + m
        except: return 0
    elif hasattr(time_str, 'hour'):
        return time_str.hour * 60 + time_str.minute
    return 0

def min_to_time(minutes):
    if minutes is None: return "N/A"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"

print("Loading data...")
excel_file = '/content/Team_10_Data.xlsx'
if not os.path.exists(excel_file):
    excel_file = 'Team_10_Data.xlsx'

stops_df = pd.read_excel(excel_file, sheet_name='Delivery Stops')
fleet_df = pd.read_excel(excel_file, sheet_name='Vehicle Fleet')
depot_df = pd.read_excel(excel_file, sheet_name='Depot')
dist_df = pd.read_excel(excel_file, sheet_name='Distance Matrix (miles)', index_col=0)
time_df = pd.read_excel(excel_file, sheet_name='Travel Time Matrix (min)', index_col=0)
costs_df = pd.read_excel(excel_file, sheet_name='Cost Parameters').set_index('Parameter')['Value']

fuel_cost_per_mile = costs_df.get('Fuel_Cost_per_Liter', 1.79) * costs_df.get('Avg_Fuel_Consumption_L_per_mile', 0.15)
driver_hourly_wage = costs_df.get('Driver_Hourly_Wage', 31.0)
late_penalty_per_min = costs_df.get('Late_Delivery_Penalty_per_Hour', 51.0) / 60.0

depot_open = time_to_min(depot_df['Open_Time'].iloc[0])
depot_close = time_to_min(depot_df['Close_Time'].iloc[0])
nodes = ['Depot'] + [f"Stop {int(i)}" for i in stops_df['Stop_ID']]
node_indices = list(range(len(nodes)))

demand = {0: 0}
e_tw = {0: depot_open}
l_tw = {0: depot_close}
s_time = {0: 0}
for idx, row in stops_df.iterrows():
    node_idx = int(row['Stop_ID'])
    demand[node_idx] = row['Demand_kg']
    e_tw[node_idx] = time_to_min(row['Earliest_Delivery'])
    l_tw[node_idx] = time_to_min(row['Latest_Delivery'])
    s_time[node_idx] = row['Service_Time_min']

vehicles = fleet_df['Vehicle_ID'].tolist()
capacity = {v: fleet_df.loc[fleet_df['Vehicle_ID'] == v, 'Capacity_kg'].values[0] for v in vehicles}
fixed_cost = {v: fleet_df.loc[fleet_df['Vehicle_ID'] == v, 'Fixed_Daily_Cost'].values[0] for v in vehicles}
var_cost_per_mile = {v: fleet_df.loc[fleet_df['Vehicle_ID'] == v, 'Cost_per_mile'].values[0] for v in vehicles}
v_avail_from = {v: time_to_min(fleet_df.loc[fleet_df['Vehicle_ID'] == v, 'Available_From'].values[0]) for v in vehicles}
v_avail_until = {v: time_to_min(fleet_df.loc[fleet_df['Vehicle_ID'] == v, 'Available_Until'].values[0]) for v in vehicles}
large_trucks = [v for v in vehicles if capacity[v] > 1100]

dist_matrix = {(i, j): dist_df.loc[nodes[i], nodes[j]] for i in node_indices for j in node_indices}
time_matrix = {(i, j): time_df.loc[nodes[i], nodes[j]] for i in node_indices for j in node_indices}

# ==========================================
# 2. GUROBI MODEL
# ==========================================
model = gp.Model("HorizonLogistics_Optimized", env=env)

model.Params.TimeLimit = 3600
model.Params.MIPGap = 0.05
model.Params.Threads = 0
model.Params.Heuristics = 0.2 
# Aggressive cutting planes to close the gap faster
model.Params.Cuts = 2 

# --- Pre-processing: Combined Arc Pruning ---
# We use temporal pruning (smart) and a distance-based fallback if the model is too large
possible_arcs = []
for i in node_indices:
    for j in node_indices:
        if i == j: continue
        # 1. Temporal feasibility check
        if i != 0 and j != 0:
            if e_tw[i] + s_time[i] + time_matrix[i, j] > l_tw[j]:
                continue
        elif i == 0:
            if depot_open + time_matrix[i, j] > l_tw[j]:
                continue
        elif j == 0:
            if e_tw[i] + s_time[i] + time_matrix[i, j] > depot_close:
                continue
        possible_arcs.append((i, j))

# Variables
x = model.addVars(((i, j, k) for (i, j) in possible_arcs for k in vehicles
                   if not (i in range(25, 32) and k in large_trucks)),
                  vtype=GRB.BINARY, name="x")
y = model.addVars(vehicles, vtype=GRB.BINARY, name="y")
arr = model.addVars([(i, k) for i in node_indices if i != 0 for k in vehicles], vtype=GRB.CONTINUOUS, lb=0, name="arr")
start_time = model.addVars(vehicles, vtype=GRB.CONTINUOUS, lb=0, name="start_time")
end_time = model.addVars(vehicles, vtype=GRB.CONTINUOUS, lb=0, name="end_time")
late = model.addVars(node_indices, vtype=GRB.CONTINUOUS, lb=0, name="late")
z = model.addVars([(i, k) for i in node_indices for k in vehicles], vtype=GRB.BINARY, name="z")
hours_worked = model.addVars(vehicles, vtype=GRB.INTEGER, lb=0, name="hours_worked")

# Objective
obj = gp.quicksum(fixed_cost[k] * y[k] for k in vehicles) + \
      gp.quicksum((var_cost_per_mile[k] + fuel_cost_per_mile) * dist_matrix[i, j] * x[i, j, k]
                  for i, j, k in x.keys()) + \
      gp.quicksum(driver_hourly_wage * hours_worked[k] for k in vehicles) + \
      gp.quicksum(late_penalty_per_min * late[i] for i in node_indices if i != 0)
model.setObjective(obj, GRB.MINIMIZE)

# --- Tight Big-M Calculation ---
M_time = {}
for (i, j) in possible_arcs:
    if i != 0 and j != 0:
        M_time[i, j] = max(0, l_tw[i] + s_time[i] + time_matrix[i, j] - e_tw[j])
    elif i == 0:
        M_time[i, j] = max(0, depot_close + time_matrix[i, j] - e_tw[j])
    elif j == 0:
        M_time[i, j] = max(0, l_tw[i] + s_time[i] + time_matrix[i, j] - depot_open)

# 1. Every stop visited once
for i in node_indices:
    if i == 0: continue
    model.addConstr(gp.quicksum(x[i, j, k] for j in node_indices for k in vehicles if (i, j, k) in x) == 1)

# 2. Flow and Capacity
for k in vehicles:
    for j in node_indices:
        model.addConstr(gp.quicksum(x[i, j, k] for i in node_indices if (i, j, k) in x) ==
                        gp.quicksum(x[j, i, k] for i in node_indices if (j, i, k) in x))
    model.addConstr(gp.quicksum(demand[i] * gp.quicksum(x[i, j, k] for j in node_indices if (i, j, k) in x)
                                for i in node_indices) <= capacity[k] * y[k])
    model.addConstr(gp.quicksum(x[0, j, k] for j in node_indices if (0, j, k) in x) == y[k])
    model.addConstr(gp.quicksum(x[i, 0, k] for i in node_indices if (i, 0, k) in x) == y[k])

# 3. Time Tracking
for i, j, k in x.keys():
    if i == 0:
        model.addConstr(arr[j, k] >= start_time[k] + s_time[0] + time_matrix[i, j] - M_time[i, j] * (1 - x[i, j, k]))
    elif j == 0:
        model.addConstr(end_time[k] >= arr[i, k] + s_time[i] + time_matrix[i, j] - M_time[i, j] * (1 - x[i, j, k]))
    else:
        model.addConstr(arr[j, k] >= arr[i, k] + s_time[i] + time_matrix[i, j] - M_time[i, j] * (1 - x[i, j, k]))

# 4. Windows and Availability
for k in vehicles:
    for i in node_indices:
        if i != 0:
            model.addConstr(arr[i, k] >= e_tw[i] - (depot_close - depot_open) * (1 - gp.quicksum(x[i, j, k] for j in node_indices if (i, j, k) in x)))
            model.addConstr(late[i] >= arr[i, k] - l_tw[i])
    model.addConstr(start_time[k] >= v_avail_from[k])
    model.addConstr(end_time[k] <= v_avail_until[k])
    # Billed hours logic: tighten the bound to help the solver
    model.addConstr(hours_worked[k] * 60 >= (end_time[k] - start_time[k]) - (depot_close - depot_open) * (1 - y[k]))
    model.addConstr(hours_worked[k] >= (end_time[k] - start_time[k]) / 60 - 1)

# 5. Symmetry Breaking
for i in range(len(vehicles) - 1):
    if capacity[vehicles[i]] == capacity[vehicles[i+1]] and \
       fixed_cost[vehicles[i]] == fixed_cost[vehicles[i+1]] and \
       var_cost_per_mile[vehicles[i]] == var_cost_per_mile[vehicles[i+1]]:
        model.addConstr(y[vehicles[i+1]] <= y[vehicles[i]])

# 6. Downtown (12-16)
for i in range(12, 17):
    for k in vehicles:
        m_down = max(l_tw[i] - 600, 960 - e_tw[i]) if i != 0 else (depot_close - 600)
        model.addConstr(arr[i, k] <= 600 + m_down * (1 - z[i, k]))
        model.addConstr(arr[i, k] >= 960 - m_down * z[i, k])

# 7. Deadlines (4, 6)
for i in [4, 6]:
    for k in vehicles:
        model.addConstr(arr[i, k] <= 960 + (depot_close - depot_open) * (1 - gp.quicksum(x[i, j, k] for j in node_indices if (i, j, k) in x)))

# 8. Hendricks (8, 9)
for k in vehicles:
    model.addConstr(gp.quicksum(x[8, j, k] for j in node_indices if (8, j, k) in x) +
                    gp.quicksum(x[9, j, k] for j in node_indices if (9, j, k) in x) <= 1)

# 9. Jimmy's Stops (25, 26, 28, 29)
jimmy_stops = [25, 26, 28, 29]
v_jimmy = model.addVars(vehicles, vtype=GRB.BINARY, name="v_jimmy")
for k in vehicles:
    for i in jimmy_stops:
        model.addConstr(gp.quicksum(x[i, j, k] for j in node_indices if (i, j, k) in x) == v_jimmy[k])
    if k in large_trucks:
        model.addConstr(v_jimmy[k] == 0)
model.addConstr(gp.quicksum(v_jimmy[k] for k in vehicles) == 1)

# 10. Flatbed (1, 2, 3)
for i in [1, 2, 3]:
    model.addConstr(gp.quicksum(x[i, j, 6] for j in node_indices if (i, j, 6) in x) == 1)

# --- Checkpoint System: Load Best Solution ---
import os
checkpoint_file = "best_solution.sol"
if os.path.exists(checkpoint_file):
    print(f"Loading checkpoint from {checkpoint_file}...")
    model.read(checkpoint_file)

# --- Solver Execution with Checkpoint Save ---
model.optimize()

# Save the best found solution for next time
if model.SolCount > 0:
    model.write(checkpoint_file)
    print(f"Best solution saved to {checkpoint_file}")

# ==========================================
# 3. EXECUTION & BUSINESS REPORTING
# ==========================================
print("\nSolving with Full Gurobi Model (Tighter M)...")
model.optimize()

if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
    print(f"\n--- FINAL SOLUTION REPORT ---")
    print(f"Total Cost: ${model.objVal:.2f}")

    for k in vehicles:
        if y[k].X > 0.5:
            print(f"\nVEHICLE {k} ({'Large' if k in large_trucks else 'Small/Med'}):")

            load = sum(demand[i] for i in node_indices if any(x[i, j, k].X > 0.5 for j in node_indices if (i, j, k) in x))
            util = (load / capacity[k]) * 100
            print(f"  Utilization: {util:.1f}% | Load: {load}/{capacity[k]}kg | Billed Hours: {int(hours_worked[k].X)}")

            curr = 0
            visited = {0}
            start = start_time[k].X
            print(f"  Depot Depart: {min_to_time(start)}")
            while True:
                found = False
                for j in node_indices:
                    if curr != j and (curr, j, k) in x and x[curr, j, k].X > 0.5:
                        arrival = arr[j, k].X if j != 0 else end_time[k].X
                        departure = arrival + s_time[j]
                        print(f"  {nodes[curr]} -> {nodes[j]} | Arrive: {min_to_time(arrival)} | Depart: {min_to_time(departure)}")
                        curr = j
                        visited.add(j)
                        found = True
                        break
                if not found: break
