import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Simulation parameters
T = 15
ts = 5
td_values = np.linspace(0, T, 10000)  
window_size = 100
num_simulations = 20000  # Number of simulations per td value


topology_keys_D_L = [
    "(A1,(A2,B2))", "((A2,B2),B1)", "(A2,B2)", "((A1,B1),A2)", "((A1,B1),B2)", "(A1,B1)",
    "(A1,(B1,B2))", "(A1,(A2,B1))", "((A2,B2),B1)", "((A2,B2),A1)", "(A2,B2)",
    "((A1,B2),B1)", "((A1,A2),B1)", "(A1,B1)", "(A1,(A2,B2))", "((A2,B2),B1)", "(A2,B2)",
    "(A1,B1)", "(A1,A2)", "(A2,B1)", "A2",
    "(A1,B1)", "(A1,B2)", "(B1,B2)", "B2",
    "(A1,B1)", "(A1,A2)", "(A2,B1)", "A2",
    "(A1,B1)", "(A1,B2)", "(B1,B2)", "B2",
    "(A1,B1)", "(A1,A2)", "(A2,B1)", "A2",
    "(A1,B1)", "(A1,B2)", "(B2,B1)", "B2"
]

topology_counts_D_L = {topo: np.zeros(len(td_values)) for topo in topology_keys_D_L}
topology_keys_D = [
    "((A1, B1), (A2, B2))",
    "(((A2, B2), B1), A1)",
    "((A1, A2), B1)",
    "((A1, B1), A2)",
    "((A2, B1), A1)"
]

topology_counts_D = {topo: np.zeros(len(td_values)) for topo in topology_keys_D}

def generate_topology_1D1L(T, ts, td, topology_counts_D_L, td_index):
        #for i, td in enumerate(td_values):
        tl = np.random.uniform(0, T)  # Loss time
        g1, g2 = np.random.exponential(1, 2)  # Exponential(1) samples
        tp = g1 + ts  # Time of coalescence in the parent lineage
        tc = g2 + ts  # Time of coalescence in the daughter lineage

        if td > tl:
            # Duplication Happens Before Loss
            if td < ts:
                topology_counts_D_L["(A1,A2)"][td_index] += 1
                topology_counts_D_L["(A1,B1)"][td_index] += 1
                topology_counts_D_L["(A2,B1)"][td_index] += 1
            else:
                if td > tp:
                    if td > tc:
                        if tl < tp:
                            if tl < tc:
                                topology_counts_D_L["(A1,(A2,B2))"][td_index] += 1
                                topology_counts_D_L["((A2,B2),B1)"][td_index] += 1
                                topology_counts_D_L["((A1,B1),A2)"][td_index] += 1
                                topology_counts_D_L["((A1,B1),B2)"][td_index] += 1
                            else:
                                topology_counts_D_L["(A1,(A2,B2))"][td_index] += 1
                                topology_counts_D_L["((A2,B2),B1)"][td_index] += 1
                                topology_counts_D_L["(A1,B1)"][td_index] += 1
                        else:
                            if tl < tc:
                                topology_counts_D_L["((A1,B1),A2)"][td_index] += 1
                                topology_counts_D_L["((A1,B1),B2)"][td_index] += 1
                                topology_counts_D_L["(A2,B2)"][td_index] += 1
                            else:
                                topology_counts_D_L["(A1,B1)"][td_index] += 1
                                topology_counts_D_L["(A2,B2)"][td_index] += 1
                    else:
                        if tl < tp:
                            topology_counts_D_L["(A1,B1)"][td_index] += 1
                            topology_counts_D_L["(A2,B1)"][td_index] += 1
                            topology_counts_D_L["(A1,A2)"][td_index] += 1
                            topology_counts_D_L["(B1,B2)"][td_index] += 1
                            topology_counts_D_L["(A1,B2)"][td_index] += 1
                        else:
                            topology_counts_D_L["(A1,(A2,B2))"][td_index] += 1
                            topology_counts_D_L["((A2,B2),B1)"][td_index] += 1
                            topology_counts_D_L["(A2,B2)"][td_index] += 1
                else:
                    if td > tc:
                        if tl < tc:
                                
                                topology_counts_D_L["(A1,(A2,B2))"][td_index]+=1/2
                                topology_counts_D_L["((A2,B2),B1)"][td_index]+=1/2
                            
                                tj = np.random.exponential(1/2)
                                if tj + td < tp:
                                    topology_counts_D_L["(A1,(A2,B1))"][td_index]+=1/2
                                    topology_counts_D_L["((A1,A2),B1)"][td_index]+=1/2
                                    topology_counts_D_L["(A1,(B1,B2))"][td_index]+=1/2
                                    topology_counts_D_L["((A1,B2),B1)"][td_index]+=1/2
                                else:
                                    topology_counts_D_L["((A1,B1),B2)"][td_index]+=1/2
                                    topology_counts_D_L["((A1,B1),A2)"][td_index]+=1/2
                                    
                        else:
                            topology_counts_D_L["(A1,(A2,B2))"][td_index]+=2/3
                            topology_counts_D_L["((A2,B2),B1)"][td_index]+=2/3
                            topology_counts_D_L["(A1,B1)"][td_index]+=1/3
                    else:
                        topology_counts_D_L["(A1,B1)"][td_index]+=1/3
                        topology_counts_D_L["(A1,A2)"][td_index]+=1/6
                        topology_counts_D_L["(A2,B1)"][td_index]+=1/6
                        topology_counts_D_L["(B1,B2)"][td_index]+=1/6
                        topology_counts_D_L["(A1,B2)"][td_index]+=1/6
                        
                        #yield [("A2", "B1"), ("A2", "A1"), ("B2", "B1"), ("B2", "A1")]
        else:
            # Duplication Happens After Loss
            if tl > tp:
                if td < tc:
                    topology_counts_D_L["A2"][td_index] += 1
                    topology_counts_D_L["B2"][td_index] += 1
                else:
                    topology_counts_D_L["(A2,B2)"][td_index] += 1
            else:
                if td < tc:
                    topology_counts_D_L["(A1,A2)"][td_index] += 1
                    topology_counts_D_L["(A1,B2)"][td_index] += 1
                    topology_counts_D_L["(A2,B1)"][td_index] += 1
                    topology_counts_D_L["(B1,B2)"][td_index] += 1
                else:
                    topology_counts_D_L["(A1,(A2,B2))"][td_index] += 1
                    topology_counts_D_L["((A2,B2),B1)"][td_index] += 1




# Loop over td values and run simulations
for td_index, td in enumerate(td_values):
    for _ in range(num_simulations):
        generate_topology_1D1L(T, ts, td, topology_counts_D_L, td_index)

# Normalize the counts for each topology
for topo in topology_counts_D_L.keys():
    topology_counts_D_L[topo] /= num_simulations

# Convert results to DataFrame
df_normalized = pd.DataFrame(topology_counts_D_L)
df_normalized["td"] = td_values

# Apply a sliding window average for smoothing
df_smoothed = df_normalized.copy()

# Apply rolling mean for smoothing on each topology column
for topology in df_smoothed.columns[:-1]: 
    df_smoothed[topology] = df_smoothed[topology].rolling(window=window_size, min_periods=1).mean()

# Extract and normalize the selected topology's counts
selected_topology = "((A1,B1),A2)"
selected_counts_smoothed = df_smoothed[selected_topology].values
pdf_td = selected_counts_smoothed / np.trapz(selected_counts_smoothed, td_values)  # Normalize to integrate to 1



# Updated function to update topology counts instead of using random.choice
def generate_topology(T, ts, td, topology_counts, td_index):
    g1, g2 = np.random.exponential(1, 2)  # Exponential(1) samples
    tp = g1 + ts  # Time of coalescence in the parent lineage
    tc = g2 + ts  # Time of coalescence in the daughter lineage
    
    if td < ts:
        tj = np.random.exponential(1)  # Exponential(1) sample
        if (tj + td) < ts:
            topology_counts["((A1, A2), B1)"][td_index] += 1
            #topology_counts["((B1, B2), A1)"][td_index] += 1/2
        else:
            tj = np.random.exponential(1/2)  # Exponential(2) sample
            if (tj + ts) > tp:
                topology_counts["((A1, B1), A2)"][td_index] += 1
                #topology_counts["((A1, B1), B2)"][td_index] += 1/2
            else:
                topology_counts["((A1, A2), B1)"][td_index] += 1/2
               #topology_counts["((B1, B2), A1)"][td_index] += 1/4
                #topology_counts["((A1, B2), B1)"][td_index] += 1/4
                topology_counts["((A2, B1), A1)"][td_index] += 1/2
    else:
        if td > tp:
            tj = np.random.exponential(1)
            if td > tc:
                topology_counts["((A1, B1), (A2, B2))"][td_index] += 1
            else:
                topology_counts["((A1, B1), A2)"][td_index] += 1
                #topology_counts["((A1, B1), B2)"][td_index] += 1/2
        else:
            tj = np.random.exponential(1/2)
            if td > tc:
                if tj + td > tp:
                    topology_counts["((A1, B1), (A2, B2))"][td_index] += 1
                else:
                    topology_counts["(((A2, B2), B1), A1)"][td_index] += 1
                    #topology_counts["(((A2, B2), A1), B1)"][td_index] += 1/2
            else:
                if (tj + td) > tp:
                    topology_counts["((A1, B1), A2)"][td_index] += 1
                    #topology_counts["((A1, B1), B2)"][td_index] += 1/2
                else:
                    #topology_counts["((A1, B2), B1)"][td_index] += 1/4
                    #topology_counts["((B1, B2), A1)"][td_index] += 1/4
                    topology_counts["((A1, A2), B1)"][td_index] += 1/2
                    topology_counts["((A2, B1), A1)"][td_index] += 1/2


for td_index, td in enumerate(td_values):
    for _ in range(num_simulations):
        generate_topology(T, ts, td, topology_counts_D, td_index)

# Normalize the counts for each topology
for topo in topology_counts_D.keys():
    topology_counts_D[topo] /= num_simulations

# Convert results to DataFrame
df_normalized = pd.DataFrame(topology_counts_D)
df_normalized["td"] = td_values

# Apply a sliding window average for smoothing
df_smoothed = df_normalized.copy()
for topology in df_smoothed.columns[:-1]: 
    df_smoothed[topology] = df_smoothed[topology].rolling(window=window_size, min_periods=1).mean()




selected_topology = "((A1, B1), A2)"
selected_counts_smoothed = df_smoothed[selected_topology].values
pdf_td1 = selected_counts_smoothed / np.trapz(selected_counts_smoothed, td_values)  # Normalize to integrate to 1


#relative ratio
blue_curve =np.array([pdf_td[i]/(pdf_td1[i]+pdf_td[i]) for i in range(len(pdf_td))])
orange_curve =np.array([pdf_td1[i]/(pdf_td1[i]+pdf_td[i]) for i in range(len(pdf_td1))])


# Figure 
difference = orange_curve - blue_curve
greater_than = difference > 0
less_than = difference < 0
uncertain = np.abs(difference) < np.max(difference) * 0.3  #shade grey area 
y2 = np.where(td_values < 6, 1, td_values - 5)
# Create figure with two vertical subplots, shared x-axis
fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
ax1.plot(td_values, orange_curve, color='orange', label="1 Duplication")
ax1.plot(td_values, blue_curve, color='blue', label="1 Duplication + 1 Loss")
ax1.axvline(x=5, color='red', linestyle='--')
ax1.fill_between(td_values, blue_curve, orange_curve, where=greater_than, color='orange', alpha=0.3)
ax1.fill_between(td_values, blue_curve, orange_curve, where=less_than, color='blue', alpha=0.3)
ax1.fill_between(td_values, blue_curve, orange_curve, where=uncertain, color='grey', alpha=0.3)
leg= ax1.legend(
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0
)

fig = ax1.figure
fig.canvas.draw()

bbox = leg.get_window_extent().transformed(
    fig.dpi_scale_trans.inverted()
)

# save only the legend
fig.savefig("/N/u/samishr/Quartz/Desktop/figure3_legend.png", bbox_inches=bbox, dpi=300, transparent=True)
ax1.grid(True, linestyle='--', alpha=0.6)
ax2.plot(td_values, y2,color='orange', label='Expected Branch Length')
ax2.axvline(x=5, color='red', linestyle='--')
ax2.set_ylabel('Expected Branch Length', fontsize=10)
ax2.set_yticks(range(0, int(max(y2)) + 1))
ax2.grid(True, linestyle='--', alpha=0.6)
y3 = np.where(td_values < 5, None, td_values - 4)
ax3.plot(td_values, y3,color='blue', label='Expected Branch Length')
ax3.axvline(x=5, color='red', linestyle='--')
ax3.set_xlabel('Time of Duplication', fontsize=14)
ax3.set_ylabel('Expected Branch Length', fontsize=10)
y3 = np.where(td_values <= 5, 0, td_values - 5)
ax3.set_yticks(range(0, int(max(y3)) + 1))
ax3.grid(True, linestyle='--', alpha=0.6)
for ax in [ax1, ax2,ax3]:
    ax.axvline(x=5, color='red', linestyle='--')
plt.tight_layout()
plt.savefig('/N/u/samishr/Quartz/Desktop/figure3.png', dpi=500)
plt.show()