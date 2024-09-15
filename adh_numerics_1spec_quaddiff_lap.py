################################################################################
# Name: adh_numerics_1spec_quaddiff_lap.py
#
# Purpose: Numerics for 1 species memory-advection-diffusion model, with periodic
#          boundary conditions, quadratic diffusion, and a laplace kernel
#
# Usage: python adh_numerics_1spec_quaddiff_lap.py 100000 200000 200 1000 10 2 1 1 1 8 adh_ic_sigma0.1_gamma2_p1.inp 
#            ..\..\..\..\onedrive\adh\adh_1spec_qdl_100000_200000_200_1000_10_2_1_1_1_8.out
################################################################################

import sys, math #,random

# Get parameters from command line
curr_arg = 1
max_time = int(sys.argv[curr_arg])
curr_arg += 1
time_granularity = int(sys.argv[curr_arg])
curr_arg += 1
space_granularity = int(sys.argv[curr_arg])
curr_arg += 1
out_time_res = int(sys.argv[curr_arg])
curr_arg += 1
m_val = float(sys.argv[curr_arg])
curr_arg += 1
gamma = float(sys.argv[curr_arg])
curr_arg += 1
D_val = float(sys.argv[curr_arg])
curr_arg += 1
a1 = float(sys.argv[curr_arg]) # magnitude of environmental effect
curr_arg += 1
env_freq = float(sys.argv[curr_arg]) # frequency of environmental oscillations
curr_arg += 1
tol_exp = float(sys.argv[curr_arg]) # tolerence exponent at which to stop numerics
curr_arg += 1
infile = open(sys.argv[curr_arg],'r')
curr_arg += 1
outfile = open(sys.argv[curr_arg],'w')

# Tolerence
tol = 10.0**(-tol_exp)

# Set up arrays to store distributions and set-up initial conditions
old_u_array = []
new_u_array = []
smooth_u_array = []
flux = []
ue = []
uw = []
velocity = []
u_tot = 0.0

# Box goes from -1 to 1
box_width = 2.0

# Derived parameters
delta_t = 1/float(time_granularity)
delta_x = box_width/float(space_granularity)

# Set up environment
env = []
diffenv = []
for space in range(space_granularity):
    xx = float(space)*box_width/float(space_granularity)-box_width/2
    if abs(xx*env_freq) <= 1:
        env += [a1*(1+math.cos(math.pi*xx*env_freq))]
        diffenv += [-a1*(math.pi*env_freq)*math.sin(math.pi*xx*env_freq)]
    else:
        env += [0]
        diffenv += [0]

# Flag whether finished
finish = 0

# Get initial conditions from file
line = infile.readline()
split_line = line.rsplit()
if(len(split_line) != space_granularity):
    sys.stderr.write("Warning: input array of length %i; should be length %i\n" % (len(split_line),space_granularity))

for counter in range(space_granularity):
    old_u_array += [float(split_line[counter])]
    new_u_array += [0]
    smooth_u_array += [0]
    u_tot += old_u_array[counter]
    flux += [0]
    ue += [0]
    uw += [0]
    velocity += [0]

# Normalise    
for counter in range(space_granularity):
    old_u_array[counter] *= space_granularity/(u_tot*box_width)

# Output ICs
for counter in range(space_granularity):
    if counter == space_granularity - 1:
        outfile.write('%f\n' % (old_u_array[counter]))
    else:
        outfile.write('%f\t' % (old_u_array[counter]))

# Loop through time, solving PDE using finite difference method
u_tot = 0.0
for time in range(max_time):
    # Smoothed arrays
    for space in range(space_granularity):
        smooth_u_array[space] = 0
        #kernel_mass = 0
        for z_val in range(space_granularity):
            weight = (m_val/2)*math.exp(-m_val*abs(z_val-space)*delta_x)*delta_x
            smooth_u_array[space] += old_u_array[z_val]*weight

    for space in range(space_granularity):
        # (space-1) and (space+1) with periodic boundaries
        sp_min = (space-1) % space_granularity
        sp_pl = (space+1) % space_granularity
        # Define east and west values for u (ue and uw)
        ue[space] = old_u_array[space] + (old_u_array[sp_pl] - old_u_array[sp_min])/4
        uw[space] = old_u_array[space] - (old_u_array[sp_pl] - old_u_array[sp_min])/4
        if (ue[space]<0) or (uw[space]<0):
            entry1 = old_u_array[sp_pl] - old_u_array[space]
            entry2 = (old_u_array[sp_pl] - old_u_array[sp_min])/4
            entry3 = old_u_array[space] - old_u_array[sp_min]
            if (entry1>0) and (entry2>0) and (entry3>0):
                gradu = min(entry1,entry2,entry3)
            elif (entry1<0) and (entry2<0) and (entry3<0):
                gradu = max(entry1,entry2,entry3)
            else:
                gradu = 0
            ue[space] = old_u_array[space] + gradu
            uw[space] = old_u_array[space] - gradu
            if ue[space]<0:
                sys.stderr.write("WARNING: ue is below zero at position %i\n" % space)
                finish = 1
            if uw[space]<0:
                sys.stderr.write("WARNING: uw is below zero at position %i\n" % space)
                finish = 1
                
        # Define velocity
        velocity[space] = ((D_val*old_u_array[space]-gamma*smooth_u_array[space]-env[space])-
                           (D_val*old_u_array[sp_pl]-gamma*smooth_u_array[sp_pl]-env[sp_pl]))/delta_x
        
        # Define flux at x=space-1/2
        if space > 0:
           flux[sp_min] = max(0, velocity[sp_min])*ue[sp_min]+min(0, velocity[sp_min])*uw[space]
        
        # Define flux at x=space_granularity-1/2
        if space == space_granularity - 1:
           flux[space] = max(0, velocity[space])*ue[space]+min(0, velocity[space])*uw[0]

        # Change u-values at previous spatial location
        if space > 1:
            new_u_array[space-1] = old_u_array[space-1] - (delta_t/delta_x)*(flux[space-1]-flux[space-2])
        
        # Change u-values at 0 and 1 and space_granularity - 1
        if space == space_granularity - 1:
            new_u_array[0] = old_u_array[0] - (delta_t/delta_x)*(flux[0]-flux[space])
            new_u_array[1] = old_u_array[1] - (delta_t/delta_x)*(flux[1]-flux[0])
            new_u_array[space] = old_u_array[space] - (delta_t/delta_x)*(flux[space-1]-flux[space-2])
        
        # Check not gone below zero
        if new_u_array[space] < 0:
            sys.stderr.write("WARNING: gone below zero at position %i\n" % space)
            finish = 1
            break

    if finish == 1:
        # We have gone below zero; break
        break

    # Copy new arrays into old    
    u_tot = 0.0
    finish = 1
    for space in range(space_granularity):
      if abs(old_u_array[space] - new_u_array[space]) > tol:
          finish = 0
      old_u_array[space] = new_u_array[space]
      u_tot += new_u_array[space]
    if time % max(out_time_res,100) == 0:
      sys.stderr.write('time: %i\t%f\n' % (time,u_tot))
    # Put out resulting distribution
    if (time % out_time_res == 0) or (finish == 1):
      for counter in range(space_granularity):
        if counter == space_granularity - 1:
          outfile.write('%f\n' % (new_u_array[counter]))
        else:
          outfile.write('%f\t' % (new_u_array[counter]))

    if finish == 1:
        # Steady state has been reached
        sys.stderr.write("Steady state reached at time step %i\n" % time)
        break
