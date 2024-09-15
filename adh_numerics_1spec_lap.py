################################################################################
# Name: adh_numerics_1spec_lap.py
#
# Purpose: Numerics for 1 species memory-advection-diffusion model, with periodic
#          boundary conditions, and laplace kernel
#
# Usage: python adh_numerics_1spec_lap.py 100000 100000 200 100 14.14 2 1 1 1 8 adh_ic_gauss_sigma0.05.inp ..\..\..\..\OneDrive\ADH\adh_numerics_1spec_lap_10000_100000_200_100_14.14_2_1_1_1_8.out
################################################################################

import sys, math

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
an = float(sys.argv[curr_arg])
curr_arg += 1
env_freq = float(sys.argv[curr_arg])
curr_arg += 1
tol_exp = float(sys.argv[curr_arg])
curr_arg += 1
infile = open(sys.argv[curr_arg],'r')
curr_arg += 1
outfile = open(sys.argv[curr_arg],'w')

# Tolerence
tol = 10.0**(-tol_exp)

# Box goes from -1 to 1
box_width = 2.0

# Derived parameters
delta_t = 1/float(time_granularity)
delta_x = box_width/float(space_granularity)

# Set up arrays to store distributions and set-up initial conditions
old_u_array = []
new_u_array = []
smooth_u_array = []
u_tot = 0.0

# Flag whether finished
finish = 0

# Set up environment
env = []
diffenv = []
for space in range(space_granularity):
    xx = float(space)*box_width/float(space_granularity)-box_width/2
    if abs(xx*env_freq) <= 1:
        env += [an*(1+math.cos(math.pi*xx*env_freq))]
        diffenv += [-an*(math.pi*env_freq)*math.sin(math.pi*xx*env_freq)]
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
#    old_u_array += [max(old_u_array[counter]+(random.random()-0.5)*wave_amp,0)]
    old_u_array += [float(split_line[counter])]
    new_u_array += [0]
    smooth_u_array += [0]
    u_tot += old_u_array[counter]

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
        for z_val in range(space_granularity):
            #weight = (m_val/2)*math.exp(-m_val*(abs(z_val-space)+0.5)*delta_x)*delta_x
            #weight = (m_val/2)*math.exp(-m_val*abs(z_val-space)*delta_x)*delta_x
            weight = (math.exp(-m_val*abs(z_val-space)*delta_x)-math.exp(-m_val*(abs(z_val-space)+1)*delta_x))/2
            smooth_u_array[space] += old_u_array[z_val]*weight

    for space in range(space_granularity):
        # Change u-values
        new_u_array[space] = old_u_array[space] + (delta_t/(delta_x**2))*(
                                     D_val*(old_u_array[(space+1) % space_granularity]-2*old_u_array[space]+old_u_array[(space-1) % space_granularity]) - 
                                     (((gamma*(smooth_u_array[(space+2) % space_granularity] - smooth_u_array[space])+diffenv[(space+1) % space_granularity]*(2*delta_x))*old_u_array[(space+1) % space_granularity]) - 
                                      ((gamma*(smooth_u_array[space] - smooth_u_array[(space-2) % space_granularity])+diffenv[(space-1) % space_granularity]*(2*delta_x))*old_u_array[(space-1) % space_granularity]))/4) 

    # Ensure no negative values
    for space in range(space_granularity):
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
