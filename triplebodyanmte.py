#This file attempts to add a z dimension to our previous triplebody file, and add an animation function
#We realized that adding a z dimension not only will make the graphs more informative and realistic but will also supplement animations.
#Also, it shouldn't be too difficult in terms of the theory, according to the sources we read.

###importing necessary packages
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML    # for displaying animation inline
import mpl_toolkits.mplot3d.axes3d as p3

##defining the universal constant for gravity
grav = 6.67430*10**(-11) 

###defining derivative function for 3 body system
def deriv(t_now,r_init, m1 = 0,m2 = 0,m3 = 0):
    """A function that is designed to output derivatives and second derivatives
       with respect to time for the x, y, and z coordinates for a 3 body system
       where the only force is gravity"""    
    # Redefining initial conditions as local variables
    x1 = r_init[0]
    x_dot1 = r_init[1]
    y1 = r_init[2]
    y_dot1 = r_init[3]
    z1 = r_init[4]
    z_dot1 = r_init[5]
    x2 = r_init[6]
    x_dot2 = r_init[7]
    y2 = r_init[8]
    y_dot2 = r_init[9]
    z2 = r_init[10]
    z_dot2 = r_init[11]
    x3 = r_init[12]
    x_dot3 = r_init[13]
    y3 = r_init[14]
    y_dot3 = r_init[15]
    z3 = r_init[16]
    z_dot3 = r_init[17]

    ##defining the difference (length between) between each r vector for a simpler second deravitive equation
    dif12 = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2) + 0.1 ###this 0.1 is what is called a gravitational softener that is added in order
    dif13 = np.sqrt((x1-x3)**2+(y1-y3)**2+(z1-z3)**2) + 0.1 ###make the system more collisionless and keep the inegration time well behaved
    dif23 = np.sqrt((x2-x3)**2+(y2-y3)**2+(z2-z3)**2) + 0.1
    
    # Time derivatives for m1. Solved on paper using Newton's second. (lagrangian was too hard :(  )
    dx1_dt = x_dot1
    d2x1_dt2 = -grav*((m2*(x1-x2))/((dif12)**3)+(m3*(x1-x3))/((dif13)**3))
    dy1_dt = y_dot1
    d2y1_dt2 = -grav*((m2*(y1-y2))/((dif12)**3)+(m3*(y1-y3))/((dif13)**3))
    dz1_dt = z_dot1
    d2z1_dt2 = -grav*((m2*(z1-z2))/((dif12)**3)+(m3*(z1-z3))/((dif13)**3))
    
    # Time derivatives for m2
    dx2_dt = x_dot2
    d2x2_dt2 = -grav*((m3*(x2-x3))/((dif23)**3)+(m1*(x2-x1))/((dif12)**3))
    dy2_dt = y_dot2
    d2y2_dt2 = -grav*((m3*(y2-y3))/((dif23)**3)+(m1*(y2-y1))/((dif12)**3))
    dz2_dt = z_dot2
    d2z2_dt2 = -grav*((m3*(z2-z3))/((dif23)**3)+(m3*(z2-z1))/((dif12)**3))
                    
    # Time derivatives for m3
    dx3_dt = x_dot3
    d2x3_dt2 = -grav*((m2*(x3-x2))/((dif23)**3)+(m1*(x3-x1))/((dif13)**3))
    dy3_dt = y_dot3
    d2y3_dt2 = -grav*((m2*(y3-y2))/((dif23)**3)+(m1*(y3-y1))/((dif13)**3))
    dz3_dt = z_dot3
    d2z3_dt2 = -grav*((m2*(z3-z2))/((dif23)**3)+(m1*(z3-z1))/((dif13)**3))
                    
    return [dx1_dt, d2x1_dt2, dy1_dt, d2y1_dt2, dz1_dt, d2z1_dt2, dx2_dt, d2x2_dt2, dy2_dt, d2y2_dt2, dz2_dt, d2z2_dt2, dx3_dt, d2x3_dt2, dy3_dt, d2y3_dt2, dz3_dt, d2z3_dt2]


def triple(x1 = 0, x_dot1 = 0, y1 = 0, y_dot1 = 0, z1 = 0, z_dot1 = 0, x2 = 0, x_dot2 = 0, y2 = 0, y_dot2 = 0, z2 = 0, z_dot2 = 0,
           x3 = 0, x_dot3 = 0, y3 = 0, y_dot3 = 0, z3 = 0, z_dot3 = 0, m1 = 0, m2 = 0, m3 = 0, t_max = 5): 
    """a function to solve for a numerical solution to the 3body problem to simplify
    main code notebook. All default values are 0 except for time (as having a time 
    array from 0 to 0 makes no sense. So the default time value it 5"""

    r_init = [x1,x_dot1,z1,z_dot1,y1,y_dot1,x2,x_dot2,y2,y_dot2,z2,z_dot2,x3,x_dot3,y3,y_dot3,z3,z_dot3]
    ##setting up the time array for the solve_ivp function
    t_start = 0
    t_end = t_max
    t_span = (t_start,t_end)
    t_arr = np.arange(t_start,t_end,.05)

    # Use solve_ivp to find a numerical solution to three body problem on a plane given m1, m2, m3.
    r_soln = solve_ivp(deriv, t_span, r_init, t_eval = t_arr, rtol = 1e-8, atol = 1e-8, args = (m1,m2,m3))
    
    time = r_soln.t
    x1 = r_soln.y[0]
    dx1 = r_soln.y[1]
    y1 = r_soln.y[2]
    dy1 = r_soln.y[3]
    z1 = r_soln.y[4]
    dz1 = r_soln.y[5]
    x2 = r_soln.y[6]
    dx2 = r_soln.y[7]
    y2 = r_soln.y[8]
    dy2 = r_soln.y[9]
    z2 = r_soln.y[10]
    dz2 = r_soln.y[11]
    x3 = r_soln.y[12]
    dx3 = r_soln.y[13]
    y3 = r_soln.y[14]
    dy3 = r_soln.y[15]
    z3 = r_soln.y[16]
    dz3 = r_soln.y[17]
    
    return np.array([time, x1, dx1, y1, dy1, z1, dz1, x2, dx2, y2, dy2, z2, dz2, x3, dx3, y3, dy3, z3, dz3])

def plot(x1i = 0, x_dot1i = 0, y1i = 0, y_dot1i = 0, z1i = 0, z_dot1i = 0, x2i = 0, x_dot2i = 0, y2i = 0, y_dot2i = 0, z2i = 0, z_dot2i = 0,
           x3i = 0, x_dot3i = 0, y3i = 0, y_dot3i = 0, z3i = 0, z_dot3i = 0, m1 = 0, m2 = 0, m3 = 0, t_max = 5):
    """a function that will take in initial conditions for a three body system and output plots of
    x, y, and z position of each body, position versus velocity for each coordinate and each body, and 
    a plot of the trajectories against eachother."""

    #Using the previously defined triple function to define local variables that we will plot
    time, x1, dx1, y1, dy1, z1, dz1, x2, dx2, y2, dy2, z2, dz2, x3, dx3, y3, dy3, z3, dz3 = triple(x1i,x_dot1i,y1i,y_dot1i,z1i,z_dot1i,x2i,x_dot2i,y2i,y_dot2i,z2i,z_dot2i,x3i,x_dot3i,y3i,y_dot3i,z3i,z_dot3i,m1,m2,m3,t_max)

    #Plotting X/Y/Z vs time for the different bodies
    fig1, axs = plt.subplots(3, 3)
    fig1.suptitle('Plots of X, Y, and Z variables versus t for all three bodies')
    axs[0, 0].plot(time, x1)
    axs[0, 0].set_title('X1 Versus t')
    axs[0, 1].plot(time, y1)
    axs[0, 1].set_title('Y1 Versus t')
    axs[0, 2].plot(time, z1)
    axs[0, 2].set_title('Z1 Versus t')
    axs[1, 0].plot(time, x2)
    axs[1, 0].set_title('X2 Versus t')
    axs[1, 1].plot(time, y2)
    axs[1, 1].set_title('Y2 versus t')
    axs[1, 2].plot(time, z2)
    axs[1, 2].set_title('Z2 versus t')
    axs[2, 0].plot(time, x3)
    axs[2, 0].set_title('X3 versus t')
    axs[2, 1].plot(time, y3)
    axs[2, 1].set_title('Y3 versus t')
    axs[2, 2].plot(time, z3)
    axs[2, 2].set_title('Z3 versus t')
    fig1.tight_layout()

    #Setting labels for the first figure
    axs[0, 0].set(ylabel='position (m)')
    axs[1, 0].set(ylabel='position (m)')
    axs[2, 0].set(xlabel='time (s)',ylabel='position (m)')
    axs[2, 1].set(xlabel='time (s)')
    axs[2, 2].set(xlabel='time (s)')

    #Plotting X/Y/Z versus respective velocities for each body
    fig2, axs = plt.subplots(3, 3)
    fig2.suptitle('Plots of X, Y, and Z variables versus X, Y, and Z velocity for all three bodies')
    axs[0, 0].plot(x1, dx1, marker = ',', linestyle = 'None')
    axs[0, 0].set_title('X1 Versus Vx1')
    axs[0, 1].plot(y1, dy1, marker = ',', linestyle = 'None')
    axs[0, 1].set_title('Y1 Versus Vy1')
    axs[0, 2].plot(z1, dz1, marker = ',', linestyle = 'None')
    axs[0, 2].set_title('Z1 Versus Vz1')
    axs[1, 0].plot(x2, dx2, marker = ',', linestyle = 'None')
    axs[1, 0].set_title('X2 Versus Vx2')
    axs[1, 1].plot(y2, dy2, marker = ',', linestyle = 'None')
    axs[1, 1].set_title('Y2 Versus Vy2')
    axs[1, 2].plot(z3, dz3, marker = ',', linestyle = 'None')
    axs[1, 2].set_title('Z3 Versus Vy3')
    axs[2, 0].plot(x3, dx3, marker = ',', linestyle = 'None')
    axs[2, 0].set_title('X3 Versus Vx3')
    axs[2, 1].plot(y3, dy3, marker = ',', linestyle = 'None')
    axs[2, 1].set_title('Y3 Versus Vy3')
    axs[2, 2].plot(z3, dz3, marker = ',', linestyle = 'None')
    axs[2, 2].set_title('Z3 Versus Vz3')
    fig2.tight_layout()

    #Setting labels for the second figure
    axs[0, 0].set(ylabel='velocity (m/s)')
    axs[1, 0].set(ylabel='velocity (m/s)')
    axs[2, 0].set(xlabel='position (m)',ylabel='velocity (m/s)')
    axs[2, 1].set(xlabel='position (m)')
    axs[2, 2].set(xlabel='position (m)')

    #Plotting the trajectories on one graph
    fig3 = plt.figure().add_subplot(projection='3d')
    plt.title('Plot of Orbits (x vs y vs z)')
    fig3.plot(x1, y1, z1, 'g')
    fig3.plot(x2, y2, z2, 'b')
    fig3.plot(x3, y3, z3, 'm')
    plt.legend(['body 1', 'body 2', 'body 3'])
    fig3.set(xlabel='x position (m)',ylabel='y position (m)',zlabel='z position (m)')

    return fig1,fig2,fig3



def triplebody_animation (x1i = 0, x_dot1i = 0, y1i = 0, y_dot1i = 0, z1i = 0, z_dot1i = 0, x2i = 0, x_dot2i = 0, y2i = 0, y_dot2i = 0, z2i = 0, z_dot2i = 0, x3i = 0, x_dot3i = 0, y3i = 0, y_dot3i = 0, z3i = 0, z_dot3i = 0, m1 = 0, m2 = 0, m3 = 0, t_max = 5):
    """a function that will take in initial conditions for a three body system and output an animation of the 3d plots of the orbits."""

    #Using the previously defined triple function to define local variables that we will plot
    time, x1, dx1, y1, dy1, z1, dz1, x2, dx2, y2, dy2, z2, dz2, x3, dx3, y3, dy3, z3, dz3 = triple(x1i,x_dot1i,y1i,y_dot1i,z1i,z_dot1i,x2i,x_dot2i,y2i,y_dot2i,z2i,z_dot2i,x3i,x_dot3i,y3i,y_dot3i,z3i,z_dot3i,m1,m2,m3,t_max)
    
    time_per_frame = 1.0  # each frame corresponds to a time step of 1 in units
    # Find the number of frames needed by dividing the period by the time per frame
    num_frames = int(time[-1]/time_per_frame)
    dstep = np.max(time)/len(time)  # the integrator time step size
    # Determine how many integrator time steps per frame. If, for example, 
    #    steps_per_frame = 2, only every other data point from the integrator will
    #    be used in the animation
    steps_per_frame = int(time_per_frame/dstep)  # round down to an integer value


    fig, ax = plt.subplots(subplot_kw={'projection': '3d'}) 
    ax.set_title("3-body Animation")
    ### set the limits of the 3d graph
    ax.set_xlim3d([np.array([x1, x2, x3]).min(), np.array([x1, x2, x3]).max()])
    ax.set_ylim3d([np.array([y1, y2, y3]).min(), np.array([y1, y2, y3]).max()])
    ax.set_zlim3d([np.array([z1, z2, z3]).min(), np.array([z1, z2, z3]).max()])


    # Initialize plotting of the positions of m1 m2 and m3 for the animation
    m1_position, = ax.plot([],[],[], 'go', ms = 6)
    m2_position, = ax.plot([],[],[], 'bo', ms = 6)
    m3_position, = ax.plot([],[],[], 'mo', ms = 6)


    # Function used in the FuncAnimation to draw frames
    def animate(i):
        # This makes the plot, by moving data into line's set_data method
        j = int(i*steps_per_frame)  # only plot every steps_per_frame-th step
        ### ADD the positions of m1, m2, and m3 at frame j using set_data
        m1_position.set_data([x1[j],x1[j]], [y1[j],y1[j]])
        m1_position.set_3d_properties([z1[j], z1[j]])
        m2_position.set_data([x2[j],x2[j]], [y2[j],y2[j]])
        m2_position.set_3d_properties([z2[j], z2[j]])
        m3_position.set_data([x3[j],x3[j]], [y3[j],y3[j]])
        m3_position.set_3d_properties([z3[j], z3[j]])
        
        ###and the plots of the tracks going from the beginning to the current j
        x1_track = x1[:j+1]
        y1_track = y1[:j+1]
        z1_track = z1[:j+1]
        m1_track.set_data(x1_track, y1_track)
        m1_track.set_3d_properties(z1_track)

        x2_track = x2[:j+1]
        y2_track = y2[:j+1]
        z2_track = z2[:j+1]
        m2_track.set_data(x2_track, y2_track)
        m2_track.set_3d_properties(z2_track)

        x3_track = x3[:j+1]
        y3_track = y3[:j+1]
        z3_track = z3[:j+1]
        m3_track.set_data(x3_track, y3_track)
        m3_track.set_3d_properties(z3_track)


        return m1_position, m2_position, m3_position, m1_track, m2_track, m3_track

    ###initialize the track plots
    m1_track, = ax.plot([],[],[], 'g', lw = 2)
    m2_track, = ax.plot([],[],[], 'b', lw = 2)  
    m3_track, = ax.plot([],[],[], 'm', lw = 2)  


    # Function to draw the base frame
    def init():
        m1_position.set_data([x1[0],x1[0]], [y1[0],y1[0]])
        m1_position.set_3d_properties([z1[0], z1[0]])
        m2_position.set_data([x2[0],x2[0]], [y2[0],y2[0]])
        m2_position.set_3d_properties([z2[0], z2[0]])
        m3_position.set_data([x3[0],x3[0]], [y3[0], y3[0]])
        m3_position.set_3d_properties([z3[0], z3[0]])
        return m1_position, m2_position, m3_position

    anim = animation.FuncAnimation(fig, animate, frames = num_frames,
                                   init_func = init, blit = True,
                                   interval = 50)  # init_func sets the name of
                                                   #    the function that makes the
                                                   #    base frame
                                                   # blit = True tells the animator
                                                   #    not to redraw unchanged
                                                   #    elements. 
                                                   # interval sets the time between
                                                   #    frames to 50 milliseconds.
    display(HTML(anim.to_jshtml()))  # Display as JSHTML