
#This won't work on its own, copy and paste the relevant bits to where you need it

with h5py.File("analysis/analysis.h5", mode='r') as file: #Change this as you need to to get it to open your analysis file
    #This is all just existing  average field plotting code in 2.5D
    plt.figure(figsize=(5, 5), dpi=100)
    num_dims = len(file['tasks']['u'].dims)
    t = np.array(file['tasks']['u'].dims[0]['sim_time'])
    x = np.array(file['tasks']['u'].dims[1][0])
    z = np.array(file['tasks']['u'].dims[num_dims - 1][0])

    duration = min(tot_time, t[-1])
    if duration < average_interval: print('WARNING: averaging interval longer than simulation duration, averaging over entire duration...')
    timeframe_mask = np.logical_and(t >= duration - average_interval, t <= duration)

    t = t[timeframe_mask]

    u = np.squeeze(np.array(file['tasks']['u'])[timeframe_mask])
    v = np.squeeze(np.array(file['tasks']['v'])[timeframe_mask])
    w = np.squeeze(np.array(file['tasks']['w'])[timeframe_mask])

    u_avgt = np.mean(u, axis=0)
    v_avgt = np.mean(v, axis=0)
    w_avgt = np.mean(w, axis=0)


    tstart = duration - average_interval
    tend = duration
    
    fig.suptitle(f'Averaged from {np.round(tstart, 2)} to {np.round(tend, 2)} viscous times')

    #New stuff starts here
    ufft= sp.fft.fft2(u_avgt) #2D fft of the u velocity field
    Rows = sp.fft.fftfreq(ufft.shape[0],d=2) #Ffts of the axes for the graph
    Cols = sp.fft.fftfreq(ufft.shape[1],d=2)
    ufftcenter = sp.fft.fftshift(ufft) #Not in use rn, produces the plot with the fft in the centre
    plt.xlabel("Kx")
    plt.ylabel("Kz")
    plt.title("K space graph")


    graph = plt.pcolormesh(np.abs(Rows), np.abs(Cols), np.abs(ufft.T), cmap="gnuplot", shading='nearest', vmin="0", vmax="40000") #The np.abs is necessary to make the fft results non-complex
    plt.colorbar(graph)
    plt.show
    plt.savefig("average velocity ufft frequencies contourf new log.png") #Name the output whatever you want