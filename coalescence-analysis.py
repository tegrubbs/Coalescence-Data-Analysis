import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

"""
Load a collision events CSV (produced by collision.py) for analysis.
"""

# Global font size applied to all plots (titles, axis labels, tick labels, legends).
FONT_SIZE = 14
plt.rcParams.update({'font.size': FONT_SIZE})


def load_collision_events(filename):
    """
    Load a collision events CSV file into a DataFrame.

    Parameters:
    -----------
    filename : str
        Path to a "*-collision_events.csv" file

    Returns:
    --------
    pd.DataFrame
    """
    df = pd.read_csv(filename)
    return df


def add_relative_velocity(df, prefix):
    """
    Add a column '{prefix}_RelVel_Magnitude' with the magnitude of the
    relative velocity between Bub1 and Bub2 (|Vel1 - Vel2|).
    """
    dvx = df[f'{prefix}_Bub1_Vel_x'] - df[f'{prefix}_Bub2_Vel_x']
    dvy = df[f'{prefix}_Bub1_Vel_y'] - df[f'{prefix}_Bub2_Vel_y']
    dvz = df[f'{prefix}_Bub1_Vel_z'] - df[f'{prefix}_Bub2_Vel_z']
    df[f'{prefix}_RelVel_Magnitude'] = (dvx**2 + dvy**2 + dvz**2)**0.5
    return df


def add_approach_velocity(df, prefix):
    """
    Add a column '{prefix}_ApproachVelocity' with the component of the
    relative velocity (Vel1 - Vel2) along the line connecting the two
    bubbles' centers, i.e. how fast they are closing the gap between them
    (positive = approaching, negative = separating).
    """
    dvx = df[f'{prefix}_Bub1_Vel_x'] - df[f'{prefix}_Bub2_Vel_x']
    dvy = df[f'{prefix}_Bub1_Vel_y'] - df[f'{prefix}_Bub2_Vel_y']
    dvz = df[f'{prefix}_Bub1_Vel_z'] - df[f'{prefix}_Bub2_Vel_z']

    rx = df[f'{prefix}_Bub2_Pos_x'] - df[f'{prefix}_Bub1_Pos_x']
    ry = df[f'{prefix}_Bub2_Pos_y'] - df[f'{prefix}_Bub1_Pos_y']
    rz = df[f'{prefix}_Bub2_Pos_z'] - df[f'{prefix}_Bub1_Pos_z']
    distance = (rx**2 + ry**2 + rz**2)**0.5+1E-10

    df[f'{prefix}_ApproachVelocity'] = (dvx * rx + dvy * ry + dvz * rz) / distance
    return df


def add_bubble_diameter(df, prefix, bub, gas_density):
    """
    Add a column '{prefix}_Bub{bub}_Diameter' estimated from radius.
    """
    df[f'{prefix}_Bub{bub}_Diameter'] = df[f'{prefix}_Bub{bub}_Radius']*2.
    return df


def add_weber_number(df, prefix, liquid_density, sigma):
    """
    Add columns '{prefix}_EquivDiameter' and '{prefix}_Weber' using the
    equivalent (harmonic mean) diameter of Bub1 and Bub2:
        D_eq = 2 * d1 * d2 / (d1 + d2)
        We = liquid_density * RelVel_Magnitude^2 * D_eq / sigma
    """
    d1 = df[f'{prefix}_Bub1_Diameter']
    d2 = df[f'{prefix}_Bub2_Diameter']
    d_eq = 2 * d1 * d2 / (d1 + d2)
    df[f'{prefix}_EquivDiameter'] = d_eq
    df[f'{prefix}_Weber'] = liquid_density * df[f'{prefix}_RelVel_Magnitude']**2 * d_eq / sigma
    df[f'{prefix}_WeberTime'] = liquid_density * df[f'{prefix}_RelVel_Magnitude'] * d_eq**2 / sigma
    return df


def add_event_time(df):
    """
    Add a column 'Event_Time' equal to End_SimTime - Start_SimTime.
    """
    df['Event_Time'] = df['End_SimTime'] - df['Start_SimTime']
    return df


def get_coalescence_events(df):
    """
    Return only the events that ended in actual coalescence (End_Status == 3).
    """
    return df[df['End_Status'] == 3]


def get_bounce_events(df):
    """
    Return only the events that ended in a bounce (End_Status == 2).
    """
    return df[df['End_Status'] == 2]


if __name__ == "__main__":

    # Change this to the collision event file created by collision.py
    filename = "POR-sim/POR-sim-collision_events.csv"

    # The below properties are relevant to calculating the Weber number, drainage time correction, etc.
    gasDensity = 1.225
    sigma = 1./13.74
    liquidDensity = 1000.
    dx = 1E-4

    df = load_collision_events(filename)[:]

    df = add_relative_velocity(df, 'Start')
    #df = add_relative_velocity(df, 'End')

    df = add_approach_velocity(df, 'Start')
    #df = add_approach_velocity(df, 'End')

    for bub in (1, 2):
        df = add_bubble_diameter(df, 'Start', bub, gasDensity)
        #df = add_bubble_diameter(df, 'End', bub, gasDensity)
    df = add_weber_number(df, 'Start', liquidDensity, sigma)
    #df = add_weber_number(df, 'End', liquidDensity, sigma)

    df = add_event_time(df)

  
    coalescence_df = get_coalescence_events(df)
    bounce_df = get_bounce_events(df)

    # Additional added factor is what we subtracted when predicting the drainage time in PHASTA.
    coal_true_Drain_time = coalescence_df['Start_FilmDrainageTime'] - coalescence_df['Start_Time'] + 0.8*dx / coalescence_df['Start_RelVel_Magnitude']
    exactDrainTime =  8.*coalescence_df['Start_WeberTime']
    bounce_true_Drain_time = bounce_df['Start_FilmDrainageTime'] - bounce_df['Start_Time'] + 0.8*dx / bounce_df['Start_RelVel_Magnitude']
  
    '''
    Comparison of actual event times with drainage time estimate.
    '''

    x = coalescence_df['Start_WeberTime'] 
    y_drainage = coal_true_Drain_time
 
    y_event = coalescence_df['Event_Time']
   

    x_fit = np.linspace(x.min(), x.max(), 100)

    event_slope, event_intercept = np.polyfit(x, y_event, 1)
    y_event_fit = event_slope * x_fit + event_intercept

    y_event_pred = event_slope * x + event_intercept
    event_ss_res = np.sum((y_event - y_event_pred)**2)
    event_ss_tot = np.sum((y_event - y_event.mean())**2)
    event_r_squared = 1 - event_ss_res / event_ss_tot

    fig, ax = plt.subplots()
    ax.plot(x, y_drainage, 'o', color='#4C72B0', label='Predicted film drainage time')
    #ax.plot(x, y_drainage_total, 'o', color='#8172B2', label='Film drainage time + dx / end relative velocity')
    ax.plot(x, y_event, 'x', color='#55A868', label='Actual Event time')
    #ax.plot(x_fit, x_fit, '--', color='#808080', label='y = x')
    ax.plot(x_fit, y_event_fit, '-', color='#C44E52',
            label=f'Drainage time fit (y = {event_slope:.3g}x + {event_intercept:.3g}, R² = {event_r_squared:.3g})')
    ax.set_xlabel('WeberTime ($\\rho u D^2 / \\sigma$)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Event Time vs. Weber Time for Coalescence Events')
    ax.legend()


    '''
    Comparison of film thickness at drainage time.
    '''


    fig5, ax5 = plt.subplots()
    bounce_film_thickness = bounce_df['Drainage_FilmThickness'].dropna()/dx
    coalescence_film_thickness = coalescence_df['Drainage_FilmThickness'].dropna()/dx
    film_thickness_bins = np.histogram_bin_edges(
        np.concatenate([bounce_film_thickness, coalescence_film_thickness]),
        bins=30
    )
    ax5.hist(bounce_film_thickness, bins=film_thickness_bins, density=True, alpha=0.6, color='#4C72B0',  label='Bounce events')
    ax5.hist(coalescence_film_thickness, bins=film_thickness_bins, density=True, alpha=0.6, color='#DD8452', label='Coalescence events')
    ax5.axvline(bounce_film_thickness.mean(), linestyle='--', color='#4C72B0',
                label=f'Bounce mean ({bounce_film_thickness.mean():.2f})')
    ax5.axvline(coalescence_film_thickness.mean(), linestyle='--', color='#DD8452',
                label=f'Coalescence mean ({coalescence_film_thickness.mean():.2f})')
    ax5.axvline(coalescence_film_thickness.max(), linestyle=':', color="#E20707",
                label=f'Coalescence max ({coalescence_film_thickness.max():.2f})')

    n_bounce_above_coalescence_max = (bounce_film_thickness > coalescence_film_thickness.max()).sum()
    print(f'Bounce events above coalescence max film thickness: {n_bounce_above_coalescence_max/np.shape(bounce_film_thickness)}')

    #ax5.set_xlabel('Estimated Film Drainage Time-Event Time  (s)')
    ax5.set_xlabel('Ending film thickness (#Elements)')
    ax5.set_title('Film Thickness at Drainage Time')
    ax5.legend()

    '''
    Comparison of actual simulated event time to artificial experimental results.
    '''

    np.random.seed(42)
    x_weber_time = np.random.rand(400)* \
        (coalescence_df['Start_WeberTime'].max()-coalescence_df['Start_WeberTime'].min())+coalescence_df['Start_WeberTime'].min()

    frame_time = 1./3000.
    detection_uncertainty = 10 * frame_time 

    model_sigma = 0.0015*2.
    model_factor = np.random.normal(loc=x_weber_time*8., scale=model_sigma, size=len(x_weber_time)) + \
        np.random.uniform(-detection_uncertainty, detection_uncertainty, size=len(x_weber_time))
    model_factor[model_factor <  0.] = 0.

    y_synthetic = model_factor

    weber_time_slope, weber_time_intercept = np.polyfit(x_weber_time, y_synthetic, 1)
    x_weber_time_fit = np.linspace(x_weber_time.min(), x_weber_time.max(), 100)
    y_weber_time_fit = weber_time_slope * x_weber_time_fit + weber_time_intercept

    y_weber_time_pred = weber_time_slope * x_weber_time + weber_time_intercept
    weber_time_ss_res = np.sum((y_synthetic - y_weber_time_pred)**2)
    weber_time_ss_tot = np.sum((y_synthetic - y_synthetic.mean())**2)
    weber_time_r_squared = 1 - weber_time_ss_res / weber_time_ss_tot

    fig6, ax6 = plt.subplots(figsize=(8,6))

    
   
    ax6.plot(x_weber_time_fit, y_weber_time_fit, '-', color='#DD8452',
             label=f'Synthetic Data fit (y = {weber_time_slope:.3g}x + {weber_time_intercept:.3g}, R² = {weber_time_r_squared:.3g})')
    ax6.set_xlabel('$t_{We}$ (s)')
    ax6.set_ylabel('Event Time (s)')
    # ax6.set_title('Event Time vs. Weber Time for Coalescence Events')

    ax6.plot(x_weber_time, y_synthetic, 'x', color='#55A868', alpha=0.6, label='Synthetic data (y = 8x + noise)')
    ax6.plot(x_fit, y_event_fit, '-', color='#C44E52',
            label=f'Simulated time fit (y = {event_slope:.3g}x + {event_intercept:.3g}, R² = {event_r_squared:.3g})')
    ax6.plot(x, y_event, 'o', markersize=5, color='#4C72B0', label='Simulated Coalescence events')
    ax6.legend(loc='upper left',fontsize=12)

    plt.show()

    
