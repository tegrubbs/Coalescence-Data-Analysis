import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import re
import argparse

# Used to plot bubbles the trajectories over time for a pair of colliding bubbles.


def expand_field_names(fields):
    """
    Expand field names, converting vector fields like 'Center(x,y,z)'
    into separate fields: 'Center_x', 'Center_y', 'Center_z'

    Returns a list of expanded field names
    """
    expanded_fields = []

    for field in fields:
        if '(x,y,z)' in field:
            # Extract base name and create x, y, z variants
            base_name = field.replace('(x,y,z)', '')
            expanded_fields.extend([f"{base_name}_x", f"{base_name}_y", f"{base_name}_z"])
        else:
            expanded_fields.append(field)

    return expanded_fields


def extract_bubble_trajectory(filename, bub1_id, bub2_id):
    """
    Extract trajectory data for a specific bubble pair from coalescence tracking file.

    Parameters:
    -----------
    filename : str
        Path to the coalescenceTracking.dat file
    bub1_id : int
        ID of first bubble
    bub2_id : int
        ID of second bubble

    Returns:
    --------
    pd.DataFrame
        DataFrame containing all timesteps where this bubble pair appears,
        including Time, Center coordinates for both bubbles, and other variables
    """

    trajectory_data = []
    current_timestep = None
    current_time = None

    with open(filename, 'r') as f:
        # Read header line
        header_line = f.readline().strip()
        # Extract field names (split on whitespace)
        fields = header_line.replace('Format:', '').split()

        # Expand field names to account for vector fields
        expanded_fields = expand_field_names(fields)

        print(f"Found {len(fields)} fields in header (compact notation)")
        print(f"Expanded to {len(expanded_fields)} individual values")

        for line in f:
            line = line.strip()

            # Check if this is a timestep header line
            if line.startswith('Timestep:'):
                # Parse: "Timestep:        223  Time:   4.000000E-06  Events:     1"
                match = re.search(r'Timestep:\s+(\d+)\s+Time:\s+([\d.E+-]+)\s+Events:\s+(\d+)', line)
                if match:
                    current_timestep = int(match.group(1))
                    current_time = float(match.group(2))

            # Check if this is a data line (starts with numbers)
            elif line and (line[0].isdigit() or line[0] == '-'):
                # Split the line into values
                values = line.split()

                if len(values) >= len(expanded_fields):
                    # Extract bubble IDs
                    bub1 = int(values[0])
                    bub2 = int(values[1])

                    # Check if this is the pair we're looking for
                    if (bub1 == bub1_id and bub2 == bub2_id) or \
                       (bub1 == bub2_id and bub2 == bub1_id):

                        # Parse all values
                        parsed_values = []
                        for i, val in enumerate(values[:len(expanded_fields)]):
                            try:
                                # Try to parse as float first, then int for IDs
                                if i < 2:  # Bub1_ID and Bub2_ID
                                    parsed_values.append(int(val))
                                else:
                                    parsed_values.append(float(val))
                            except ValueError:
                                parsed_values.append(val)

                        # Create a dictionary for this timestep
                        data_dict = {
                            'Timestep': current_timestep,
                            'Timestep_Time': current_time  # Time from the timestep header
                        }

                        # Add all other fields (including the Time field from the data row)
                        for field_name, value in zip(expanded_fields, parsed_values):
                            data_dict[field_name] = value

                        trajectory_data.append(data_dict)

    # Convert to DataFrame
    df = pd.DataFrame(trajectory_data)

    return df


def plot_trajectories(df, bub1_id, bub2_id):
    """
    Create plots showing the trajectory of two bubbles over time.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with trajectory data
    bub1_id : int
        ID of first bubble
    bub2_id : int
        ID of second bubble
    """

    if len(df) == 0:
        print("No data found for this bubble pair!")
        return

    # Check if we have the bubble position coordinates
    required_cols = ['Bub1_Pos_x', 'Bub1_Pos_y', 'Bub1_Pos_z',
                     'Bub2_Pos_x', 'Bub2_Pos_y', 'Bub2_Pos_z', 'Timestep_Time']

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        print(f"Available columns: {df.columns.tolist()}")
        return

    # Create separate figure for 3D trajectory plot
    fig_3d = plt.figure(figsize=(10, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.plot(df['Bub1_Pos_x'], df['Bub1_Pos_y'], df['Bub1_Pos_z'],
               'b-o', label=f'Bubble {bub1_id}', markersize=4, linewidth=2)
    ax_3d.plot(df['Bub2_Pos_x'], df['Bub2_Pos_y'], df['Bub2_Pos_z'],
               'r-s', label=f'Bubble {bub2_id}', markersize=4, linewidth=2)
    ax_3d.set_xlabel('X Position')
    ax_3d.set_ylabel('Y Position')
    ax_3d.set_zlabel('Z Position')
    ax_3d.set_xlim(-0.005, 0.005)
    ax_3d.set_ylim(0, 0.01)
    ax_3d.set_zlim(0, 0.02)
    # Set aspect ratio based on the axis ranges (1:1:2)
    ax_3d.set_box_aspect([1, 1, 2])
    ax_3d.set_title(f'3D Trajectory: Bubbles {bub1_id} and {bub2_id}')
    ax_3d.legend()
    ax_3d.grid(True)

    # Save 3D plot
    fig_3d.savefig(f'trajectory_3d_bubbles_{bub1_id}_{bub2_id}.png', dpi=150, bbox_inches='tight')
    print(f"3D trajectory plot saved as 'trajectory_3d_bubbles_{bub1_id}_{bub2_id}.png'")

    # Create figure with multiple subplots for 2D plots
    fig = plt.figure(figsize=(15, 10))

    # 1. X coordinate vs Time
    ax2 = fig.add_subplot(2, 3, 1)
    ax2.plot(df['Timestep_Time'], df['Bub1_Pos_x'], 'b-o', label=f'Bubble {bub1_id}', markersize=3)
    ax2.plot(df['Timestep_Time'], df['Bub2_Pos_x'], 'r-s', label=f'Bubble {bub2_id}', markersize=3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('X Position')
    ax2.set_title('X Position vs Time')
    ax2.legend()
    ax2.grid(True)

    # 2. Y coordinate vs Time
    ax3 = fig.add_subplot(2, 3, 2)
    ax3.plot(df['Timestep_Time'], df['Bub1_Pos_y'], 'b-o', label=f'Bubble {bub1_id}', markersize=3)
    ax3.plot(df['Timestep_Time'], df['Bub2_Pos_y'], 'r-s', label=f'Bubble {bub2_id}', markersize=3)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Y Position vs Time')
    ax3.legend()
    ax3.grid(True)

    # 3. Z coordinate vs Time
    ax4 = fig.add_subplot(2, 3, 3)
    ax4.plot(df['Timestep_Time'], df['Bub1_Pos_z'], 'b-o', label=f'Bubble {bub1_id}', markersize=3)
    ax4.plot(df['Timestep_Time'], df['Bub2_Pos_z'], 'r-s', label=f'Bubble {bub2_id}', markersize=3)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Z Position')
    ax4.set_title('Z Position vs Time')
    ax4.legend()
    ax4.grid(True)

    # 4. XY plane trajectory
    ax5 = fig.add_subplot(2, 3, 4)
    ax5.plot(df['Bub1_Pos_x'], df['Bub1_Pos_y'], 'b-o', label=f'Bubble {bub1_id}', markersize=3)
    ax5.plot(df['Bub2_Pos_x'], df['Bub2_Pos_y'], 'r-s', label=f'Bubble {bub2_id}', markersize=3)
    ax5.set_xlabel('X Position')
    ax5.set_ylabel('Y Position')
    ax5.set_title('XY Plane Trajectory')
    ax5.legend()
    ax5.grid(True)
    ax5.axis('equal')

    # 5. Distance between bubbles vs Time (with FilmThickness overlay)
    ax6 = fig.add_subplot(2, 3, 5)
    distance = np.sqrt(
        (df['Bub1_Pos_x'] - df['Bub2_Pos_x'])**2 +
        (df['Bub1_Pos_y'] - df['Bub2_Pos_y'])**2 +
        (df['Bub1_Pos_z'] - df['Bub2_Pos_z'])**2
    )
    ax6.plot(df['Timestep_Time'], distance, 'g-o', markersize=3, label='Distance')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Distance', color='g')
    ax6.tick_params(axis='y', labelcolor='g')
    ax6.grid(True, alpha=0.3)

    # Add FilmThickness on secondary y-axis
    ax6_twin = ax6.twinx()
    ax6_twin.plot(df['Timestep_Time'], df['FilmThickness'], 'm-s', markersize=3, label='FilmThickness')
    ax6_twin.set_ylabel('FilmThickness', color='m')
    ax6_twin.tick_params(axis='y', labelcolor='m')

    ax6.set_title('Distance & FilmThickness vs Time')

    # Combine legends from both axes
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='best')

    fig.tight_layout()
    fig.savefig(f'trajectory_2d_plots_{bub1_id}_{bub2_id}.png', dpi=150, bbox_inches='tight')
    print(f"2D plots saved as 'trajectory_2d_plots_{bub1_id}_{bub2_id}.png'")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Plot trajectories of a bubble pair from coalescence tracking data'
    )
    parser.add_argument('bub1_id', type=int, help='ID of first bubble')
    parser.add_argument('bub2_id', type=int, help='ID of second bubble')
    parser.add_argument('--file', type=str, default='coalescenceTracking.dat',
                        help='Path to coalescence tracking data file (default: coalescenceTracking.dat)')

    args = parser.parse_args()

    print(f"Extracting trajectory for bubble pair ({args.bub1_id}, {args.bub2_id})...")
    df = extract_bubble_trajectory(args.file, args.bub1_id, args.bub2_id)

    if len(df) == 0:
        print(f"\nNo data found for bubble pair ({args.bub1_id}, {args.bub2_id})")
        print("Please check that the bubble IDs are correct.")
        return

    print(f"\nFound {len(df)} timesteps with this bubble pair")
    print(f"Time range: {df['Timestep_Time'].min():.6e} to {df['Timestep_Time'].max():.6e}")
    print(f"Timestep range: {df['Timestep'].min()} to {df['Timestep'].max()}")

    # Display summary statistics
    print("\nSummary statistics:")
    print(f"Bubble {args.bub1_id} position X: {df['Bub1_Pos_x'].min():.6e} to {df['Bub1_Pos_x'].max():.6e}")
    print(f"Bubble {args.bub1_id} position Y: {df['Bub1_Pos_y'].min():.6e} to {df['Bub1_Pos_y'].max():.6e}")
    print(f"Bubble {args.bub1_id} position Z: {df['Bub1_Pos_z'].min():.6e} to {df['Bub1_Pos_z'].max():.6e}")
    print(f"Bubble {args.bub2_id} position X: {df['Bub2_Pos_x'].min():.6e} to {df['Bub2_Pos_x'].max():.6e}")
    print(f"Bubble {args.bub2_id} position Y: {df['Bub2_Pos_y'].min():.6e} to {df['Bub2_Pos_y'].max():.6e}")
    print(f"Bubble {args.bub2_id} position Z: {df['Bub2_Pos_z'].min():.6e} to {df['Bub2_Pos_z'].max():.6e}")

    # Save trajectory data to CSV
    output_file = f'trajectory_bubbles_{args.bub1_id}_{args.bub2_id}.csv'
    df.to_csv(output_file, index=False)
    print(f"\nTrajectory data saved to '{output_file}'")

    # Create plots
    print("\nCreating plots...")
    plot_trajectories(df, args.bub1_id, args.bub2_id)


if __name__ == "__main__":
    main()
