import pandas as pd
import re

"""
This script will process the coalescence tracking data to find the start and end of any bubble collisions. 
This is useful for later analysis of the data, IE, creating histograms of approach angle for coalesced bubbles. 
"""

def expand_field_names(fields):
    """
    Expand field names, converting vector fields like 'Center(x,y,z)'
    into separate fields: 'Center_x', 'Center_y', 'Center_z'

    Returns a list of expanded field names and a mapping of indices
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


def process_coalescence_tracking(filename):
    """
    Process coalescence tracking data to find unique collision events
    and record their start and end points with all 44 variables.

    Parameters:
    -----------
    filename : str
        Path to the coalescenceTracking.dat file

    Returns:
    --------
    pd.DataFrame
        DataFrame containing unique collision events with all variables
        at start and end of each event
    """

    # Dictionary to store collision events
    # Key: (Bub1_ID, Bub2_ID, sequence), Value: dict with all start/end data
    events = {}

    # Track the current sequence number for each bubble pair
    # Key: (Bub1_ID, Bub2_ID), Value: current sequence number
    pair_sequences = {}

    # Track which bubble pairs have completed events
    # Key: (Bub1_ID, Bub2_ID), Value: set of completed sequence numbers
    completed_pairs = {}

    # Track the last-seen status for each event, used to detect the
    # 0 -> 1 transition (start of film drainage)
    # Key: (Bub1_ID, Bub2_ID, sequence), Value: last status seen
    last_status = {}

    current_timestep = None
    current_time = None

    with open(filename, 'r') as f:
        # Read header line
        header_line = f.readline().strip()
        # Extract field names (split on whitespace)
        fields = header_line.replace('Format:', '').split()

        # Expand field names to account for vector fields
        expanded_fields = expand_field_names(fields)
        #print (expanded_fields)
        print(f"Found {len(fields)} fields in header (compact notation)")
        print(f"Expanded to {len(expanded_fields)} individual values")

        for line in f:
            line = line.strip()

            # Check if this is a timestep header line
            if line.startswith('Timestep:'):
                #print(line)
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
                    # Convert values to appropriate types
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

                    # Extract key information
                    bub1_id = int(values[0])
                    bub2_id = int(values[1])

                    # Find status field index in expanded fields
                    status_idx = expanded_fields.index('Status')
                    status = int(float(values[status_idx]))

                    # Film thickness is column 19 (index 18) in the input data
                    film_thickness_idx = expanded_fields.index('FilmThickness')
                    film_thickness = parsed_values[film_thickness_idx]

                    # Create bubble pair identifier (without sequence)
                    pair_key = (bub1_id, bub2_id)

                    # Initialize tracking for this pair if needed
                    if pair_key not in pair_sequences:
                        pair_sequences[pair_key] = 0
                        completed_pairs[pair_key] = set()

                    # Get current sequence number for this pair
                    current_seq = pair_sequences[pair_key]
                    event_key = (bub1_id, bub2_id, current_seq)

                    # Status 0 handling: new approach after a completed event
                    if status == 0 and current_seq in completed_pairs[pair_key]:
                        pair_sequences[pair_key] += 1
                        current_seq = pair_sequences[pair_key]
                        event_key = (bub1_id, bub2_id, current_seq)

                    # If this is a new event, store all start data
                    if event_key not in events:
                        event_data = {
                            'Start_Timestep': current_timestep,
                            'Start_SimTime': current_time,
                            'Event_Sequence': current_seq,  # Track which sequence this is
                            'Event_Completed': False  # Flag to track if end has been recorded
                        }

                        # Store all variables with "Start_" prefix
                        for field_name, value in zip(expanded_fields, parsed_values):
                            event_data[f'Start_{field_name}'] = value

                        events[event_key] = event_data
                        last_status[event_key] = status

                    else:
                        # Only update if we haven't already recorded the end
                        if not events[event_key]['Event_Completed']:
                            # Status flips from 0 (approach) to 1 (film drainage):
                            # record the film thickness at the moment drainage begins
                            if last_status.get(event_key) == 0 and status == 1 \
                                    and 'Drainage_FilmThickness' not in events[event_key]:
                                events[event_key]['Drainage_Timestep'] = current_timestep
                                events[event_key]['Drainage_SimTime'] = current_time
                                events[event_key]['Drainage_FilmThickness'] = film_thickness

                            # Status 2 = bounce, status 3 = actual overlap/coalescence, status 4 = artificial coalescence
                            if status in [2, 3, 4]:
                                events[event_key]['End_Timestep'] = current_timestep
                                events[event_key]['End_SimTime'] = current_time
                                events[event_key]['Event_Completed'] = True
                                completed_pairs[pair_key].add(current_seq)

                                # Store all variables with "End_" prefix
                                for field_name, value in zip(expanded_fields, parsed_values):
                                    events[event_key][f'End_{field_name}'] = value

                                # For bounce events, a bubble position of exactly
                                # (0, 0, 0) means that bubble left the domain
                                if status == 2:
                                    bub1_left = (events[event_key]['End_Bub1_Pos_x'] == 0.
                                                 and events[event_key]['End_Bub1_Pos_y'] == 0.
                                                 and events[event_key]['End_Bub1_Pos_z'] == 0.)
                                    bub2_left = (events[event_key]['End_Bub2_Pos_x'] == 0.
                                                 and events[event_key]['End_Bub2_Pos_y'] == 0.
                                                 and events[event_key]['End_Bub2_Pos_z'] == 0.)
                                    events[event_key]['End_Bub1_Left_Domain'] = bub1_left
                                    events[event_key]['End_Bub2_Left_Domain'] = bub2_left
                                    events[event_key]['End_Bubble_Left_Domain'] = bub1_left or bub2_left

                            last_status[event_key] = status

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(events, orient='index')
    df = df.reset_index(drop=True)

    # Remove the internal tracking flag (keep Event_Sequence as it's useful info)
    if 'Event_Completed' in df.columns:
        df = df.drop('Event_Completed', axis=1)

    # Reorder columns to put Event_Sequence near the beginning
    if 'Event_Sequence' in df.columns:
        cols = df.columns.tolist()
        cols.remove('Event_Sequence')
        # Put Event_Sequence after Start_Timestep
        if 'Start_Timestep' in cols:
            idx = cols.index('Start_Timestep') + 1
            cols.insert(idx, 'Event_Sequence')
        else:
            cols.insert(0, 'Event_Sequence')
        df = df[cols]

    # Calculate duration for completed events using simulation time from Timestep headers
    if 'Start_SimTime' in df.columns and 'End_SimTime' in df.columns:
        df['Duration_Time'] = df.apply(
            lambda row: row['End_SimTime'] - row['Start_SimTime'] if pd.notna(row.get('End_SimTime')) else None,
            axis=1
        )

    if 'Start_Timestep' in df.columns and 'End_Timestep' in df.columns:
        df['Duration_Timesteps'] = df.apply(
            lambda row: row['End_Timestep'] - row['Start_Timestep'] if pd.notna(row.get('End_Timestep')) else None,
            axis=1
        )

    return df


if __name__ == "__main__":

    # Process the data file
    # Change this to point to the coalescence data file output by PHASTA
    filename = "Artificial-Data-Test/PressureDriven-VT/PressureDriven-VT.dat"

    print("Processing coalescence tracking data...")
    df = process_coalescence_tracking(filename)

    print(f"\nFound {len(df)} unique collision events")
    print(f"Total columns in DataFrame: {len(df.columns)}")

    # Count coalesced vs still-in-progress events (bounced pairs are discarded entirely)
    if 'End_Timestep' in df.columns:
        completed = df['End_Timestep'].notna().sum()
        ongoing = df['End_Timestep'].isna().sum()
    else:
        completed = 0
        ongoing = len(df)
    if 'End_Status' in df.columns:
        actual_coal = (df['End_Status'] == 3).sum()
        artificial_coal = (df['End_Status'] == 4).sum()
    else:
        actual_coal = artificial_coal = 0
    print(f"\nCompleted events: {completed}")
    print(f"  Actual coalescence (status 3): {actual_coal}")
    print(f"  Artificial coalescence (status 4): {artificial_coal}")
    print(f"  Bounce (status 2): {completed - actual_coal - artificial_coal}")
    print(f"In-progress events (no endpoint yet): {ongoing}")
    if 'End_Bubble_Left_Domain' in df.columns:
        left_domain = df['End_Bubble_Left_Domain'].fillna(False).sum()
        print(f"  Bounce events where a bubble left the domain: {left_domain}")

    # Show sample of key columns
    print("\nSample of key columns:")
    key_cols = ['Start_Timestep', 'Start_SimTime', 'Event_Sequence', 'Start_Bub1_ID', 'Start_Bub2_ID',
                'Start_Status', 'End_Timestep', 'End_SimTime',
                'End_Status', 'Duration_Time', 'Duration_Timesteps']
    available_cols = [col for col in key_cols if col in df.columns]
    print(df[available_cols].head(10))

    print(f"\nAll columns in DataFrame:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")

    # Save to CSV
    output_file = filename.split('.dat')[0]+"-collision_events.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Also save a summary with just key information
    summary_cols = ['Start_Timestep', 'Start_SimTime', 'Event_Sequence', 'Start_Bub1_ID', 'Start_Bub2_ID',
                    'Start_FilmThickness', 'Start_Status',
                    'Drainage_Timestep', 'Drainage_SimTime', 'Drainage_FilmThickness',
                    'End_Timestep', 'End_SimTime', 'End_FilmThickness', 'End_Status',
                    'End_Bub1_Left_Domain', 'End_Bub2_Left_Domain', 'End_Bubble_Left_Domain',
                    'Duration_Time', 'Duration_Timesteps']
    summary_cols = [col for col in summary_cols if col in df.columns]
    if summary_cols:
        summary_file = filename.split('.dat')[0]+"-collision_events_summary.csv"
        df[summary_cols].to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file}")
