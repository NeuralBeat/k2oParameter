import os
import pandas as pd 
import numpy as np


def mimic_knock_detection(sequence, params):
    """
    Mimics the knock detection algorithm for a single sequence using provided parameters.
    
    Parameters:
    - sequence: A NumPy array (or similar) representing the acceleration data for a single knock event.
    - params: A dictionary containing parameters for the knock detection algorithm.
    
    Returns:
    - int has_it_knocked: 1 if the sequence is considered a valid knock, 0 otherwise.
    """
    # Load and init alld necessary parameters
    Has_It_Knocked = 0
    ZImpactReturn = False
    Very_High_Impact = False
    DeployFlag = False 
    prebuffer_size = 8
    knock_event_length_avg = 65
    knock_event_ended = prebuffer_size + knock_event_length_avg

    knock_algo_Vel_Zlatched = params['KLOPFALGO_VelThrZLatched_s32']
    knock_algo_Vel_Zlatched_Counter = params['KLOPFALGO_VelLatchedCounterThr_u8']
    knock_algo_HFA_Zlatched = params['KLOPFALGO_HfaThrZLatched_s16']
    knock_algo_HFA_Zlatched_Counter = params['KLOPFALGO_HfaLatchedCounterThr_u8']
    very_high_counter = params['KLOPFALGO_VelVeryHighCounterThr_u8']
    knock_algo_AngleXLatched = params['KLOPFALGO_WinkelThrXLatched_s16'] 

    sequence_df = reshape_sequence(sequence)
    sequence_df = sequence_df*(-1)

    # Calculate the velocities as cumulative sum (integration) of each buffer
    sequence_df['xBuffer_quasi_velocity'] = np.cumsum(sequence_df['xBuffer'])
    sequence_df['yBuffer_quasi_velocity'] = np.cumsum(sequence_df['yBuffer'])
    sequence_df['zBuffer_quasi_velocity'] = np.cumsum(sequence_df['zBuffer'])

    # Calculate the energies of each buffer
    sequence_df['xBuffer_quasi_energy'] = (sequence_df['xBuffer_quasi_velocity'])**2
    sequence_df['yBuffer_quasi_energy'] = (sequence_df['yBuffer_quasi_velocity'])**2
    sequence_df['zBuffer_quasi_energy'] = (sequence_df['zBuffer_quasi_velocity'])**2

    sequence_df['xBuffer_quasi_work'] = np.cumsum(np.abs(sequence_df['xBuffer_quasi_velocity']))
    sequence_df['yBuffer_quasi_work'] = np.cumsum(np.abs(sequence_df['yBuffer_quasi_velocity']))
    sequence_df['zBuffer_quasi_work'] = np.cumsum(np.abs(sequence_df['zBuffer_quasi_velocity']))

    knock_event_ended_dynamic = find_knock_event_ended(sequence_df['zBuffer'], knock_algo_HFA_Zlatched, knock_algo_HFA_Zlatched_Counter, prebuffer_size)

    # Nulling values after the knock_event_ended parameter by setting them to 0
    # Update based on the dynamically determined knock_event_ended value
    if knock_event_ended_dynamic is not None and knock_event_ended_dynamic != 0:
        knock_event_ended = knock_event_ended_dynamic
        sequence_df.loc[knock_event_ended_dynamic+1:, 'xBuffer_quasi_velocity'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'yBuffer_quasi_velocity'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'zBuffer_quasi_velocity'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'xBuffer_quasi_energy'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'yBuffer_quasi_energy'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'zBuffer_quasi_energy'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'xBuffer_quasi_work'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'yBuffer_quasi_work'] = 0
        sequence_df.loc[knock_event_ended_dynamic+1:, 'zBuffer_quasi_work'] = 0

    else:
        sequence_df.loc[knock_event_ended+1:, 'xBuffer_quasi_velocity'] = 0
        sequence_df.loc[knock_event_ended+1:, 'yBuffer_quasi_velocity'] = 0
        sequence_df.loc[knock_event_ended+1:, 'zBuffer_quasi_velocity'] = 0
        sequence_df.loc[knock_event_ended+1:, 'xBuffer_quasi_energy'] = 0
        sequence_df.loc[knock_event_ended+1:, 'yBuffer_quasi_energy'] = 0
        sequence_df.loc[knock_event_ended+1:, 'zBuffer_quasi_energy'] = 0
        sequence_df.loc[knock_event_ended+1:, 'xBuffer_quasi_work'] = 0
        sequence_df.loc[knock_event_ended+1:, 'yBuffer_quasi_work'] = 0
        sequence_df.loc[knock_event_ended+1:, 'zBuffer_quasi_work'] = 0


    #Calculate several keay values
    maxZ_Acc = np.abs(sequence_df['zBuffer']).max()
    maxX_Vel = np.abs(sequence_df['xBuffer_quasi_velocity']).max()
    maxY_Vel = np.abs(sequence_df['yBuffer_quasi_velocity']).max()
    maxZ_Vel = np.abs(sequence_df['zBuffer_quasi_velocity']).max()
    minZ_Vel = sequence_df['zBuffer_quasi_velocity'].min()
    
    divXZ = maxZ_Vel/maxX_Vel
    divYZ = maxZ_Vel/maxY_Vel

    ZImpactReturn = calculate_ZImpactReturn(maxZ_Vel, maxX_Vel, maxY_Vel, minZ_Vel, knock_algo_AngleXLatched)
    Very_High_Impact = calculate_Very_High_Impact(sequence_df['zBuffer'], sequence_df['zBuffer_quasi_velocity'], very_high_counter)
    DeployFlag = calculate_DeployFlag(sequence_df['zBuffer'], sequence_df['zBuffer_quasi_velocity'], prebuffer_size, knock_event_ended, knock_algo_HFA_Zlatched_Counter, knock_algo_Vel_Zlatched_Counter, knock_algo_HFA_Zlatched, knock_algo_Vel_Zlatched)
    Has_It_Knocked = calculate_Has_It_Knocked(DeployFlag[0], Very_High_Impact, ZImpactReturn)

    return Has_It_Knocked  # Default outcome if no knock is detected


# Implement the dynamic knock_event_ended determination
def find_knock_event_ended(zBuffer, threshold, counter_threshold, prebuffer_size):
     event_counter = 0  # Initialize the counter
 
     # Convert zBuffer to absolute values for comparison
     refBuffer = np.abs(zBuffer)
     refBuffer_exceedings = (refBuffer[prebuffer_size:] > threshold).sum()

     # Iterate through each item starting from prebuffer_size
     for index in range(prebuffer_size, len(refBuffer)):
         if refBuffer[index] > threshold:
             event_counter += 1  # Increment counter if value exceeds threshold
             
             if event_counter > counter_threshold:
                 return index+prebuffer_size # Return the current index if counter exceeds counter_threshold
        
     return None

def calculate_ZImpactReturn(maxZ_Vel, maxX_Vel, maxY_Vel, minZ_Vel, knock_algo_AngleXLatched):
    
    result = False

    conditionVelZMinMax = (maxZ_Vel > (minZ_Vel * (-1))) or (maxZ_Vel < maxY_Vel) or (3*maxZ_Vel < 2*maxX_Vel)

    if conditionVelZMinMax:
        result = False

    conditionXZ = (2 * maxZ_Vel > 3 * maxX_Vel)
    conditionYZ = (3 * maxZ_Vel > knock_algo_AngleXLatched * maxY_Vel)
   
    # Evaluate the condition for ZImpactReturn
    if conditionXZ and conditionYZ:
        result = True

    print('Quotient Z/X is big enough: ', conditionXZ,  ' ---> For positive Recognition required: True')
    print('Quotient Z/Y is big enough: ', conditionYZ,  ' ---> For positive Recognition required: True')
    print('ZImpactReturn: ', result,    '              ---> For positive Recognition required: True')
    return result

##### Calculate Very_High_Impact & update flag
def calculate_Very_High_Impact(zBuffer, xBuffer_quasi_velocity, very_high_counter):
    
    result = False
    # Convert to absolute values
    abs_zBuffer = np.abs(zBuffer)
    abs_xBuffer_quasi_velocity = np.abs(xBuffer_quasi_velocity)
    
    # (1) Check if more than 5 values in zBuffer are greater than 32767
    condition1 = np.sum(abs_zBuffer > 32767) > very_high_counter
    
    # (2) Check if more than 5 values in xBuffer_quasi_velocity are greater than 16*11250
    condition2 = np.sum(abs_xBuffer_quasi_velocity > (16 * 11250)) > very_high_counter
    

    if condition1 or condition2:
        result = True
    print('Very_High_Acc_Impact: ', condition1, '       ---> For positive Recognition required: False')
    print('Very_High_Vel_Impact: ', condition2, '       ---> For positive Recognition required: False')
    print('Very_High_Impact: ', result, '           ---> For positive Recognition required: False')

    return result

##### Calculate DeployFlag & update flag
def calculate_DeployFlag(zBuffer, zBuffer_quasi_velocity, prebuffer_size, knock_event_ended, Acc_Latched_Counter, Vel_Latched_Counter, Acc_Latched_Threshold, Vel_Latched_Threshold):
    
    result = False
    # Slice the input Series to the specified range
    zBuffer_sliced = zBuffer[prebuffer_size:knock_event_ended]
    zBuffer_quasi_velocity_sliced = zBuffer_quasi_velocity[prebuffer_size:knock_event_ended]
    
    # Convert to absolute values
    zBuffer_abs = np.abs(zBuffer_sliced)
    zBuffer_quasi_velocity_abs = np.abs(zBuffer_quasi_velocity_sliced)
    
    # Count values less than 8500
    count_zBuffer = np.sum(zBuffer_abs < Acc_Latched_Threshold)
    
    count_zBuffer_quasi_velocity = np.sum(zBuffer_quasi_velocity_abs < Vel_Latched_Threshold)
    

    # Determine DeployFlag
    if count_zBuffer > Acc_Latched_Counter or count_zBuffer_quasi_velocity > Vel_Latched_Counter:
        result = False
    
    else:
        result = True
    
    print('Acc_Latched_Counter: ', count_zBuffer, '           ---> For positive Recognition required: <', Acc_Latched_Counter)
    print('Vel_Latched_Counter: ', count_zBuffer_quasi_velocity, '            ---> For positive Recognition required: <', Vel_Latched_Counter)
    print('DeployFlag: ', result,  '                 ---> For positive Recognition required: True')
    return [result, count_zBuffer, count_zBuffer_quasi_velocity]

##### Calculate Has_It_Knocked & update flag
def calculate_Has_It_Knocked(DeployFlag, Very_High_Impact, ZImpactReturn):

    result = False

    if DeployFlag and not Very_High_Impact and ZImpactReturn:
        result = True

    print('Knock Detected: ', result, '             ---> If this flag is True, then the knock has been detected')
    return result


def reshape_sequence(sequence):
    """
    Reshapes a flattened sequence array into a structured array with separate columns for x, y, z.
    
    Parameters:
    - sequence: A flattened array of shape (270,) with the order x,y,z,x,y,z,...
    
    Returns:
    - DataFrame with columns ['xBuffer', 'yBuffer', 'zBuffer']
    """
    # Reshape the sequence to have 3 columns
    reshaped_sequence = sequence.reshape(-1, 3)
    
    # Create a DataFrame
    df = pd.DataFrame(reshaped_sequence, columns=['xBuffer', 'yBuffer', 'zBuffer'])
    
    return df
