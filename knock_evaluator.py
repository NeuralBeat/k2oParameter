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
    has_it_knocked = 0
    ZImpactReturn = False
    Very_High_Impact = False
    DeployFlag = False 
    prebuffer_size = 8
    knock_event_length_avg = 65
    knock_event_ended = prebuffer_size + knock_event_length_avg

    enable_threshold = params['KLOPFALGO_EnableThr_s16']
    knock_algo_Vel_Zlatched = params['KLOPFALGO_VelThrZLatched_s32']
    knock_algo_Vel_Zlatched_Counter = params['KLOPFALGO_VelLatchedCounterThr_u8']
    knock_algo_HFA_Zlatched = params['KLOPFALGO_HfaThrZLatched_s16']
    knock_algo_HFA_Zlatched_Counter = params['KLOPFALGO_HfaLatchedCounterThr_u8']
    very_high_counter = params['KLOPFALGO_VelVeryHighCounterThr_u8']
    knock_algo_AngleXLatched = params['KLOPFALGO_WinkelThrXLatched_s16'] 
    knock_algo_AngleYLatched = params['KLOPFALGO_WinkelThrYLatched_s16'] 
    knock_algo_AngleZLatched = params['KLOPFALGO_WinkelThrZLatched_s16']



    return has_it_knocked  # Default outcome if no knock is detected


# Implement the dynamic knock_event_ended determination
def find_knock_event_ended(zBuffer, threshold, counter_threshold, prebuffer_size):
     event_counter = 0  # Initialize the counter
 
     # Convert zBuffer to absolute values for comparison
     refBuffer = np.abs(zBuffer)
     refBuffer_exceedings = (refBuffer[prebuffer_size:] > threshold).sum()

     print(refBuffer_exceedings)
     # Iterate through each item starting from prebuffer_size
     for index in range(prebuffer_size, len(refBuffer)):
         if refBuffer[index] > threshold:
             event_counter += 1  # Increment counter if value exceeds threshold
             
             if event_counter > counter_threshold:
                 return index+prebuffer_size # Return the current index if counter exceeds counter_threshold
        
     return None