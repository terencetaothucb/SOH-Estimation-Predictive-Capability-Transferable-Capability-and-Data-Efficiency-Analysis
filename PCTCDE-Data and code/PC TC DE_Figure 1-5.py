#!/usr/bin/env python
# coding: utf-8

# # IC Curve and Feature Extraction

# In[27]:


# CALCE CS2_35 IC curve
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter

folder_path = r'\CS2_35'

files = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
files.sort(key=lambda x: x[1])

sheet_names = ['Channel_1-006', 'Channel_1-008']
colors = cm.Blues(np.linspace(0.2, 1, len(files)))
# Global font settings
plt.rcParams["font.family"] = ["Times New Roman", "serif"]  
# Create a dQ/dV plot
plt.figure(figsize=(3, 2), dpi=600)

for file_idx, (file, _) in enumerate(files):
    file_path = os.path.join(folder_path, file)
    data = None
    
    # Read the data table
    for sheet_name in sheet_names:
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            break
        except:
            continue
    
    if data is None:
        continue
    
    # Filter valid loops
    data = data[data['Cycle_Index'] >= 2]
    
    # Process each loop
    for cycle_num, cycle_data in data.groupby('Cycle_Index'):
        filtered_data = cycle_data[cycle_data['Step_Index'] == 2]
        
        if not filtered_data.empty:
            # Extract raw data
            t = filtered_data['Test_Time(s)'].values
            Q = filtered_data['Charge_Capacity(Ah)'].values
            Q = Q - Q[0]  
            V = filtered_data['Voltage(V)'].values
            
            # Data smoothing parameters
            window_length = 21  
            polyorder = 3       
            
            # Skip loops with insufficient data
            if len(Q) < window_length:
                continue
            
            try:
                # Savitzky-Golay
                Q_smooth = savgol_filter(Q, window_length, polyorder)
                V_smooth = savgol_filter(V, window_length, polyorder)
                
                # Calculate the derivative
                dQ_dt = np.gradient(Q_smooth, t)
                dV_dt = np.gradient(V_smooth, t)
                dQ_dV = dQ_dt / dV_dt
                
                # Outlier handling
                dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)
                
                # Draw IC curves
                plt.plot(V_smooth, dQ_dV, 
                        linewidth=1, 
                        color=colors[file_idx],
                        alpha=0.9)
                
            except Exception as e:
                print(f"Error processing {file} Cycle {cycle_num}: {str(e)}")

plt.title('IC Curves in CALCE_Source', fontsize=11)
plt.xlabel('Voltage (V)', fontsize=11)
plt.ylabel('dQ/dV (Ah/V)', fontsize=11)
plt.tight_layout()
plt.show()


# In[ ]:


# CS2_33 Feature Extraction
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter, find_peaks

# Set the path and parameters
folder_path = r"CS2_33'  
sheet_names = ['Channel_1-006', 'Channel_1-008']

# Get the files and sort them by time
files = [(f, os.path.getmtime(os.path.join(folder_path, f))) 
         for f in os.listdir(folder_path) if f.endswith('.xlsx')]
files.sort(key=lambda x: x[1])

colors = cm.Blues(np.linspace(0.2, 1, len(files)))
features = []

plt.figure(figsize=(4, 3), dpi=600)

for file_idx, (file, _) in enumerate(files):
    file_path = os.path.join(folder_path, file)
    data = None

    for sheet_name in sheet_names:
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            break
        except:
            continue

    if data is None:
        continue

    data = data[data['Cycle_Index'] >= 2]

    for cycle_num, cycle_data in data.groupby('Cycle_Index'):
        filtered_data = cycle_data[cycle_data['Step_Index'] == 2]
        if not filtered_data.empty:
            t = filtered_data['Test_Time(s)'].values
            Q = filtered_data['Charge_Capacity(Ah)'].values
            V = filtered_data['Voltage(V)'].values
            Q = Q - Q[0]

            # Smoothing processing
            window_size_Q = min(len(Q), 21)
            if window_size_Q % 2 == 0:
                window_size_Q -= 1
            polyorder_Q = min(3, window_size_Q - 1)

            window_size_V = min(len(V), 21)
            if window_size_V % 2 == 0:
                window_size_V -= 1
            polyorder_V = min(3, window_size_V - 1)

            try:
                Q_smooth = savgol_filter(Q, window_size_Q, polyorder_Q)
                V_smooth = savgol_filter(V, window_size_V, polyorder_V)

                dQ_dt = np.gradient(Q_smooth, t)
                dV_dt = np.gradient(V_smooth, t)
                dQ_dV = np.nan_to_num(dQ_dt / dV_dt, posinf=0, neginf=0)

                window_size = min(15, len(dQ_dV) // 2 * 2 - 1)
                polyorder_dQ_dV = min(3, window_size - 1)
                if window_size > 5:
                    dQ_dV_filtered = savgol_filter(dQ_dV, window_size, polyorder_dQ_dV)
                else:
                    dQ_dV_filtered = dQ_dV

                # Feature Extraction
                if len(dQ_dV_filtered) > 0:
                    peaks, _ = find_peaks(dQ_dV_filtered, distance=10)

                    if len(peaks) >= 1:
                        peak_values = dQ_dV_filtered[peaks]
                        sorted_indices = np.argsort(peak_values)[-2:]
                        peak_indices = peaks[sorted_indices]
                        peak_indices = peak_indices[np.argsort(peak_indices)]

                        idx1 = peak_indices[0]
                        idx2 = peak_indices[1] if len(peak_indices) > 1 else idx1

                        first_peak = dQ_dV_filtered[idx1]
                        voltage_at_first = V_smooth[idx1]
                        second_peak = dQ_dV_filtered[idx2]
                        voltage_at_second = V_smooth[idx2]
                    else:
                        first_peak = voltage_at_first = second_peak = voltage_at_second = 0

                    mask = (V_smooth >= 3.75) & (V_smooth <= 4.0)
                    area = np.trapz(dQ_dV_filtered[mask], V_smooth[mask]) if np.any(mask) else 0

                    avg_gradient = np.mean(dQ_dV_filtered[:idx1]) if idx1 > 1 else 0

                    features.append({
                        'File': file,
                        'Cycle': cycle_num,
                        'First_Peak_dQdV': round(first_peak, 4),
                        'Voltage_at_First': round(voltage_at_first, 4),
                        'Second_Peak_dQdV': round(second_peak, 4),
                        'Voltage_at_Second': round(voltage_at_second, 4),
                        'Area_3.75_4.0': round(area, 4),
                        'Avg_Gradient_to_First': round(avg_gradient, 4)
                    })

                # Plot IC curves
                plt.plot(V_smooth, dQ_dV_filtered, linewidth=1, color=colors[file_idx], alpha=0.9)

            except Exception as e:
                print(f"Error processing {file} Cycle {cycle_num}: {str(e)}")

# Save Features
df_features = pd.DataFrame(features)
#df_features.to_csv('dQdV_extracted_features.csv', index=False)

# Show Figure
plt.title('dQ/dV Curves for Each Cycle', fontsize=10)
plt.xlabel('Voltage (V)', fontsize=10)
plt.ylabel('dQ/dV (Ah/V)', fontsize=10)
plt.tight_layout()
plt.show()


# In[1]:


# CY25-05_1-#10 IC curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter

file_path = r'TJU_Data\CY25-05_1-#10.csv'

# Global font settings
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  

def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):
    data_smoothed = np.copy(data)
    n = len(data_smoothed)
    
    if n < window_size:
        return data_smoothed  
    
    # Mark outliers that are beyond the range
    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)
    
    # Handle each outlier
    for i in np.where(outliers)[0]:
        # Determine the window range
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)  
        
        window = data_smoothed[start:end]
        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
        
        if len(window_non_outlier) > 0:
            data_smoothed[i] = np.mean(window_non_outlier)
        else:
            expand = 2
            while len(window_non_outlier) == 0 and (start - expand >= 0 or end + expand < n):
                new_start = max(0, i - window_size - expand)
                new_end = min(n, i + window_size + 1 + expand)
                window = data_smoothed[new_start:new_end]
                window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                expand += 1
            
            if len(window_non_outlier) > 0:
                data_smoothed[i] = np.mean(window_non_outlier)
            else:
                data_smoothed[i] = np.median(data_smoothed)
    
    # Secondary smoothing:ensure a natural transition
    if len(data_smoothed) >= 5:
        data_smoothed = savgol_filter(data_smoothed, 5, 2)
    
    return data_smoothed

try:
    data = pd.read_csv(file_path)
    print(f"Data successfully read，totally{len(data)}rows")
except Exception as e:
    print(f"Failed to read file: {e}")
    exit()

# Filter charging data
charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]

cycles = sorted(charge_data['cycle number'].unique())
print(f"Find{len(cycles)}cycles")

# Set the color mapping
colors = cm.Blues(np.linspace(0.2, 1, len(cycles)))

# Create figure
plt.figure(figsize=(3, 2), dpi=600)

# Data smoothing parameter
window_length = 21  
polyorder = 3       

# dQ threshold
dq_threshold = 1e-5

# Outlier range threshold
upper_limit = 8    
lower_limit = -1e-2 

# Filter condition parameters
voltage_low = 3.8   
voltage_high = 4.3  
dqdv_cutoff = 2.5     

# Process each loop
for i, cycle_num in enumerate(cycles):
    filtered_data = charge_data[charge_data['cycle number'] == cycle_num]
    
    if not filtered_data.empty:
        t = filtered_data['time/s'].values
        Q = filtered_data['Q charge/mA.h'].values / 1000  
        Q = Q - Q[0]  
        V = filtered_data['Ecell/V'].values
        
        if len(Q) < window_length:
            print(f"Cycle {cycle_num} insufficient data{window_length}，skip")
            continue
        
        try:
            cumulative_dq = Q  
            valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
            
            if len(valid_indices) == 0:
                print(f"Cycle {cycle_num} all dQ less than threshold，skip")
                continue
                
            cutoff_idx = valid_indices[-1] + 1
            t = t[:cutoff_idx]
            Q = Q[:cutoff_idx]
            V = V[:cutoff_idx]
            
            # Savitzky-Golay
            Q_smooth = savgol_filter(Q, window_length, polyorder)
            V_smooth = savgol_filter(V, window_length, polyorder)
            
            # Calculate the derivative
            dQ_dt = np.gradient(Q_smooth, t)
            dV_dt = np.gradient(V_smooth, t)
            
            dV_dt[dV_dt == 0] = 1e-3
            dQ_dV = dQ_dt / dV_dt
            
            # Outlier handling
            dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)
            
            dQ_dV_smoothed = replace_extreme_values(
                dQ_dV, 
                upper_threshold=upper_limit, 
                lower_threshold=lower_limit
            )
            
            voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
            keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)

            V_filtered = V_smooth[keep_mask]
            dQ_dV_filtered = dQ_dV_smoothed[keep_mask]

            if len(V_filtered) > 10: 
                # 绘制曲线
                plt.plot(V_filtered, dQ_dV_filtered, 
                        linewidth=1, 
                        color=colors[i],
                        alpha=0.9,
                        label=f'Cycle {cycle_num}' if i % 5 == 0 else "")
            else:
                print(f"Cycle {cycle_num} Too few data points after filtering, skipped mapping")
            
        except Exception as e:
            print(f"Error {e} handling cycle :{cycle_num} ")
            continue

plt.xlabel('Voltage (V)', fontsize=11)
plt.ylabel('dQ/dV (Ah/V)', fontsize=11)
plt.title('IC Curves in TJU_Source', fontsize=11)
#plt.grid(True, linestyle='--', alpha=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')
plt.ylim(-0.5, 1e1)
plt.tight_layout()
plt.show()  


# In[62]:


# Feature Extraction CY25-05_1 Single Battery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapz

file_path = r'TJU_Data\CY25-05_1-#13.csv'

# Global font settings
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False

def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):

    data_smoothed = np.copy(data)
    n = len(data_smoothed)
    
    if n < window_size:
        return data_smoothed
    
    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)
    
    for i in np.where(outliers)[0]:
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = data_smoothed[start:end]
        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
        
        if len(window_non_outlier) > 0:
            data_smoothed[i] = np.mean(window_non_outlier)
        else:
            expand = 2
            while len(window_non_outlier) == 0 and (start - expand >= 0 or end + expand < n):
                new_start = max(0, i - window_size - expand)
                new_end = min(n, i + window_size + 1 + expand)
                window = data_smoothed[new_start:new_end]
                window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                expand += 1
            
            if len(window_non_outlier) > 0:
                data_smoothed[i] = np.mean(window_non_outlier)
            else:
                data_smoothed[i] = np.median(data_smoothed)
    
    if len(data_smoothed) >= 5:
        data_smoothed = savgol_filter(data_smoothed, 5, 2)
    
    return data_smoothed

try:
    data = pd.read_csv(file_path)
except Exception as e:
    exit()

# Filter charging data
charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]
cycles = sorted(charge_data['cycle number'].unique())


features = []

# Parameter settings
window_length = 21
polyorder = 3
dq_threshold = 1e-5
upper_limit = 8
lower_limit = -1e-2
voltage_low = 3.8
voltage_high = 4.3
dqdv_cutoff = 2.5

peak1_params = {
    'height': 0.4,  
    'distance': 5,  
    'prominence': 0.01 
}

peak2_params = {
    'height': 0.4, 
    'distance': 8,  
    'prominence': 0.1 
}
# Process each loop and extract features
for i, cycle_num in enumerate(cycles):
    filtered_data = charge_data[charge_data['cycle number'] == cycle_num]
    
    if not filtered_data.empty:
        t = filtered_data['time/s'].values
        Q = filtered_data['Q charge/mA.h'].values / 1000
        Q = Q - Q[0]
        V = filtered_data['Ecell/V'].values
        
        if len(Q) < window_length:
            continue
        
        try:
            cumulative_dq = Q
            valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
            
            if len(valid_indices) == 0:
                continue
                
            cutoff_idx = valid_indices[-1] + 1
            t = t[:cutoff_idx]
            Q = Q[:cutoff_idx]
            V = V[:cutoff_idx]
            
            # Smoothing processing
            Q_smooth = savgol_filter(Q, window_length, polyorder)
            V_smooth = savgol_filter(V, window_length, polyorder)
            
            # Calculate dQ/dV
            dQ_dt = np.gradient(Q_smooth, t)
            dV_dt = np.gradient(V_smooth, t)
            dV_dt[dV_dt == 0] = 1e-3
            dQ_dV = dQ_dt / dV_dt
            dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)
            dQ_dV_smoothed = replace_extreme_values(dQ_dV, upper_limit, lower_limit)
            
            # Data filtering
            voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
            keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)
            V_filtered = V_smooth[keep_mask]
            dQ_dV_filtered = dQ_dV_smoothed[keep_mask]
            
            if len(V_filtered) <= 10:
                continue
            
            sorted_idx = np.argsort(V_filtered)
            V_sorted = V_filtered[sorted_idx]
            dQ_dV_sorted = dQ_dV_filtered[sorted_idx]

            peak1_voltage = np.nan
            peak1_value = np.nan

            mask_36 = V_sorted > 3.6
            if np.sum(mask_36) > 0:
                v_36 = V_sorted[mask_36]
                dqdv_36 = dQ_dV_sorted[mask_36]

                peaks_36, _ = find_peaks(dqdv_36, **peak1_params)

                if len(peaks_36) > 0:
                    first_peak_idx = peaks_36[0] 
                    peak1_voltage = v_36[first_peak_idx]
                    peak1_value = dqdv_36[first_peak_idx]

            peak2_voltage = np.nan
            peak2_value = np.nan

            mask_40 = (V_sorted > 4.0) & (V_sorted < 4.15)
            if np.sum(mask_40) > 0:
                v_40 = V_sorted[mask_40]
                dqdv_40 = dQ_dV_sorted[mask_40]

                peaks_40, _ = find_peaks(dqdv_40,** peak2_params)

                if len(peaks_40) > 0:
                    first_peak_idx = peaks_40[0]  
                    peak2_voltage = v_40[first_peak_idx]
                    peak2_value = dqdv_40[first_peak_idx]

            area = np.nan
            area_mask = (V_sorted >= 3.4) & (V_sorted <= 4.15)
            if np.sum(area_mask) >= 2:  
                v_area = V_sorted[area_mask]
                dqdv_area = dQ_dV_sorted[area_mask]
                area = trapz(dqdv_area, v_area)  
            else:
                print("error")

            features.append({
                'cycle_number': cycle_num,
                'peak1_voltage(>3.6V)': peak1_voltage,
                'peak1_dqdv': peak1_value,
                'peak2_voltage(>4.0V)': peak2_voltage,
                'peak2_dqdv': peak2_value,
                'area_3.7-4.0V': area
            })

            plt.plot(V_sorted, dQ_dV_sorted, linewidth=1, alpha=0.7, label=f'Cycle {cycle_num}')
            if not np.isnan(peak1_voltage):
                plt.scatter(peak1_voltage, peak1_value, color='red', s=30, zorder=5)  # 第一个峰标记为红色
            if not np.isnan(peak2_voltage):
                plt.scatter(peak2_voltage, peak2_value, color='green', s=30, zorder=5)  # 第二个峰标记为绿色
            
        except Exception as e:
            print(f"Error {e} handling cycle :{cycle_num} ")
            continue

features_df = pd.DataFrame(features)
print(features_df)
#features_df.to_csv('\TJU_CY25_05_1_#13_features.csv', index=False)

plt.axvline(x=3.6, color='gray', linestyle='--', alpha=0.5, label='3.6V')
plt.axvline(x=4.0, color='gray', linestyle='-.', alpha=0.5, label='4.0V')
plt.xlabel('Voltage (V)', fontsize=12)
plt.ylabel('dQ/dV (Ah/V)', fontsize=12)
plt.title('dQ/dV Curves with Detected Peaks', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.show()


# In[ ]:


#Feature Extarction CY25 05_1 2-19 Batteries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapz
import os

base_path = r'TJU_Data'
output_path = r'TJU_CY25_05_1_features_all.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for file_index in range(2, 20): 
        file_path = os.path.join(base_path, f'CY25-05_1-#{file_index}.csv')
        
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            continue

        charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]
        cycles = sorted(charge_data['cycle number'].unique())
        features = []
        
        window_length = 21
        polyorder = 3
        dq_threshold = 1e-5
        upper_limit = 8
        lower_limit = -1e-2
        voltage_low = 3.8
        voltage_high = 4.3
        dqdv_cutoff = 2.5

        peak1_params = {'height': 0.4, 'distance': 8, 'prominence': 0.15}
        peak2_params = {'height': 0.4, 'distance': 8, 'prominence': 0.1}

        for cycle_num in cycles:
            filtered_data = charge_data[charge_data['cycle number'] == cycle_num]
            if filtered_data.empty:
                continue

            try:
                t = filtered_data['time/s'].values
                Q = filtered_data['Q charge/mA.h'].values / 1000
                Q = Q - Q[0]
                V = filtered_data['Ecell/V'].values

                if len(Q) < window_length:
                    continue

                cumulative_dq = Q
                valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
                if len(valid_indices) == 0:
                    continue

                cutoff_idx = valid_indices[-1] + 1
                t = t[:cutoff_idx]
                Q = Q[:cutoff_idx]
                V = V[:cutoff_idx]

                Q_smooth = savgol_filter(Q, window_length, polyorder)
                V_smooth = savgol_filter(V, window_length, polyorder)

                dQ_dt = np.gradient(Q_smooth, t)
                dV_dt = np.gradient(V_smooth, t)
                dV_dt[dV_dt == 0] = 1e-3
                dQ_dV = dQ_dt / dV_dt
                dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)

                def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):
                    data_smoothed = np.copy(data)
                    n = len(data_smoothed)
                    if n < window_size:
                        return data_smoothed
                    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)
                    for i in np.where(outliers)[0]:
                        start = max(0, i - window_size)
                        end = min(n, i + window_size + 1)
                        window = data_smoothed[start:end]
                        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                        if len(window_non_outlier) > 0:
                            data_smoothed[i] = np.mean(window_non_outlier)
                        else:
                            data_smoothed[i] = np.median(data_smoothed)
                    if len(data_smoothed) >= 5:
                        data_smoothed = savgol_filter(data_smoothed, 5, 2)
                    return data_smoothed

                dQ_dV_smoothed = replace_extreme_values(dQ_dV, upper_limit, lower_limit)

                voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
                keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)
                V_filtered = V_smooth[keep_mask]
                dQ_dV_filtered = dQ_dV_smoothed[keep_mask]
                if len(V_filtered) <= 10:
                    continue

                sorted_idx = np.argsort(V_filtered)
                V_sorted = V_filtered[sorted_idx]
                dQ_dV_sorted = dQ_dV_filtered[sorted_idx]

                peak1_voltage, peak1_value, peak2_voltage, peak2_value = np.nan, np.nan, np.nan, np.nan

                mask_36 = V_sorted > 3.65
                if np.sum(mask_36) > 0:
                    v_36 = V_sorted[mask_36]
                    dqdv_36 = dQ_dV_sorted[mask_36]
                    peaks_36, _ = find_peaks(dqdv_36, **peak1_params)
                    if len(peaks_36) > 0:
                        first_peak_idx = peaks_36[0]
                        peak1_voltage = v_36[first_peak_idx]
                        peak1_value = dqdv_36[first_peak_idx]

                mask_40 = V_sorted > 3.9
                if np.sum(mask_40) > 0:
                    v_40 = V_sorted[mask_40]
                    dqdv_40 = dQ_dV_sorted[mask_40]
                    peaks_40, _ = find_peaks(dqdv_40, **peak2_params)
                    if len(peaks_40) > 0:
                        first_peak_idx = peaks_40[0]
                        peak2_voltage = v_40[first_peak_idx]
                        peak2_value = dqdv_40[first_peak_idx]

                area = np.nan
                area_mask = (V_sorted >= 3.4) & (V_sorted <= 4.15)
                if np.sum(area_mask) >= 2:
                    v_area = V_sorted[area_mask]
                    dqdv_area = dQ_dV_sorted[area_mask]
                    area = trapz(dqdv_area, v_area)

                features.append({
                    'cycle_number': cycle_num,
                    'peak1_voltage(>3.6V)': peak1_voltage,
                    'peak1_dqdv': peak1_value,
                    'peak2_voltage(>4.0V)': peak2_voltage,
                    'peak2_dqdv': peak2_value,
                    'area_3.7-4.0V': area
                })

            except Exception as e:
                print(f"Error {e} handling cycle :{cycle_num} ")
                continue

        features_df = pd.DataFrame(features)
        sheet_name = f'Battery_#{file_index}'
        features_df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[25]:


# TJU CY25-025_1-#1 IC curve 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter

file_path = r'TJU_Data\CY25-025_1-#1.csv'

# Global font settings
plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  

def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):
    data_smoothed = np.copy(data)
    n = len(data_smoothed)
    
    if n < window_size:
        return data_smoothed  

    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)

    for i in np.where(outliers)[0]:

        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)  

        window = data_smoothed[start:end]
        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
        
        if len(window_non_outlier) > 0:
          
            data_smoothed[i] = np.mean(window_non_outlier)
        else:
            
            expand = 2
            while len(window_non_outlier) == 0 and (start - expand >= 0 or end + expand < n):
                new_start = max(0, i - window_size - expand)
                new_end = min(n, i + window_size + 1 + expand)
                window = data_smoothed[new_start:new_end]
                window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                expand += 1
            
            if len(window_non_outlier) > 0:
                data_smoothed[i] = np.mean(window_non_outlier)
            else:
                
                data_smoothed[i] = np.median(data_smoothed)

    if len(data_smoothed) >= 5:
        data_smoothed = savgol_filter(data_smoothed, 5, 2)
    
    return data_smoothed

try:
    data = pd.read_csv(file_path)
except Exception as e:
    print(f"Error {e} handling cycle :{cycle_num} ")
    exit()

charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]

cycles = sorted(charge_data['cycle number'].unique())

colors = cm.Blues(np.linspace(0.2, 1, len(cycles)))

plt.figure(figsize=(3, 2), dpi=600)

# Data smoothing parameter
window_length = 21  
polyorder = 3      

dq_threshold = 1e-5

upper_limit = 8    
lower_limit = -1e-2 

voltage_low = 3.8   
voltage_high = 4.3  
dqdv_cutoff = 2     

for i, cycle_num in enumerate(cycles):
   
    filtered_data = charge_data[charge_data['cycle number'] == cycle_num]

    if not filtered_data.empty:
       
        t = filtered_data['time/s'].values
        Q = filtered_data['Q charge/mA.h'].values / 1000  
        Q = Q - Q[0] 
        V = filtered_data['Ecell/V'].values

        if len(Q) < window_length:
            continue
        
        try:
            
            cumulative_dq = Q  
            valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
            
            if len(valid_indices) == 0:
                continue

            cutoff_idx = valid_indices[-1] + 1
            t = t[:cutoff_idx]
            Q = Q[:cutoff_idx]
            V = V[:cutoff_idx]
            
            # Savitzky-Golay
            Q_smooth = savgol_filter(Q, window_length, polyorder)
            V_smooth = savgol_filter(V, window_length, polyorder)

            dQ_dt = np.gradient(Q_smooth, t)
            dV_dt = np.gradient(V_smooth, t)

            dV_dt[dV_dt == 0] = 1e-3
            dQ_dV = dQ_dt / dV_dt

            dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)

            dQ_dV_smoothed = replace_extreme_values(
                dQ_dV, 
                upper_threshold=upper_limit, 
                lower_threshold=lower_limit
            )

            voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
            keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)

            V_filtered = V_smooth[keep_mask]
            dQ_dV_filtered = dQ_dV_smoothed[keep_mask]

            if len(V_filtered) > 10:  
                
                plt.plot(V_filtered, dQ_dV_filtered, 
                        linewidth=1, 
                        color=colors[i],
                        alpha=0.9,
                        label=f'Cycle {cycle_num}' if i % 5 == 0 else "")
            else:
                print("error")
            
        except Exception as e:
            print(f"Error {e} handling cycle :{cycle_num} ")
            continue

plt.xlabel('Voltage (V)', fontsize=11)
plt.ylabel('dQ/dV (Ah/V)', fontsize=11)
plt.title('IC Curves in TJU_Target', fontsize=11)
#plt.grid(True, linestyle='--', alpha=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')
plt.ylim(-0.5, 1e1)
plt.tight_layout()
plt.show()


# In[1]:


# Feature Extraction CY25-025_1 Single Battery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapz

file_path = r'TJU_Data\CY25-025_1-#3.csv'

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False

def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):
    data_smoothed = np.copy(data)
    n = len(data_smoothed)
    
    if n < window_size:
        return data_smoothed
    
    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)
    
    for i in np.where(outliers)[0]:
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = data_smoothed[start:end]
        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
        
        if len(window_non_outlier) > 0:
            data_smoothed[i] = np.mean(window_non_outlier)
        else:
            expand = 2
            while len(window_non_outlier) == 0 and (start - expand >= 0 or end + expand < n):
                new_start = max(0, i - window_size - expand)
                new_end = min(n, i + window_size + 1 + expand)
                window = data_smoothed[new_start:new_end]
                window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                expand += 1
            
            if len(window_non_outlier) > 0:
                data_smoothed[i] = np.mean(window_non_outlier)
            else:
                data_smoothed[i] = np.median(data_smoothed)
    
    if len(data_smoothed) >= 5:
        data_smoothed = savgol_filter(data_smoothed, 5, 2)
    
    return data_smoothed

try:
    data = pd.read_csv(file_path)
except Exception as e:
    exit()

charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]
cycles = sorted(charge_data['cycle number'].unique())

features = []

window_length = 21
polyorder = 3
dq_threshold = 1e-5
upper_limit = 8
lower_limit = -1e-2
voltage_low = 3.8
voltage_high = 4.3
dqdv_cutoff = 2.5

peak1_params = {
    'height': 0.4, 
    'distance': 8,  
    'prominence': 0.05  
}

peak2_params = {
    'height': 0.4,  
    'distance': 8, 
    'prominence': 0.1  
}

for i, cycle_num in enumerate(cycles):
    filtered_data = charge_data[charge_data['cycle number'] == cycle_num]
    
    if not filtered_data.empty:
        t = filtered_data['time/s'].values
        Q = filtered_data['Q charge/mA.h'].values / 1000
        Q = Q - Q[0]
        V = filtered_data['Ecell/V'].values
        
        if len(Q) < window_length:
            continue
        
        try:
           
            cumulative_dq = Q
            valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
            
            if len(valid_indices) == 0:
                continue
                
            cutoff_idx = valid_indices[-1] + 1
            t = t[:cutoff_idx]
            Q = Q[:cutoff_idx]
            V = V[:cutoff_idx]
            
            Q_smooth = savgol_filter(Q, window_length, polyorder)
            V_smooth = savgol_filter(V, window_length, polyorder)

            dQ_dt = np.gradient(Q_smooth, t)
            dV_dt = np.gradient(V_smooth, t)
            dV_dt[dV_dt == 0] = 1e-3
            dQ_dV = dQ_dt / dV_dt
            dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)
            dQ_dV_smoothed = replace_extreme_values(dQ_dV, upper_limit, lower_limit)

            voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
            keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)
            V_filtered = V_smooth[keep_mask]
            dQ_dV_filtered = dQ_dV_smoothed[keep_mask]
            
            if len(V_filtered) <= 10:
                continue

            sorted_idx = np.argsort(V_filtered)
            V_sorted = V_filtered[sorted_idx]
            dQ_dV_sorted = dQ_dV_filtered[sorted_idx]

            peak1_voltage = np.nan
            peak1_value = np.nan

            mask_36 = V_sorted > 3.63
            if np.sum(mask_36) > 0:
                v_36 = V_sorted[mask_36]
                dqdv_36 = dQ_dV_sorted[mask_36]

                peaks_36, _ = find_peaks(dqdv_36, **peak1_params)

                if len(peaks_36) > 0:
                    first_peak_idx = peaks_36[0]  
                    peak1_voltage = v_36[first_peak_idx]
                    peak1_value = dqdv_36[first_peak_idx]

            peak2_voltage = np.nan
            peak2_value = np.nan

            mask_40 = V_sorted > 3.9
            if np.sum(mask_40) > 0:
                v_40 = V_sorted[mask_40]
                dqdv_40 = dQ_dV_sorted[mask_40]

                peaks_40, _ = find_peaks(dqdv_40,** peak2_params)

                if len(peaks_40) > 0:
                    first_peak_idx = peaks_40[0] 
                    peak2_voltage = v_40[first_peak_idx]
                    peak2_value = dqdv_40[first_peak_idx]

            area = np.nan
            area_mask = (V_sorted >= 3.4) & (V_sorted <= 4.15)
            if np.sum(area_mask) >= 2: 
                v_area = V_sorted[area_mask]
                dqdv_area = dQ_dV_sorted[area_mask]
                area = trapz(dqdv_area, v_area) 
            else:
                print("error")
    

            features.append({
                'cycle_number': cycle_num,
                'peak1_voltage(>3.6V)': peak1_voltage,
                'peak1_dqdv': peak1_value,
                'peak2_voltage(>4.0V)': peak2_voltage,
                'peak2_dqdv': peak2_value,
                'area_3.7-4.0V': area
            })

            plt.plot(V_sorted, dQ_dV_sorted, linewidth=1, alpha=0.7, label=f'Cycle {cycle_num}')
            if not np.isnan(peak1_voltage):
                plt.scatter(peak1_voltage, peak1_value, color='red', s=30, zorder=5)  # 第一个峰标记为红色
            if not np.isnan(peak2_voltage):
                plt.scatter(peak2_voltage, peak2_value, color='green', s=30, zorder=5)  # 第二个峰标记为绿色
            
        except Exception as e:
            print(f"{cycle_num} error: {e}")
            continue

features_df = pd.DataFrame(features)
print(features_df)

#features_df.to_csv('\TJU_CY25_025_1_#2_features.csv', index=False)

plt.axvline(x=3.6, color='gray', linestyle='--', alpha=0.5, label='3.6V')
plt.axvline(x=4.0, color='gray', linestyle='-.', alpha=0.5, label='4.0V')
plt.xlabel('Voltage (V)', fontsize=12)
plt.ylabel('dQ/dV (Ah/V)', fontsize=12)
plt.title('dQ/dV Curves with Detected Peaks', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.show()


# In[2]:


#Feature Extraction CY25 025_1 All 1-7 Batteries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapz
import os

base_path = r'TJU_Data'
output_path = r'TJU_CY25_025_1_features_all.xlsx'

with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    for file_index in range(1, 8):  
        file_path = os.path.join(base_path, f'CY25-025_1-#{file_index}.csv')
        
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"File Failed: {e}")
            continue

        charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]
        cycles = sorted(charge_data['cycle number'].unique())
        features = []

        window_length = 21
        polyorder = 3
        dq_threshold = 1e-5
        upper_limit = 8
        lower_limit = -1e-2
        voltage_low = 3.8
        voltage_high = 4.3
        dqdv_cutoff = 2.5

        peak1_params = {'height': 0.4, 'distance': 8, 'prominence': 0.05}
        peak2_params = {'height': 0.4, 'distance': 8, 'prominence': 0.1}

        for cycle_num in cycles:
            filtered_data = charge_data[charge_data['cycle number'] == cycle_num]
            if filtered_data.empty:
                continue

            try:
                t = filtered_data['time/s'].values
                Q = filtered_data['Q charge/mA.h'].values / 1000
                Q = Q - Q[0]
                V = filtered_data['Ecell/V'].values

                if len(Q) < window_length:
                    continue

                cumulative_dq = Q
                valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
                if len(valid_indices) == 0:
                    continue

                cutoff_idx = valid_indices[-1] + 1
                t = t[:cutoff_idx]
                Q = Q[:cutoff_idx]
                V = V[:cutoff_idx]

                Q_smooth = savgol_filter(Q, window_length, polyorder)
                V_smooth = savgol_filter(V, window_length, polyorder)

                dQ_dt = np.gradient(Q_smooth, t)
                dV_dt = np.gradient(V_smooth, t)
                dV_dt[dV_dt == 0] = 1e-3
                dQ_dV = dQ_dt / dV_dt
                dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)

                def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):
                    data_smoothed = np.copy(data)
                    n = len(data_smoothed)
                    if n < window_size:
                        return data_smoothed
                    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)
                    for i in np.where(outliers)[0]:
                        start = max(0, i - window_size)
                        end = min(n, i + window_size + 1)
                        window = data_smoothed[start:end]
                        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                        if len(window_non_outlier) > 0:
                            data_smoothed[i] = np.mean(window_non_outlier)
                        else:
                            data_smoothed[i] = np.median(data_smoothed)
                    if len(data_smoothed) >= 5:
                        data_smoothed = savgol_filter(data_smoothed, 5, 2)
                    return data_smoothed

                dQ_dV_smoothed = replace_extreme_values(dQ_dV, upper_limit, lower_limit)

                voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
                keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)
                V_filtered = V_smooth[keep_mask]
                dQ_dV_filtered = dQ_dV_smoothed[keep_mask]
                if len(V_filtered) <= 10:
                    continue

                sorted_idx = np.argsort(V_filtered)
                V_sorted = V_filtered[sorted_idx]
                dQ_dV_sorted = dQ_dV_filtered[sorted_idx]

                peak1_voltage, peak1_value, peak2_voltage, peak2_value = np.nan, np.nan, np.nan, np.nan

                mask_36 = V_sorted > 3.63
                if np.sum(mask_36) > 0:
                    v_36 = V_sorted[mask_36]
                    dqdv_36 = dQ_dV_sorted[mask_36]
                    peaks_36, _ = find_peaks(dqdv_36, **peak1_params)
                    if len(peaks_36) > 0:
                        first_peak_idx = peaks_36[0]
                        peak1_voltage = v_36[first_peak_idx]
                        peak1_value = dqdv_36[first_peak_idx]

                mask_40 = V_sorted > 3.9
                if np.sum(mask_40) > 0:
                    v_40 = V_sorted[mask_40]
                    dqdv_40 = dQ_dV_sorted[mask_40]
                    peaks_40, _ = find_peaks(dqdv_40, **peak2_params)
                    if len(peaks_40) > 0:
                        first_peak_idx = peaks_40[0]
                        peak2_voltage = v_40[first_peak_idx]
                        peak2_value = dqdv_40[first_peak_idx]

                area = np.nan
                area_mask = (V_sorted >= 3.4) & (V_sorted <= 4.15)
                if np.sum(area_mask) >= 2:
                    v_area = V_sorted[area_mask]
                    dqdv_area = dQ_dV_sorted[area_mask]
                    area = trapz(dqdv_area, v_area)

                features.append({
                    'cycle_number': cycle_num,
                    'peak1_voltage(>3.6V)': peak1_voltage,
                    'peak1_dqdv': peak1_value,
                    'peak2_voltage(>4.0V)': peak2_voltage,
                    'peak2_dqdv': peak2_value,
                    'area_3.7-4.0V': area
                })

            except Exception as e:
                print(f"Error {e} handling cycle :{cycle_num} ")
                continue

        features_df = pd.DataFrame(features)
        sheet_name = f'Battery_#{file_index}'
        features_df.to_excel(writer, sheet_name=sheet_name, index=False)


# In[45]:


# CY35-05_1-#1 IC curve
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter

file_path = r'TJU_Data\CY35-05_1-#1.csv'

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False  
def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):
    data_smoothed = np.copy(data)
    n = len(data_smoothed)
    
    if n < window_size:
        return data_smoothed  

    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)

    for i in np.where(outliers)[0]:
       
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)  
  
        window = data_smoothed[start:end]
        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
        
        if len(window_non_outlier) > 0:
            
            data_smoothed[i] = np.mean(window_non_outlier)
        else:
            
            expand = 2
            while len(window_non_outlier) == 0 and (start - expand >= 0 or end + expand < n):
                new_start = max(0, i - window_size - expand)
                new_end = min(n, i + window_size + 1 + expand)
                window = data_smoothed[new_start:new_end]
                window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                expand += 1
            
            if len(window_non_outlier) > 0:
                data_smoothed[i] = np.mean(window_non_outlier)
            else:
               
                data_smoothed[i] = np.median(data_smoothed)

    if len(data_smoothed) >= 5:
        data_smoothed = savgol_filter(data_smoothed, 5, 2)
    
    return data_smoothed

def has_constant_dqdv_in_range(V, dQ_dV, voltage_min, voltage_max, min_points=5, flatness_threshold=0.05):

    voltage_range_mask = (V >= voltage_min) & (V <= voltage_max)
    points_in_range = np.sum(voltage_range_mask)

    if points_in_range < min_points:
        return False

    range_dqdv = dQ_dV[voltage_range_mask]

    std_dev = np.std(range_dqdv)
    value_range = np.max(range_dqdv) - np.min(range_dqdv)

    return std_dev < flatness_threshold and value_range < flatness_threshold * 2

try:
    data = pd.read_csv(file_path)
except Exception as e:
    print(f"File Failed: {e}")
    exit()

charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]

cycles = sorted(charge_data['cycle number'].unique())

colors = cm.Blues(np.linspace(0.2, 1, len(cycles)))
plt.rcParams["font.family"] = ["Times New Roman", "serif"]  

plt.figure(figsize=(3, 2), dpi=600)

window_length = 21  
polyorder = 3      

dq_threshold = 1e-5

upper_limit = 8   
lower_limit = -1e-2 

voltage_low = 4.0   
voltage_high = 4.3  
dqdv_cutoff = 2     

check_voltage_min = 3.3  
check_voltage_max = 3.6  
min_points_in_range = 8  
flatness_threshold = 0.05 

for i, cycle_num in enumerate(cycles):
  
    filtered_data = charge_data[charge_data['cycle number'] == cycle_num]

    if not filtered_data.empty:
        
        t = filtered_data['time/s'].values
        Q = filtered_data['Q charge/mA.h'].values / 1000 
        Q = Q - Q[0]  
        V = filtered_data['Ecell/V'].values

        if len(Q) < window_length:
            print(f"Cycle {cycle_num} data is lacked{window_length}，skip")
            continue
        
        try:
           
            cumulative_dq = Q  
            valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
            
            if len(valid_indices) == 0:
                print(f"Cycle {cycle_num} dQ less than threshold，skip")
                continue

            cutoff_idx = valid_indices[-1] + 1
            t = t[:cutoff_idx]
            Q = Q[:cutoff_idx]
            V = V[:cutoff_idx]
            
            # Savitzky-Golay
            Q_smooth = savgol_filter(Q, window_length, polyorder)
            V_smooth = savgol_filter(V, window_length, polyorder)

            dQ_dt = np.gradient(Q_smooth, t)
            dV_dt = np.gradient(V_smooth, t)

            dV_dt[dV_dt == 0] = 1e-3
            dQ_dV = dQ_dt / dV_dt

            dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)

            dQ_dV_smoothed = replace_extreme_values(
                dQ_dV, 
                upper_threshold=upper_limit, 
                lower_threshold=lower_limit
            )
            

            if has_constant_dqdv_in_range(V_smooth, dQ_dV_smoothed,
                                       check_voltage_min, check_voltage_max,
                                       min_points_in_range,
                                       flatness_threshold):
                print(f"Cycle {cycle_num} is deleted")
                continue

            voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
            keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)
 
            V_filtered = V_smooth[keep_mask]
            dQ_dV_filtered = dQ_dV_smoothed[keep_mask]

            if len(V_filtered) > 10:  
                
                plt.plot(V_filtered, dQ_dV_filtered, 
                        linewidth=1, 
                        color=colors[i],
                        alpha=0.9,
                        label=f'Cycle {cycle_num}' if i % 5 == 0 else "")
            else:
                print(f"Cycle {cycle_num} data is lacked，skip")
            
        except Exception as e:
            print(f"Error {e} handling cycle :{cycle_num} ")
            continue

plt.xlabel('Voltage (V)', fontsize=11)
plt.ylabel('dQ/dV (Ah/V)', fontsize=11)
#plt.title('dQ/dV Curves for Each Charge Cycle', fontsize=14)
#plt.grid(True, linestyle='--', alpha=0.7)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='best')
plt.ylim(-0.5, 1e1)

plt.tight_layout()
plt.show()  


# In[93]:


# Feature Extraction CY35-05_1 Single Battery
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import trapz

file_path = r'TJU_Data\CY35-05_1-#1.csv'

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["axes.unicode_minus"] = False

def replace_extreme_values(data, upper_threshold=10, lower_threshold=-1e-2, window_size=5):
    data_smoothed = np.copy(data)
    n = len(data_smoothed)
    
    if n < window_size:
        return data_smoothed
    
    outliers = (data_smoothed > upper_threshold) | (data_smoothed < lower_threshold)
    
    for i in np.where(outliers)[0]:
        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)
        window = data_smoothed[start:end]
        window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
        
        if len(window_non_outlier) > 0:
            data_smoothed[i] = np.mean(window_non_outlier)
        else:
            expand = 2
            while len(window_non_outlier) == 0 and (start - expand >= 0 or end + expand < n):
                new_start = max(0, i - window_size - expand)
                new_end = min(n, i + window_size + 1 + expand)
                window = data_smoothed[new_start:new_end]
                window_non_outlier = window[(window <= upper_threshold) & (window >= lower_threshold)]
                expand += 1
            
            if len(window_non_outlier) > 0:
                data_smoothed[i] = np.mean(window_non_outlier)
            else:
                data_smoothed[i] = np.median(data_smoothed)
    
    if len(data_smoothed) >= 5:
        data_smoothed = savgol_filter(data_smoothed, 5, 2)
    
    return data_smoothed

try:
    data = pd.read_csv(file_path)
except Exception as e:
    print(f"File Failed: {e}")
    exit()

charge_data = data[(data['Ecell/V'] != 0) & (data['Q charge/mA.h'] != 0)]
cycles = sorted(charge_data['cycle number'].unique())
print(f"Find{len(cycles)}cycles")

features = []

window_length = 21
polyorder = 3
dq_threshold = 1e-5
upper_limit = 8
lower_limit = -1e-2
voltage_low = 3.8
voltage_high = 4.3
dqdv_cutoff = 2.5

peak1_params = {
    'height': 0.4, 
    'distance': 8,  
    'prominence': 0.15  
}

peak2_params = {
    'height': 0.4,  
    'distance': 8,  
    'prominence': 0.1 
}

for i, cycle_num in enumerate(cycles):
    filtered_data = charge_data[charge_data['cycle number'] == cycle_num]
    
    if not filtered_data.empty:
        t = filtered_data['time/s'].values
        Q = filtered_data['Q charge/mA.h'].values / 1000
        Q = Q - Q[0]
        V = filtered_data['Ecell/V'].values
        
        if len(Q) < window_length:
            print(f"Cycle {cycle_num} data is lacked，skip")
            continue
        
        try:
            
            cumulative_dq = Q
            valid_indices = np.where(cumulative_dq >= dq_threshold)[0]
            
            if len(valid_indices) == 0:
                print(f"Cycle {cycle_num} dQ less than threshold，skip")
                continue
                
            cutoff_idx = valid_indices[-1] + 1
            t = t[:cutoff_idx]
            Q = Q[:cutoff_idx]
            V = V[:cutoff_idx]

            Q_smooth = savgol_filter(Q, window_length, polyorder)
            V_smooth = savgol_filter(V, window_length, polyorder)

            dQ_dt = np.gradient(Q_smooth, t)
            dV_dt = np.gradient(V_smooth, t)
            dV_dt[dV_dt == 0] = 1e-3
            dQ_dV = dQ_dt / dV_dt
            dQ_dV = np.nan_to_num(dQ_dV, posinf=0, neginf=0)
            dQ_dV_smoothed = replace_extreme_values(dQ_dV, upper_limit, lower_limit)

            voltage_range_mask = (V_smooth >= voltage_low) & (V_smooth <= voltage_high)
            keep_mask = ~voltage_range_mask | (dQ_dV_smoothed >= dqdv_cutoff)
            V_filtered = V_smooth[keep_mask]
            dQ_dV_filtered = dQ_dV_smoothed[keep_mask]
            
            if len(V_filtered) <= 10:
                print(f"Cycle {cycle_num} data is lacked，skip")
                continue

            sorted_idx = np.argsort(V_filtered)
            V_sorted = V_filtered[sorted_idx]
            dQ_dV_sorted = dQ_dV_filtered[sorted_idx]

            peak1_voltage = np.nan
            peak1_value = np.nan

            mask_36 = V_sorted > 3.65
            if np.sum(mask_36) > 0:
                v_36 = V_sorted[mask_36]
                dqdv_36 = dQ_dV_sorted[mask_36]

                peaks_36, _ = find_peaks(dqdv_36, **peak1_params)

                if len(peaks_36) > 0:
                    first_peak_idx = peaks_36[0]  
                    peak1_voltage = v_36[first_peak_idx]
                    peak1_value = dqdv_36[first_peak_idx]

            peak2_voltage = np.nan
            peak2_value = np.nan

            mask_40 = V_sorted > 3.9
            if np.sum(mask_40) > 0:
                v_40 = V_sorted[mask_40]
                dqdv_40 = dQ_dV_sorted[mask_40]

                peaks_40, _ = find_peaks(dqdv_40,** peak2_params)

                if len(peaks_40) > 0:
                    first_peak_idx = peaks_40[0]  
                    peak2_voltage = v_40[first_peak_idx]
                    peak2_value = dqdv_40[first_peak_idx]

            area = np.nan
            area_mask = (V_sorted >= 3.4) & (V_sorted <= 4.15)
            if np.sum(area_mask) >= 2:  
                v_area = V_sorted[area_mask]
                dqdv_area = dQ_dV_sorted[area_mask]
                area = trapz(dqdv_area, v_area)
            else:
                print(f"Cycle {cycle_num} Area is failed to calculate")

            features.append({
                'cycle_number': cycle_num,
                'peak1_voltage(>3.6V)': peak1_voltage,
                'peak1_dqdv': peak1_value,
                'peak2_voltage(>4.0V)': peak2_voltage,
                'peak2_dqdv': peak2_value,
                'area_3.7-4.0V': area
            })

            plt.plot(V_sorted, dQ_dV_sorted, linewidth=1, alpha=0.7, label=f'Cycle {cycle_num}')
            if not np.isnan(peak1_voltage):
                plt.scatter(peak1_voltage, peak1_value, color='red', s=30, zorder=5)  
            if not np.isnan(peak2_voltage):
                plt.scatter(peak2_voltage, peak2_value, color='green', s=30, zorder=5)  
            
        except Exception as e:
            print(f"Error {e} handling cycle :{cycle_num} ")
            continue

features_df = pd.DataFrame(features)
print(features_df)

#features_df.to_csv('TJU_CY35_05_1_features.csv', index=False)

plt.axvline(x=3.6, color='gray', linestyle='--', alpha=0.5, label='3.6V')
plt.axvline(x=4.0, color='gray', linestyle='-.', alpha=0.5, label='4.0V')
plt.xlabel('Voltage (V)', fontsize=12)
plt.ylabel('dQ/dV (Ah/V)', fontsize=12)
plt.title('dQ/dV Curves with Detected Peaks', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.5)
#plt.legend(fontsize=8, loc='best')
plt.tight_layout()
plt.show()


# # Transfer Learning

# In[18]:


#CALCE XGboost with transfer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

source_df = pd.read_excel("CALCE_source_features.xlsx")
target_df = pd.read_excel(r"CALCE_target_features.xlsx")
feature_cols = ['P1_x','P1_y','P2_x', 'P2_y', 'P12_Ar']
#feature_cols = ['Cycle','First_Peak_dQdV','Voltage_at_First','Second_Peak_dQdV', 'Voltage_at_Second', 'Area_3.7_4.2']
#feature_cols = ['First_Peak_dQdV']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

X_target = target_df[feature_cols].values
y_target = target_df[label_col].values

n_total = len(X_target)
n_labeled = n_total // 4

X_target_labeled = X_target[:n_labeled]
y_target_labeled = y_target[:n_labeled]

X_target_unlabeled = X_target[n_labeled:]
y_target_unlabeled = y_target[n_labeled:]  

scaler = StandardScaler()
scaler.fit(np.vstack([X_source, X_target_labeled]))
X_source = scaler.transform(X_source)
X_target_labeled = scaler.transform(X_target_labeled)
X_target_unlabeled = scaler.transform(X_target_unlabeled)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtrain_target_labeled = xgb.DMatrix(X_target_labeled, label=y_target_labeled)
dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Step 1: Train in the source domain
print("🔧 Training on source domain...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

# Step 2: Fine-tuning uses the first 1/4 of the data in the target domain
print("🔧 Fine-tuning on target domain (1/4)...")
model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)

# Step 3: Predict the latter 3/4 capacity of the target domain
y_pred = model.predict(dtest_target_unlabeled)

output_df = pd.DataFrame({
    "Cycle": np.arange(n_labeled, n_total),
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target[n_labeled:]  
})
#output_df.to_excel(r"xgb_predicted_target_with_transfer.xlsx", index=False)
print("\n✅ Prediction complete. Results saved to 'xgb_predicted_target_3_4.xlsx'")


# In[93]:


#CALCE XGboost with transfer with peak1 features
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 加载数据
source_df = pd.read_excel("CALCE_source_features.xlsx")
target_df = pd.read_excel(r"CALCE_target_features_corrected_all.xlsx")
target_df = pd.read_excel(r"CALCE_target_features.xlsx")
feature_cols = ['Cycle','P1_x','P1_y']
#feature_cols = ['Cycle','First_Peak_dQdV','Voltage_at_First','Second_Peak_dQdV', 'Voltage_at_Second', 'Area_3.7_4.2']
#feature_cols = ['First_Peak_dQdV']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

X_target = target_df[feature_cols].values
y_target = target_df[label_col].values

n_total = len(X_target)
n_labeled = n_total // 4

X_target_labeled = X_target[:n_labeled]
y_target_labeled = y_target[:n_labeled]

X_target_unlabeled = X_target[n_labeled:]
y_target_unlabeled = y_target[n_labeled:]  

scaler = StandardScaler()
scaler.fit(np.vstack([X_source, X_target_labeled]))
X_source = scaler.transform(X_source)
X_target_labeled = scaler.transform(X_target_labeled)
X_target_unlabeled = scaler.transform(X_target_unlabeled)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtrain_target_labeled = xgb.DMatrix(X_target_labeled, label=y_target_labeled)
dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Step 1: Train in the Source domain
print("🔧 Training on source domain...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

# Step 2: Fine-tuning uses the first 1/4 of the data in the target domain
print("🔧 Fine-tuning on target domain (1/4)...")
model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)

# Step 3: Predict the latter 3/4 capacity of the target domain
y_pred = model.predict(dtest_target_unlabeled)

# Output the results
output_df = pd.DataFrame({
    "Cycle": np.arange(n_labeled, n_total),
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target[n_labeled:]  
})
#output_df.to_excel(r"xgb_predicted_target_P1_with_transfer.xlsx", index=False)


# In[74]:


# CALCE no transfer 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

source_df = pd.read_excel("CALCE_source_features.xlsx")
target_df = pd.read_excel(r"CALCE_target_features.xlsx")

feature_cols = ['Cycle','F1','F2','F3', 
                'F4', 'F5']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

# Segment the target domain data and only use the last 3/4 part
split_index = int(len(target_df) * 1/4)  
target_df_subset = target_df.iloc[split_index:]  

X_target = target_df_subset[feature_cols].values
y_target = target_df_subset[label_col].values

scaler = StandardScaler()

scaler.fit(np.vstack([X_source, target_df[feature_cols].values]))
X_source = scaler.transform(X_source)
X_target = scaler.transform(X_target)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtest_target = xgb.DMatrix(X_target)  

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

print("🚀 Training model on source domain only...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

print("📊 Predicting on 3/4 target domain...")
y_pred = model.predict(dtest_target)

output_df = pd.DataFrame({
    "Cycle": target_df_subset["Cycle"].values,
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target
})

#output_df.to_excel(r"xgb_predicted_target_no_transfer.xlsx", index=False)

if len(y_pred) == len(y_target):
    mse = mean_squared_error(y_target, y_pred)
    mae = mean_absolute_error(y_target, y_pred)
    print(f"Evaluation Metrics：\nMSE = {mse:.4f}, MAE = {mae:.4f}")
    print(f"Estimated Samples: {len(y_pred)}")


# In[1]:


# CALCE Plot figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = True 

no_transfer_df = pd.read_excel(r"xgb_predicted_target_no_transfer.xlsx")
with_transfer_df = pd.read_excel(r"xgb_predicted_target_with_transfer.xlsx")
with_F12_transfer_df = pd.read_excel(r"D:\桌面上的文件\xgb_predicted_target_P1_with_transfer.xlsx")

full_target_df = pd.read_excel(r"CALCE_target_features.xlsx")

no_transfer_df=no_transfer_df[:-100]
with_transfer_df=with_transfer_df[:-100]
with_F12_transfer_df=with_F12_transfer_df[:-100]
full_target_df=full_target_df[:-100]

true_capacity = full_target_df[['Cap']].rename(columns={'Cap': 'True_Capacity'})
true_capacity['Period'] = true_capacity.index + 1  

n_total = len(true_capacity)
split_index = int(n_total * 1/4)  
split_period = split_index + 20 

no_transfer_data = pd.DataFrame({
    'Period': range(split_period, split_period + len(no_transfer_df)),
    'Predicted_No_Transfer': no_transfer_df['Predicted_Capacity'].values
})

with_transfer_data = pd.DataFrame({
    'Period': range(split_period, split_period + len(with_transfer_df)),
    'Predicted_Capacity_Transfer': with_transfer_df['Predicted_Capacity'].values
})

with_F12_transfer_data = pd.DataFrame({
    'Period': range(split_period, split_period + len(with_F12_transfer_df)),
    'Predicted_Capacity_Transfer': with_F12_transfer_df['Predicted_Capacity'].values
})

initial_capacity = true_capacity['True_Capacity'].iloc[0]

true_capacity['SOH'] = true_capacity['True_Capacity'] / initial_capacity
no_transfer_data['SOH'] = no_transfer_data['Predicted_No_Transfer'] / initial_capacity
with_transfer_data['SOH'] = with_transfer_data['Predicted_Capacity_Transfer'] / initial_capacity
with_F12_transfer_data['SOH'] = with_F12_transfer_data['Predicted_Capacity_Transfer'] / initial_capacity

plt.figure(figsize=(3, 2), dpi=600)

plt.plot(true_capacity['Period'], 
         true_capacity['SOH'], 
         color='gray', linewidth=2, label='True SOH')

plt.plot(no_transfer_data['Period'], no_transfer_data['SOH'], 
         color='#8ebcdb', linestyle='--', linewidth=1.5, label='No Transfer')

plt.plot(with_transfer_data['Period'], with_transfer_data['SOH'], 
         color='#a79fce', linewidth=1.5, label='With Transfer')

plt.plot(with_F12_transfer_data['Period'], with_F12_transfer_data['SOH'], 
         color='#fe9f69', linewidth=1.5, label='With P1_Features Transfer')

plt.xlabel('Lifetime (Cycles)', fontsize=11)
plt.ylabel('SOH (a.u.)', fontsize=11)
plt.title('SOH Estimation Using One Battery in CALCE')
plt.tight_layout()
plt.show()


# In[7]:


#CALCE Calculate absolute error 
true_soh_segment = true_capacity.loc[true_capacity['Period'] >= split_period, 'SOH'].values

min_len = min(len(true_soh_segment),
              len(no_transfer_data['SOH']),
              len(with_transfer_data['SOH']),
              len(with_F12_transfer_data['SOH']))

true_soh_segment = true_soh_segment[:min_len]
no_transfer_err = np.abs(no_transfer_data['SOH'].values[:min_len] - true_soh_segment)
with_transfer_err = np.abs(with_transfer_data['SOH'].values[:min_len] - true_soh_segment)
with_F12_transfer_err = np.abs(with_F12_transfer_data['SOH'].values[:min_len] - true_soh_segment)

cycles = no_transfer_data['Period'].values[:min_len]

plt.figure(figsize=(3, 2), dpi=600)

plt.plot(cycles, no_transfer_err, color='#8ebcdb', linestyle='--', linewidth=1.5, label='No Transfer Error')
plt.plot(cycles, with_transfer_err, color='#a79fce', linewidth=1.5, label='With Transfer Error')
plt.plot(cycles, with_F12_transfer_err, color='#fe9f69', linewidth=1.5, label='With P1_Features Transfer Error')

plt.xlabel('Lifetime (Cycles)', fontsize=11)
plt.ylabel('Absolute Error (a.u.)', fontsize=11)
plt.title('Estimation Error in CALCE')
plt.ylim(-0.02,0.2)
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

pred_no_transfer = no_transfer_data['SOH'].values[:min_len]
pred_with_transfer = with_transfer_data['SOH'].values[:min_len]
pred_F12_transfer = with_F12_transfer_data['SOH'].values[:min_len]

def calc_metrics(true, pred, name):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    print(f"{name} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

# 计算
calc_metrics(true_soh_segment, pred_no_transfer, "No Transfer")
calc_metrics(true_soh_segment, pred_with_transfer, "With Transfer")
calc_metrics(true_soh_segment, pred_F12_transfer, "F12 Transfer")


# In[30]:


# CALCE DE Analysis
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker 

source_df = pd.read_excel(r"CALCE_source_features.xlsx")
target_df = pd.read_excel(r"CALCE_target_features.xlsx")

label_col = 'Cap' 

feature_addition_order = [
    ['Cycle'],
    ['Cycle', 'P1_x'],
    ['Cycle', 'P1_x', 'P1_y'],
    ['Cycle', 'P1_x', 'P1_y', 'P2_x'],
    ['Cycle', 'P1_x', 'P1_y',  'P2_x', 'P2_y'],
    ['Cycle', 'P1_x', 'P1_y',  'P2_x', 'P2_y', 'P12_Ar']
]

results = []

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

for i, feature_cols in enumerate(feature_addition_order):
    print(f"\n🚀 Iteration {i+1}/{len(feature_addition_order)}: Using features: {feature_cols}")

    X_source = source_df[feature_cols].values
    y_source = source_df[label_col].values
    X_target = target_df[feature_cols].values
    y_target = target_df[label_col].values

    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target]))
    X_source = scaler.transform(X_source)
    X_target = scaler.transform(X_target)

    dtrain_source = xgb.DMatrix(X_source, label=y_source)
    dtest_target = xgb.DMatrix(X_target)

    model = xgb.train(params, dtrain_source, num_boost_round=300)

    y_pred = model.predict(dtest_target)

    capacity_nominal = y_target[0]
    soh_true = y_target / capacity_nominal
    soh_pred = y_pred / capacity_nominal

    rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    mae = mean_absolute_error(soh_true, soh_pred)

    results.append({
        'features': feature_cols,
        'rmse': rmse,
        'mae': mae
    })
    print(f"📊 Evaluation (SOH): RMSE = {rmse:.4f}, MAE = {mae:.4f}")

results_df = pd.DataFrame([{
    'Iteration': i+1,
    'Features Added': ", ".join(r['features']),
    'RMSE': r['rmse'],
    'MAE': r['mae']
} for i, r in enumerate(results)])
#results_df.to_excel(r"CALCE_xgb_feature_addition_results_summary.xlsx", index=False)

plt.figure(figsize=(3, 2), dpi=600)

x = range(1, len(results)+1)
rmse_values = [r['rmse'] for r in results]
mae_values = [r['mae'] for r in results]

ax1 = plt.gca()
line1, = ax1.plot(x, rmse_values, marker='o', color='#599CB4', label='RMSE')
ax1.set_xlabel('Number of Features Added', fontsize=9)
ax1.set_ylabel('RMSE', fontsize=8, color='#599CB4')
ax1.tick_params(axis='y', labelcolor='#599CB4', labelsize=9)
ax1.set_xticks(x)
ax1.tick_params(axis='x', labelsize=9)

ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax2 = ax1.twinx()
line2, = ax2.plot(x, mae_values, marker='s', color='#C25759', label='MAE')
ax2.set_ylabel('MAE', fontsize=8, color='#C25759')
ax2.tick_params(axis='y', labelcolor='#C25759', labelsize=9)
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

lines = [line1, line2]
ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='upper right')

plt.tight_layout()
plt.title("Data Efficiency in CALCE", fontsize=10)
plt.show()
print("\n✅ All iterations complete. Results saved to Excel and visualization done.")


# 特征可解释性，迁移性

# In[19]:


#CALCE SHAP
import shap
import matplotlib.pyplot as plt

print("\n🧠 Running SHAP to explain feature contributions...")

# 1. Create an interpreter
explainer = shap.Explainer(model)

# 2. Calculate SHAP values
shap_values = explainer(X_target_unlabeled)
#shap_values = explainer(X_target_labeled) 

# 3. Global feature importance
#shap.summary_plot(shap_values, X_target_unlabeled, feature_names=selected_features)
shap.summary_plot(shap_values, X_target_unlabeled)
#shap.summary_plot(shap_values, X_target_labeled)
plt.close()

# 4. Local explanation
shap.plots.waterfall(shap_values[0], max_display=10)


# In[20]:


import shap
import matplotlib.pyplot as plt

FIG_SIZE = (4, 3)  
DPI = 600

CUSTOM_FEATURE_NAMES = [
    "P1_y",         
    "P1_x",       
    "P2_y",      
    "P2_x",    
    "P12_Ar"     
]

SAVE_PATH_SUMMARY = "shap_summary_plot.png"
SAVE_PATH_WATERFALL = "shap_waterfall_plot_sample0.png"


print("\n🧠 Running SHAP to explain feature contributions...")

explainer = shap.Explainer(model)

shap_values = explainer(X_target_unlabeled)

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 11  

plt.figure(figsize=FIG_SIZE)
plt.title('SHAP Analysis in CALCE')

shap.summary_plot(
    shap_values, 
    X_target_unlabeled,
    feature_names=CUSTOM_FEATURE_NAMES,
    plot_size=(4, 3) 
)

plt.savefig(
    SAVE_PATH_SUMMARY,
    dpi=DPI,
    bbox_inches='tight'  
)
plt.close()  

plt.figure(figsize=FIG_SIZE)

shap.plots.waterfall(
    shap_values[0],
    max_display=10,
    #feature_names=CUSTOM_FEATURE_NAMES 
)

plt.savefig(
    SAVE_PATH_WATERFALL,
    dpi=DPI,
    bbox_inches='tight'
)
plt.close()


# In[77]:


# Rank CALCE TJU
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = True  

features = ["P1_y", "P1_x", "P2_y", "P2_x", "P12_Ar"]  
correlation = [5, 2, 4, 1, 3] 
importance = [4, 2, 1, 5, 3]  

correlation = [1, 3, 4, 5, 2]  
importance = [1, 4, 3, 5, 2]  

correlation = [2, 3, 4, 5, 1]  
importance = [1, 4, 3, 5, 2]  

migration = np.array(correlation) - np.array(importance)  

x = np.arange(len(features)) 
width = 0.25  

plt.figure(figsize=(3, 3),dpi=600)  

bars1 = plt.bar(x - width, correlation, width, label='Predictability\n(Correlation)', 
                color='#E89DA0')
bars2 = plt.bar(x, importance, width, label='Predictability and transferability\n(Importance)', 
                color='#B2D3A4')
bars3 = plt.bar(x + width, migration, width, label='transferability\n(Importance - Correlation)', 
                color='#88CEE6')

def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height}', ha='center', va='bottom', fontsize=10)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.xlabel("Features", fontsize=11)
plt.ylabel("Rank Value", fontsize=11)
plt.title("Rank of Feature Performance in CALCE", fontsize=12)
plt.title("Rank of Feature Performance in TJU", fontsize=12)

plt.xticks(x, features)

plt.ylim(min(migration)-1, max(max(correlation), max(importance)) + 1)

plt.tight_layout()

plt.show()


# In[8]:


#TJU XGBoost with transfer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

source_df = pd.read_excel("TJU_CY25_05_1_features.xlsx")
target_df = pd.read_excel(r"TJU_CY25_025_1_features.xlsx")  
target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:]

feature_cols = ['P1_x','P1_y','P2_x', 
                'P2_y', 'P12_Ar']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

X_target = target_df[feature_cols].values
y_target = target_df[label_col].values

n_total = len(X_target)
n_labeled = n_total // 4

X_target_labeled = X_target[:n_labeled]
y_target_labeled = y_target[:n_labeled]

X_target_unlabeled = X_target[n_labeled:]
y_target_unlabeled = y_target[n_labeled:]  

scaler = StandardScaler()
scaler.fit(np.vstack([X_source, X_target_labeled]))
X_source = scaler.transform(X_source)
X_target_labeled = scaler.transform(X_target_labeled)
X_target_unlabeled = scaler.transform(X_target_unlabeled)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtrain_target_labeled = xgb.DMatrix(X_target_labeled, label=y_target_labeled)
dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

print("🔧 Training on source domain...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

print("🔧 Fine-tuning on target domain (1/4)...")
model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)

y_pred = model.predict(dtest_target_unlabeled)

output_df = pd.DataFrame({
    "Cycle": np.arange(n_labeled, n_total),
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target[n_labeled:] 
})
#output_df.to_excel(r"TJU_xgb_predicted_target_with_transfer.xlsx", index=False)


# In[9]:


# TJU SHAP
import shap
import matplotlib.pyplot as plt

print("\n🧠 Running SHAP to explain feature contributions...")

explainer = shap.Explainer(model)

shap_values = explainer(X_target_unlabeled)

#shap.summary_plot(shap_values, X_target_unlabeled, feature_names=selected_features)
shap.summary_plot(shap_values, X_target_unlabeled)
#shap.summary_plot(shap_values, X_target_labeled)
plt.close()

shap.plots.waterfall(shap_values[0], max_display=10)


# In[15]:


import shap
import matplotlib.pyplot as plt

FIG_SIZE = (4, 3)  
DPI = 600

CUSTOM_FEATURE_NAMES = [
    "P1_x",          
    "P1_y",       
    "P2_x",       
    "P2_y",   
    "P12_Ar"     
]

SAVE_PATH_SUMMARY = "shap_summary_plot.png"
SAVE_PATH_WATERFALL = "shap_waterfall_plot_sample0.png"


print("\n🧠 Running SHAP to explain feature contributions...")

# 1. Create an interpreter
explainer = shap.Explainer(model)

# 2. Calculate SHAP values
shap_values = explainer(X_target_unlabeled)

# 3. Global Feature Importance 

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.size"] = 11 

plt.figure(figsize=FIG_SIZE)
plt.title('SHAP Analysis in TJU')

shap.summary_plot(
    shap_values, 
    X_target_unlabeled,
    feature_names=CUSTOM_FEATURE_NAMES,
    plot_size=(4, 3) 
)
plt.title('SHAP Analysis in TJU')

plt.savefig(
    SAVE_PATH_SUMMARY,
    dpi=DPI,
    bbox_inches='tight'  
)
plt.close() 

plt.figure(figsize=FIG_SIZE)

shap.plots.waterfall(
    shap_values[0],
    max_display=10,
    #feature_names=CUSTOM_FEATURE_NAMES  
)

plt.savefig(
    SAVE_PATH_WATERFALL,
    dpi=DPI,
    bbox_inches='tight'
)
plt.title('Shap Analysis in TJU')
plt.close()

print("📈 SHAP plots saved to:")
print(f" - {SAVE_PATH_SUMMARY} (global importance)")
print(f" - {SAVE_PATH_WATERFALL} (1st sample breakdown)")


# In[31]:


# DE Analysis TJU
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

source_df = pd.read_excel("TJU_CY25_05_1_features.xlsx")
target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:]
# source_df = source_df[:-150] 
# target_df = target_df[:-150] 

label_col = 'Cap'

feature_addition_order = [
    ['cycle_number'],
    ['cycle_number', 'P1_x'],
    ['cycle_number', 'P1_x', 'P1_y'],
    #['Cycle', 'First_Peak_dQdV', 'Voltage_at_First'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x'],
    ['cycle_number', 'P1_x', 'P1_y',  'P2_x', 'P2_y'],
    ['cycle_number', 'P1_x', 'P1_y',  'P2_x', 'P2_y', 'P12_Ar']
]

results = []

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

for i, feature_cols in enumerate(feature_addition_order):
    print(f"\n🚀 Iteration {i+1}/{len(feature_addition_order)}: Using features: {feature_cols}")

    X_source = source_df[feature_cols].values
    y_source = source_df[label_col].values
    X_target = target_df[feature_cols].values
    y_target = target_df[label_col].values
  
    scaler = StandardScaler()
    scaler.fit(np.vstack([X_source, X_target]))
    X_source = scaler.transform(X_source)
    X_target = scaler.transform(X_target)

    dtrain_source = xgb.DMatrix(X_source, label=y_source)
    dtest_target = xgb.DMatrix(X_target)
 
    model = xgb.train(params, dtrain_source, num_boost_round=300)

    y_pred = model.predict(dtest_target)

    capacity_nominal = y_target[0]
    soh_true = y_target / capacity_nominal
    soh_pred = y_pred / capacity_nominal

    rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))
    mae = mean_absolute_error(soh_true, soh_pred)
    
    
    results.append({
        'features': feature_cols,
        'rmse': rmse,
        'mae': mae,
        'predictions': y_pred
    })
    
    print(f"📊 Evaluation: RMSE = {rmse:.4f}, MAE = {mae:.4f}")

    output_df = pd.DataFrame({
        "Cycle": target_df['cycle_number'].values,
        "Predicted_Capacity": y_pred,
        "True_Capacity": y_target,
        "Iteration": i+1,
        "Features": ", ".join(feature_cols)
    })

results_df = pd.DataFrame([{
    'Iteration': i+1,
    'Features Added': ", ".join(r['features']),
    'RMSE': r['rmse'],
    'MAE': r['mae']
} for i, r in enumerate(results)])
results_df.to_excel(r"TJU_xgb_feature_addition_results_summary.xlsx", index=False)

plt.figure(figsize=(3, 2), dpi=600)

x = range(1, len(results)+1)
rmse_values = [r['rmse'] for r in results]
mae_values = [r['mae'] for r in results]

ax1 = plt.gca()  
line1, = ax1.plot(x, rmse_values, marker='o', color='#599CB4', label='RMSE')
ax1.set_xlabel('Number of Features Added', fontsize=9)
ax1.set_ylabel('RMSE', fontsize=8, color='#599CB4')  
ax1.tick_params(axis='y', labelcolor='#599CB4', labelsize=9) 
ax1.set_xticks(x)
ax1.tick_params(axis='x', labelsize=9)

ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

ax2 = ax1.twinx()  
line2, = ax2.plot(x, mape_values, marker='s', color='#C25759', label='MAE')
ax2.set_ylabel('MAE', fontsize=8, color='#C25759') 
ax2.tick_params(axis='y', labelcolor='#C25759', labelsize=9) 

ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

lines = [line1, line2]
ax1.legend(lines, [l.get_label() for l in lines], fontsize=7, loc='upper right')

plt.tight_layout()
plt.title("Data Efficiency in TJU",fontsize=10)
plt.show()

print("\n✅ All iterations complete. Results saved to Excel files and visualization.")


# In[32]:


#TJU DE Analysis Across Batteries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

source_excel_path = r"TJU_CY25_05_1_features_all_1.xlsx"
target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:]

exclude_sheets = ["Battery_#4","Battery_#5","Battery_#8","Battery_#9","Battery_#13"]
excel_file = pd.ExcelFile(source_excel_path)
valid_sheets = [s for s in excel_file.sheet_names if s not in exclude_sheets]

feature_addition_order = [
    ['cycle_number'],
    ['cycle_number', 'P1_x'],
    ['cycle_number', 'P1_x', 'P1_y'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x', 'P2_y'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x', 'P2_y', 'P12_Ar']
]

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

all_results = []  
label_col = 'Cap'

for sheet in valid_sheets:
    source_df = pd.read_excel(source_excel_path, sheet_name=sheet)[5:]

    for feature_cols in feature_addition_order:
        X_source = source_df[feature_cols].values
        y_source = source_df[label_col].values
        X_target = target_df[feature_cols].values
        y_target = target_df[label_col].values

        scaler = StandardScaler()
        scaler.fit(np.vstack([X_source, X_target]))
        X_source = scaler.transform(X_source)
        X_target = scaler.transform(X_target)

        dtrain = xgb.DMatrix(X_source, label=y_source)
        dtest = xgb.DMatrix(X_target)
        model = xgb.train(params, dtrain, num_boost_round=300)
        y_pred = model.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_target, y_pred))
  
        capacity_nominal = y_target[0]
        soh_true = y_target / capacity_nominal
        soh_pred = y_pred / capacity_nominal

        rmse = np.sqrt(mean_squared_error(soh_true, soh_pred))

        all_results.append({
            "Source_Battery": sheet,
            "Feature_Count": len(feature_cols),
            "RMSE": rmse
        })

results_df = pd.DataFrame(all_results)

plt.figure(figsize=(4, 3), dpi=600)
sns.violinplot(x="Feature_Count", y="RMSE", data=results_df, inner="box", palette="Set2")

plt.xlabel("Number of Features Added", fontsize=16)
plt.ylabel("Estimation RMSE", fontsize=16)
plt.title("Error Distribution Across Batteries in TJU", fontsize=18)

ax = plt.gca()

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)

ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.show()


# In[115]:


#The influence of the number and the number of source domain batteries on SOH error
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

source_excel_path = r"TJU_CY25_05_1_features_all_1.xlsx"
target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:-100]

exclude_sheets = ["Battery_#4","Battery_#5","Battery_#8","Battery_#9","Battery_#13"]
excel_file = pd.ExcelFile(source_excel_path)
valid_sheets = [s for s in excel_file.sheet_names if s not in exclude_sheets]

feature_addition_order = [
    ['cycle_number'],
    ['cycle_number', 'P1_x'],
    ['cycle_number', 'P1_x', 'P1_y'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x', 'P2_y'],
    ['cycle_number', 'P1_x', 'P1_y', 'P2_x', 'P2_y', 'P12_Ar']
]

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

label_col = 'Cap'

heatmap_data = np.zeros((len(valid_sheets), len(feature_addition_order)))

for n_batteries in range(1, len(valid_sheets) +1):
   
    selected_sheets = valid_sheets[:n_batteries]

    source_df_list = [pd.read_excel(source_excel_path, sheet_name=s)[5:] for s in selected_sheets]
    combined_source_df = pd.concat(source_df_list, ignore_index=True)

    for j, feature_cols in enumerate(feature_addition_order):
        X_source = combined_source_df[feature_cols].values
        y_source = combined_source_df[label_col].values
        X_target = target_df[feature_cols].values
        y_target = target_df[label_col].values

        scaler = StandardScaler()
        scaler.fit(np.vstack([X_source, X_target]))
        X_source = scaler.transform(X_source)
        X_target = scaler.transform(X_target)

        dtrain = xgb.DMatrix(X_source, label=y_source)
        dtest = xgb.DMatrix(X_target)
        model = xgb.train(params, dtrain, num_boost_round=300)
        y_pred = model.predict(dtest)

        rmse = np.sqrt(mean_squared_error(y_target, y_pred))
        heatmap_data[n_batteries-1, j] = rmse

plt.figure(figsize=(6, 5), dpi=600)
sns.heatmap(heatmap_data, annot=True, fmt=".2e", cmap="YlGnBu", cbar_kws={'label': 'RMSE'})

plt.xlabel("Number of Features Added", fontsize=14)
plt.ylabel("Number of Source Batteries Used", fontsize=14)
plt.title("RMSE Heatmap (Incremental Source Batteries)", fontsize=16)

plt.xticks(ticks=np.arange(len(feature_addition_order)) + 0.5,
           labels=[len(f) for f in feature_addition_order],
           rotation=0, fontsize=12)
plt.yticks(ticks=np.arange(len(valid_sheets)) + 0.5,
           labels=np.arange(1, len(valid_sheets) + 1),
           rotation=0, fontsize=12)

plt.tight_layout()
plt.show()


# In[32]:


#TJU no transfer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

source_df = pd.read_excel("TJU_CY25_05_1_features.xlsx")
target_df = pd.read_excel(r"TJU_CY25_025_1_features.xlsx")
target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:]

feature_cols = ['P1_x','P1_y','P2_x', 
                'P2_y', 'P12_Ar']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

split_index = int(len(target_df) * 1/4)  
target_df_subset = target_df.iloc[split_index:]  

X_target = target_df_subset[feature_cols].values
y_target = target_df_subset[label_col].values

scaler = StandardScaler()

scaler.fit(np.vstack([X_source, target_df[feature_cols].values]))
X_source = scaler.transform(X_source)
X_target = scaler.transform(X_target)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtest_target = xgb.DMatrix(X_target) 

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

print("🚀 Training model on source domain only...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

print("📊 Predicting on 3/4 target domain...")
y_pred = model.predict(dtest_target)

output_df = pd.DataFrame({
    "Cycle": target_df_subset["cycle_number"].values,
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target
})

#output_df.to_excel(r"TJU_xgb_predicted_target_no_transfer.xlsx", index=False)
if len(y_pred) == len(y_target):
    mse = mean_squared_error(y_target, y_pred)
    mae = mean_absolute_error(y_target, y_pred)
    print(f"Evaluation Metrics：\nMSE = {mse:.4f}, MAE = {mae:.4f}")
    print(f"Estimated Samples: {len(y_pred)}")


# In[113]:


#TJU XGboost with transfer P1 features
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

source_df = pd.read_excel("TJU_CY25_05_1_features_all_1.xlsx",sheet_name="Battery_#2")
target_df = pd.read_excel(r"TJU_CY25_025_1_features.xlsx")
target_df = pd.read_excel(r"TJU_CY35_05_1_features.xlsx")[10:]
feature_cols = ['P1_x','P1_y']
#feature_cols = ['Cycle','First_Peak_dQdV','Voltage_at_First','Second_Peak_dQdV', 'Voltage_at_Second', 'Area_3.7_4.2']
#feature_cols = ['First_Peak_dQdV']
label_col = 'Cap'

X_source = source_df[feature_cols].values
y_source = source_df[label_col].values

X_target = target_df[feature_cols].values
y_target = target_df[label_col].values

n_total = len(X_target)
n_labeled = n_total // 4

X_target_labeled = X_target[:n_labeled]
y_target_labeled = y_target[:n_labeled]

X_target_unlabeled = X_target[n_labeled:]
y_target_unlabeled = y_target[n_labeled:]  

scaler = StandardScaler()
scaler.fit(np.vstack([X_source, X_target_labeled]))
X_source = scaler.transform(X_source)
X_target_labeled = scaler.transform(X_target_labeled)
X_target_unlabeled = scaler.transform(X_target_unlabeled)

dtrain_source = xgb.DMatrix(X_source, label=y_source)
dtrain_target_labeled = xgb.DMatrix(X_target_labeled, label=y_target_labeled)
dtest_target_unlabeled = xgb.DMatrix(X_target_unlabeled)

params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Step 1: Train in source domain
print("🔧 Training on source domain...")
model = xgb.train(params, dtrain_source, num_boost_round=300)

# Step 2: Fine-tuning uses the first 1/4 of the data in the target domain
print("🔧 Fine-tuning on target domain (1/4)...")
model = xgb.train(params, dtrain_target_labeled, num_boost_round=100, xgb_model=model)

# Step 3: Predict the latter 3/4 capacity of the target domain
y_pred = model.predict(dtest_target_unlabeled)

# Output results
output_df = pd.DataFrame({
    "Cycle": np.arange(n_labeled, n_total),
    "Predicted_Capacity": y_pred,
    "True_Capacity": y_target[n_labeled:]  
})
output_df.to_excel(r"TJU_xgb_predicted_target_P1_with_transfer.xlsx", index=False)


# In[5]:


#CALCE Correlation heatmap 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
import sys

def analyze_excel_correlation(file_path, output_image_path='correlation_heatmap.png'):
    
    try:
       
        df = pd.read_excel(file_path)

        if df.empty:
            print("Error：No data in Excel！")
            return
  
        all_columns = df.columns.tolist()

        feature_columns = ['C', 'D', 'E', 'F', 'G']
        target_column = 'I'

        try:
            
            feature_names = [all_columns[ord(col)-ord('A')] for col in feature_columns]
            target_name = all_columns[ord(target_column)-ord('A')]

            missing_columns = [col for col in feature_names + [target_name] if col not in all_columns]
            if missing_columns:
                print(f"Error：can not find column {', '.join(missing_columns)}")
                return
 
            selected_df = df[feature_names + [target_name]].copy()

            non_numeric_cols = selected_df.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"Transform: {', '.join(non_numeric_cols)}")
                for col in non_numeric_cols:
                    selected_df[col] = pd.to_numeric(selected_df[col], errors='coerce')
 
            if selected_df.isna().any().any():
                print("Warning：data has nan")
                selected_df = selected_df.fillna(0)

            correlation = selected_df.corr()

            print("\n Correlation matrix：")
            print(correlation)
 
            plt.rcParams["font.family"] = ["Times New Roman", "serif"] 
           
            plt.figure(figsize=(3, 2),dpi=600)
 
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', 
                        linewidths=0.5, annot_kws={'size': 9},vmin=-1,vmax=1)
            plt.title('Correlation Analysis in CALCE_Source', fontsize=11)
            plt.tight_layout()

            plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
 
            if 'ipykernel' in sys.modules:
                plt.show()
            else:
                print("Run this script in an environment that supports graphical user interface")
                
        except IndexError:
            print(f"Error：can not obtain column {', '.join(feature_columns + [target_column])}")
            return
            
    except FileNotFoundError:
        print(f"Error：can nit find file {file_path}")
    except Exception as e:
        print(f"error：{e}")

if __name__ == "__main__":
    file_path = 'CALCE_source_features.xlsx'  
    analyze_excel_correlation(file_path)


# In[2]:


#TJU Correlation heatmap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl import load_workbook
import sys

def analyze_excel_correlation(file_path, output_image_path='correlation_heatmap.png'):
    
    try:

        df = pd.read_excel(file_path)

        if df.empty:
            print("错误：Excel文件中没有数据！")
            return

        all_columns = df.columns.tolist()

        feature_columns = ['B', 'C', 'D', 'E', 'F']
        target_column = 'G'

        try:

            feature_names = [all_columns[ord(col)-ord('A')] for col in feature_columns]
            target_name = all_columns[ord(target_column)-ord('A')]

            missing_columns = [col for col in feature_names + [target_name] if col not in all_columns]
            if missing_columns:
                print(f"Error：can not find column {', '.join(missing_columns)}")
                return
            
            selected_df = df[feature_names + [target_name]].copy()

            non_numeric_cols = selected_df.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                print(f"Warning：Please transform: {', '.join(non_numeric_cols)}")
                for col in non_numeric_cols:
                    selected_df[col] = pd.to_numeric(selected_df[col], errors='coerce')
 
            if selected_df.isna().any().any():
                print("Warning：data has nan")
                selected_df = selected_df.fillna(0)
 
            correlation = selected_df.corr()

            print("\n Correlation matrix：")
            print(correlation)
 
            plt.rcParams["font.family"] = ["Times New Roman", "serif"] 
        
            plt.figure(figsize=(3, 2),dpi=600)

            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', 
                        linewidths=0.5, annot_kws={'size': 9},vmin=-1,vmax=1)
           
            plt.title('Correlation Analysis in TJU_Target',fontsize=11)
            plt.tight_layout()
            
            plt.savefig(output_image_path, dpi=300, bbox_inches='tight')

            if 'ipykernel' in sys.modules:
                plt.show()
            else:
                print("Run this script in an environment that supports graphical user interface")
                
        except IndexError:
            print(f"Error：can not obtain column {', '.join(feature_columns + [target_column])}")
            return
            
    except FileNotFoundError:
        print(f"Error：can not find file {file_path}")
    except Exception as e:
        print(f"Error：{e}")

if __name__ == "__main__":
    file_path = 'TJU_CY35_05_1_features.xlsx' 
    analyze_excel_correlation(file_path)


# In[50]:


#Feature curves
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"TJU_CY35_05_1_features.xlsx"
df = pd.read_excel(file_path)[10:]

cycle_col = df.columns[0]
features = ["P1_x", "P1_y", "P2_x", "P2_y", "P12_Ar"]

ylabels = {
    "P1_x": "P1_x (V)",
    "P1_y": "P1_y (Q/V)",
    "P2_x": "P2_x (Voltage)",
    "P2_y": "P2_y (Q/Voltage)",
    "P12_Ar": "P12_Ar (Q)"
}

for feature in features:
    plt.figure(figsize=(2, 2), dpi=600)
    plt.plot(df[cycle_col], df[feature], label=feature, color="#599CB4")
    plt.xlabel("LifeTime (Cycles)", fontsize=12)
    plt.ylabel(ylabels[feature], fontsize=12)
    plt.title(f"Feature : {feature}", fontsize=12)
    plt.tight_layout()
    plt.show()


# In[49]:


#CALCE Feature curves
import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = r"CALCE_target_features.xlsx"
df = pd.read_excel(file_path)

features = {
    "P1_x": "P1_x (V)",
    "P1_y": "P1_y (Q/V)",
    "P2_x": "P2_x (Voltage)",
    "P2_y": "P2_y (Q/Voltage)",
    "P12_Ar": "P12_Ar (Q)"
}

x_values = range(len(df))

for feature, ylabel in features.items():
    plt.figure(figsize=(2, 2), dpi=600)
    plt.plot(x_values, df[feature], label=feature, color="#9FBA95")
    plt.xlabel("LifeTime (Cycles)", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"Feature : {feature}", fontsize=12)
    plt.tight_layout()
    plt.show()


# In[36]:


#CALCE SOH curve
import pandas as pd

source_df = pd.read_excel("CALCE_source_features.xlsx")
target_df = pd.read_excel("CALCE_target_features.xlsx")

source_cap = source_df["Cap"]
target_cap = target_df["Cap"]

source_soh = source_cap / source_cap.iloc[0]
target_soh = target_cap / target_cap.iloc[0]

plt.figure(figsize=(3, 2), dpi=600)
plt.plot(source_soh.index, source_soh, label="Source Domain (CALCE)", linewidth=2, color='blue')
plt.plot(target_soh.index, target_soh, label="Target Domain (CALCE)", linewidth=2, color='red')

plt.title("SOH of CALCE Batteries", fontsize=12)
plt.xlabel("Lifetime (Cycles)", fontsize=10)
plt.ylabel("SOH (a.u.)", fontsize=10)
#plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.show()

