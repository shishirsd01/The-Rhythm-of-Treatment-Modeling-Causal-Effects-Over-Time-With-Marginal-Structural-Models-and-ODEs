#Population level ODE without MSM and GPS
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pysindy as ps

# Load and combine all patient data
processed_patients = []

for i in range(1, 29):
    try:
        filename = f"HUPA{i:04d}P.csv"
        df_patient = pd.read_csv(filename, sep=';')
       # df_patient.drop(['time', 'steps','heart_rate'], axis=1, inplace=True)
        new_time = np.arange(0, 5*len(df_patient), 5)
        df_subset = df_patient.iloc[:len(new_time)].copy()
        df_subset['time'] = new_time
        df_subset.set_index('time', inplace=True)
        df_subset['patient_id'] = i
        processed_patients.append(df_subset)
    except Exception as e:
        print(f"Error with patient {i}: {e}")   
        

combined_df = pd.concat(processed_patients, axis=0)

# Discover the Population ODE
features = ["glucose","carb_input", "bolus_volume_delivered", "basal_rate"]
X = combined_df[features].dropna()
X_smooth = X.apply(lambda col: savgol_filter(col, window_length=5, polyorder=2), axis=0)
X_np = X_smooth.to_numpy()

dt = 5.0
optimizer = ps.STLSQ(threshold=0.0001)
library = ps.PolynomialLibrary(degree=2)

#here we use SINDy to get ODE equations
model_population = ps.SINDy(optimizer=optimizer, feature_library=library, feature_names=['G', 'c', 'ba', 'bo'] )
model_population.fit(X_np, t=dt)



print("Population ODE without MSM")
print(model_population.equations()[0])
