import numpy as np
import pandas as pd
import setting as st
import os

if __name__ == "__main__":
    testpath = './test'
    result_folder_path = './result.csv'

    result = st.model(testpath)
    print(result)
    if not os.path.isfile(result_folder_path):
        with open(result_folder_path, mode='w') as f:
            f.close()
    result.to_csv(result_folder_path, index=None)
