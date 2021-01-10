import warnings
import missingno as msno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

plt.style.use('seaborn')
sns.set(font_scale=1.5)


warnings.filterwarnings("ignore")

%matplotlib inline


# 기본적 모듈을 임포트 해줍니다.

os.listdir("../input")

# input의 하위폴더를 확인해줍니다.

df_train = pd.read_csv(
    "../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv(
    "../input/house-prices-advanced-regression-techniques/test.csv")

# 트레인, 테스트 데이터를 불러옵니다.
