import sklearn
import keras
import warnings
import missingno as msno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

plt.style.use('seaborn')
# 이 두줄은 본 필자가 항상 쓰는 방법입니다. matplotlib 의 기본 scheme 말고 seaborn scheme 을 세팅하고, 일일이 graph 의 font size 를 지정할 필요 없이 seaborn 의 font_scale 을 사용하면 편합니다.
sns.set(font_scale=1.5)

# ignore warnings
# 워닝 메세지를 생략해 줍니다. 차후 버전관리를 위해 필요한 정보라고 생각하시면 주석처리 하시면 됩니다.
warnings.filterwarnings('ignore')

%matplotlib inline


# 기본적 모듈을 임포트 해줍니다.

os.listdir("../input")

# input의 하위폴더를 확인해줍니다.

df_train = pd.read_csv(
    "../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv(
    "../input/house-prices-advanced-regression-techniques/test.csv")
df_submit = pd.read_csv('../input/sample_submission.csv')

# 트레인, 테스트 데이터를 불러옵니다.
