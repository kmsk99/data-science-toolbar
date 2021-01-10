# importing all the required ML packages
# 유명한 randomforestclassfier 입니다.
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics  # 모델의 평가를 위해서 씁니다

model = RandomForestClassifier()
model.fit(X_tr, y_tr)
prediction = model.predict(X_vld)

print('총 {}명 중 {:.2f}% 정확도로 생존을 맞춤'.format(
    y_vld.shape[0], 100 * metrics.accuracy_score(prediction, y_vld)))
