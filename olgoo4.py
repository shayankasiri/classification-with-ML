import numpy as np
import pandas as pd
import rasterio
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
##########################################################
#plotting
class_labels = {
    1: 'Water',
    2: 'urban',
    3: 'vegetation'
}
# ----- colormap -----
colors = ['blue', 'red', 'green']
cmap = ListedColormap(colors)

# ----- legend -----
legend_patches = [
    mpatches.Patch(color=colors[i-1], label=class_labels[i])
    for i in class_labels
]
##########################################################
# reading pic
with rasterio.open(r'C:\Users\Administrator\Downloads\sentinel_2A (3).tif') as src:
    img=src.read()
    profile=src.profile
    img=img.transpose(1,2,0).astype(float)
    rows, cols, bands = img.shape

##########################################################
# reading training and test
data=pd.read_csv(r'C:\Users\Administrator\Downloads\level 3.csv')
x=data[['B2','B3','B4','B5','B6','B7','B8']].values.astype(float)
y=data['landcover'].values.astype(int)
##########################################################
# normalization
if x.max()>100:
    x=x/10000
    img_norm=img/10000
else:
    img_norm=img.copy()
##########################################################
# split data to train and test
x_train, x_test, y_train, y_test=train_test_split(
    x , y , test_size=0.3 , stratify=y , random_state=42)

scaler=StandardScaler().fit(x_train)
x_train_s=scaler.transform(x_train)
x_test_s=scaler.transform(x_test)
##########################################################
# matrix => rows=pixels , cols=bands   (whole image must be classified in the end)
flat=img_norm.reshape(-1, bands)
flat_s = scaler.transform(flat)
##########################################################

#SVM classifier
svm_rbf=SVC(kernel='rbf' , C=10 , gamma=0.1)
svm_rbf.fit(x_train_s , y_train)

y_pred_svm= svm_rbf.predict(x_test_s)
pred_rbf_full=svm_rbf.predict(flat_s).reshape(rows, cols)

#for third question
y_train_pred = svm_rbf.predict(x_train_s)
print("classification report for test svm:")
print(classification_report(y_train, y_train_pred))
kappa1 = cohen_kappa_score(y_train, y_train_pred)
print("Kappa random forest:", kappa1)
###################
print("classification report svm:")
print(classification_report(y_test , y_pred_svm))
print("confusion matrix svm:")
print(confusion_matrix(y_test , y_pred_svm))
kappa2 = cohen_kappa_score(y_test, y_pred_svm)
print("Kappa svm:", kappa2)

plt.figure(figsize=(7, 7))
plt.imshow(pred_rbf_full, cmap=cmap)
plt.title('Land Cover Classification - SVM')
plt.axis('off')
plt.legend(
    handles=legend_patches,
    loc='lower right',
    fontsize=10
)
plt.show()
#####################################################################################

# random forest classifier
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42) 
rf.fit(x_train_s, y_train)

y_pred_rf = rf.predict(x_test_s)
pred_rf_full = rf.predict(flat_s).reshape(rows, cols)

#for third question
y_train_pred = rf.predict(x_train_s)
print("classification report for test random forest:")
print(classification_report(y_train, y_train_pred))
kappa1 = cohen_kappa_score(y_train, y_train_pred)
print("Kappa random forest:", kappa1)
###########
print("classification report random forest:")
print(classification_report(y_test, y_pred_rf))
print("confusion matrix random forest:")
print(confusion_matrix(y_test, y_pred_rf))
kappa2 = cohen_kappa_score(y_test, y_pred_rf)
print("Kappa random forest:", kappa2)

plt.figure(figsize=(7, 7))
plt.imshow(pred_rf_full, cmap=cmap)
plt.title('Land Cover Classification - RANDOM FOREST')
plt.axis('off')
plt.legend(
    handles=legend_patches,
    loc='lower right',
    fontsize=10
)
plt.show()

#######################################################################

#XGBOOST
y_train_xgb = y_train - 1
y_test_xgb = y_test - 1

xgb = XGBClassifier(
    n_estimators=200,      # تعداد درخت‌ها
    max_depth=6,           # حداکثر عمق هر درخت
    learning_rate=0.1,     # نرخ یادگیری
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',  # چون مسئله چندکلاسه است
    num_class=len(np.unique(y_train_xgb)), # تعداد کلاس‌ها
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(x_train_s, y_train_xgb) 
y_pred_xgb = xgb.predict(x_test_s)
pred_xgb_full = xgb.predict(flat_s).reshape(rows, cols)

print("classification report xgbosst:")
print(classification_report(y_test_xgb, y_pred_xgb))
print("confusion matrix xgboost:")
print(confusion_matrix(y_test_xgb, y_pred_xgb))
kappa = cohen_kappa_score(y_test_xgb, y_pred_xgb)
print("Kappa xgboost:", kappa)

plt.figure(figsize=(7, 7))
plt.imshow(pred_xgb_full, cmap=cmap)
plt.title('Land Cover Classification - XGBOOST')
plt.axis('off')
plt.legend(
    handles=legend_patches,
    loc='lower right',
    fontsize=10
)
plt.show()
##############################################################

# LIGHTGBM
y_train_lgbm = y_train - 1
y_test_lgbm = y_test - 1

lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)

lgbm.fit(x_train_s, y_train_lgbm)

y_pred_lgbm = lgbm.predict(x_test_s)

pred_lgbm_full = lgbm.predict(flat_s).reshape(rows, cols)

print("classification report lightgmb:")
print(classification_report(y_test_lgbm, y_pred_lgbm))
print("confusion matrix lightgbm:")
print(confusion_matrix(y_test_lgbm, y_pred_lgbm))
kappa = cohen_kappa_score(y_test_lgbm, y_pred_lgbm)
print("Kappa lightgbm:", kappa)

plt.figure(figsize=(7, 7))
plt.imshow(pred_lgbm_full, cmap=cmap)
plt.title('Land Cover Classification - LIGHTGBM')
plt.axis('off')
plt.legend(
    handles=legend_patches,
    loc='lower right',
    fontsize=10
)
plt.show()