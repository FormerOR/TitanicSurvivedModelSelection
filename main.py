import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.tree import DecisionTreeClassifier as DTC

train = pandas.read_csv('train.csv', index_col=0)
test = pandas.read_csv('test.csv', index_col=0)

'''
进行数据预处理
'''

# 查看数据的特征
# print(test.describe())

# 按实际情况分析，Ticket、Cabin、Name这三列数据对生存率影响不大，不考虑，删除列
train.drop(['Cabin', 'Ticket', 'Name'], inplace=True, axis=1)
test.drop(['Cabin', 'Ticket', 'Name'], inplace=True, axis=1)

# 发现Age列数据有缺失，使用平均值填充
train["Age"] = train["Age"].fillna(train['Age'].median())
test["Age"] = test["Age"].fillna(test['Age'].median())

# 填充完成进行检查数据
# print(train.describe())
# print(test.describe())
# print(test['Age'].describe())

# Sex列是male和female（字符串类数据） 机器学习中无法直接利用，需要将其转化为数值型0、1
train.loc[train["Sex"] == 'male', "Sex"] = 0
train.loc[train["Sex"] == 'female', "Sex"] = 1
test.loc[test["Sex"] == 'male', "Sex"] = 0
test.loc[test["Sex"] == 'female', "Sex"] = 1

# 检查是否修改成功
# print(train["Sex"].unique())
# print(test["Sex"].unique())

# 查看另外一列的数据值类型
# print(train["Embarked"].unique())

# 同样的，Embarked列的值分为S、C、Q和空，需要将其修改为0、1、2，又因为为空的数据只有两个，所以将其行删除
# 在这里遇到困难上网查询如何删除值为空的行，其中一个方法是先将其填充为9999再找到其索引，利用索引将其行删除
# 但我不是很明白为什么不能直接如下方法寻找空值
# find_index=train[(train.Embarked == '')].index.tolist()
# 运行结果打印下标是[]，也就是空列表

train["Embarked"] = train["Embarked"].fillna("9999")
find_index_train = train[(train.Embarked == '9999')].index.tolist()
test["Embarked"] = test["Embarked"].fillna("9999")
find_index_test = test[(test.Embarked == '9999')].index.tolist()

# print(find_index)
# 结果为find_index=[62, 830]
train = train.drop(find_index_train)
test = test.drop(find_index_test)

# print(train["Embarked"].unique())
# print(test["Embarked"].unique())
# 结果为['S' 'C' 'Q']，修改数据成功

# 然后将S、C、Q修改为0、1、2
train.loc[train["Embarked"] == "S", 'Embarked'] = 0
train.loc[train["Embarked"] == "C", 'Embarked'] = 1
train.loc[train["Embarked"] == "Q", 'Embarked'] = 2

test.loc[test["Embarked"] == "S", 'Embarked'] = 0
test.loc[test["Embarked"] == "C", 'Embarked'] = 1
test.loc[test["Embarked"] == "Q", 'Embarked'] = 2
# print(train["Embarked"].unique())
# print(test["Embarked"].unique())
# [0 1 2]修改成功

# print(train)
# print(test)
'''
数据预处理基本完成
'''

'''
选择模型进行机器学习
'''

# print(train.columns)

# 保留的特征
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

data = train[features].values.copy()
# print(data)
target = train.Survived.values.copy()
# print(target)

Xtest = test[features].values.copy()
# print(Xtest)

skf = StratifiedKFold(n_splits=5)

# 平均得分
scores = []

# 遍历5次
for train0, test0 in skf.split(data, target):
    # print(train)
    # print(test)

    # 训练数据 测试数据
    X_train = data[train0]
    X_test = data[test0]
    Y_train = target[train0]
    Y_test = target[test0]

    # 线性回归
    linear = LinearRegression()
    linear.fit(X_train, Y_train)

    # 预测
    y = linear.predict(X_test)
    # print(y)
    # print(y_test)

    # 测值大于0.6则认为正例
    Y_pred = y >= 0.6

    # 得分和准确率
    score = (Y_pred == Y_test).mean()
    # print(score)

    scores.append(score)

print('线性回归模型得分：', numpy.array(scores).mean())

# 逻辑斯蒂回归模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lr = LogisticRegression(max_iter=2000)

# 快速得到交叉验证后的得分
scores = cross_val_score(lr,data,target,cv=5)

# 平均得分
print('逻辑斯蒂回归模型得分：',scores.mean())

# 随机森林模型
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(data, target)

# rfc.predict(Xtest)

scores = cross_val_score(rfc, data, target, cv=5)

print('随机森林模型得分：', scores.mean())

'''
结合三个没有调参的模型的得分，使用随机森林比较适合
'''
