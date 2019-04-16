from sklearn import tree
from typing_extensions import Final

BUMPY: Final = 1
SMOOTH: Final = 2

features = [[140,1], [130,1],[150, 2],[170,2]]
labels = ['orange', 'orange', 'apple', 'apple']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

apple_weight_1 = 190
answer = clf.predict([[apple_weight_1, SMOOTH]])
print (answer)
orange_weight_1 = 120
answer = clf.predict([[orange_weight_1, BUMPY]])
print (answer)