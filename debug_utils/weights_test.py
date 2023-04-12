import numpy as np
preds = []
pred1 = np.zeros((2, 2, 2))
pred2 = np.ones((2, 2, 2))
print(pred1)

print(pred2)
preds.append(pred1)
preds.append(pred2)
preds = np.mean(preds, axis=0)
print('----------------')
print(preds)

