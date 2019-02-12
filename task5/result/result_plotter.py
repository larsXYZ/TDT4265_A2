import pickle
import matplotlib.pyplot as plt

with open ('with_expanded_dataset', 'rb') as fp:
    test_acc_extended = pickle.load(fp)

with open ('without_expanded_dataset', 'rb') as fp:
    test_acc_not_extended = pickle.load(fp)

plt.plot(test_acc_extended, label="Test accuracy with extended dataset")
plt.plot(test_acc_not_extended, label="Test accuracy without extended dataset")
plt.grid(color='grey', linestyle='-', linewidth=1)
plt.legend()
plt.show()