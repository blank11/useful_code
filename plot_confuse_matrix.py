import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline # ipynb

def plot_confuse_matrix(cm, classes, normalized=False, title='Confuse Matrix', plt.cm.Blues):
	if normalized:
		cm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
		print('Normalized confuse matrix')
	else:
		print('Confuse matrix, without normalized')

	print(cm)

	plt.imshow(cm, interplation='nearest', cmp=cmp)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arrange(len(classes))
	plt.yticks(tick_marks, classsed, rotation=45)
	plt.xticks(tick_marks, classes)

	fmt = '.2f' if normalized else 'd'
	thresh = cm.max / 2.

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt), horizentalalignment='center', color='white' if cm[i, j] > thresh else 'black')

	plt.tight_layout()
	plt.xlabel('True label')
	plt.ylabel('Predict label')