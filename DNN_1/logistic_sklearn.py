from utils import _initialize, optimizer
import sklearn
from sklearn.linear_model import LogisticRegression

# 1. Choose DATA : Titanic / Digit
# 2. Adjust Hyperparameters
# ========================= EDIT HERE ========================
# DATA
DATA_NAME = 'Digit'

# HYPERPARAMETERS
num_epochs = 300
# ============================================================

assert DATA_NAME in ['Titanic', 'Digit','Basic_coordinates']

# Load dataset, model and evaluation metric
train_data, test_data, _, metric = _initialize(DATA_NAME)
train_x, train_y = train_data

num_data, num_features = train_x.shape
print('# of Training data : ', num_data)
ACC = 0.0
# ========================= EDIT HERE ========================
# Make model & optimizer
x = train_x
y = train_y.reshape(x.shape[0], 1).ravel()
test_x, test_y = test_data
test_y = test_y.reshape(test_x.shape[0], 1).ravel()
# TRAIN
model = LogisticRegression(max_iter = num_epochs).fit(x, y)
# EVALUATION
pred = model.predict(test_x)
ACC = model.score(test_x, test_y)

# ============================================================

print('ACC on Test Data : %.2f ' % ACC)
