import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = sys.argv[1]
worm_num = int(sys.argv[2])
print(algorithm, ' worm_num: ', worm_num)

file_pattern = f'../data/generated/saved_Y/{{}}__{algorithm}_worm_{worm_num}'
Y0_tr = np.loadtxt(file_pattern.format('Y0_tr'))
Y1_tr = np.loadtxt(file_pattern.format('Y1_tr'))
Y0_tst = np.loadtxt(file_pattern.format('Y0_tst'))
Y1_tst = np.loadtxt(file_pattern.format('Y1_tst'))
B_train_1 = np.loadtxt(file_pattern.format('B_train_1'))
B_test_1 = np.loadtxt(file_pattern.format('B_test_1'))

Y0_tr = Y0_tr.reshape(Y0_tr.shape[0],-1)
Y1_tr = Y1_tr.reshape(Y1_tr.shape[0],-1)
Y0_tst = Y0_tst.reshape(Y0_tst.shape[0],-1)
Y1_tst = Y1_tst.reshape(Y1_tst.shape[0],-1)
Ydiff_tr = Y1_tr - Y0_tr
Ydiff_tst = Y1_tst - Y0_tst

### Behaviour decodability evaluation
acc_list = []
for i in tqdm(np.arange(10)):
    b_predictor = tf.keras.Sequential([layers.Dense(8, activation='linear')]) 
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    b_predictor.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = b_predictor.fit(Y1_tr,
                          B_train_1,
                          epochs=100,
                          batch_size=100,
                          validation_data=(Y1_tst, B_test_1),
                          verbose=0
                          )

    B1_tst_pred = b_predictor(Y1_tst).numpy().argmax(axis=1)
    acc_list.append(accuracy_score(B1_tst_pred, B_test_1))
# Saving the metrics
acc_list = np.array(acc_list)
np.savetxt('../data/generated/evaluation_metrics/acc_list_' + algorithm + '_worm_' +  str(worm_num),acc_list)

