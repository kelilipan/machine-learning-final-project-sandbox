import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Experiment with deeplearning eh multi layer JST
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

classifier = 'JST_keras_v2_test_overnight'
postfix = '_no_outliers_heatmap'


df = pd.read_csv('data/main_dataset_no_outliers_v2.csv')
y = df['best_pos']
X = df.drop(['best_pos'], axis=1)
d = joblib.load('data/label_encoder_model_v2.sav')
print(X.shape, y.shape)

feature = ['weak_foot', 'skill_moves', 'shooting', 'attacking_crossing',
       'attacking_finishing', 'attacking_volleys', 'skill_dribbling',
       'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
       'movement_agility', 'movement_balance', 'power_shot_power',
       'power_long_shots', 'mentality_positioning', 'mentality_vision',
       'mentality_penalties', 'goalkeeping_diving', 'goalkeeping_handling',
       'goalkeeping_kicking', 'goalkeeping_positioning',
       'goalkeeping_reflexes',
        'potential', 'overall', 'pace', 'passing', 'dribbling', 'defending', 'physic'] #combine with my previous feature
X = X[feature]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, to_categorical(y), test_size=0.25, random_state=10101)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(29,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
	keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(5, activation=tf.nn.softmax),
])
adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
model.fit(X_train, y_train, epochs=500, batch_size=1)
test_loss, test_acc = model.evaluate(X_test, y_test)
classes = d['best_pos'].classes_
print(*zip(range(8),classes),sep='\n')
result = model.predict(X_test)
result = np.argmax(result,axis=1)
result
result = pd.DataFrame(result)
result_inverse = d['best_pos'].inverse_transform(np.ravel(result))
np.argmax(y_test, axis=1)
print(classification_report(np.argmax(y_test, axis=1), result, zero_division=True))
report = classification_report(np.argmax(y_test, axis=1), result, zero_division=True, output_dict=True)
df_report = pd.DataFrame(report).T
df_report.to_csv('output/'+ classifier + postfix +'.csv')
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), result)
joblib.dump(conf_matrix,'conf_matrix_jsv.sav')
#save model
filename = 'model/'+ classifier + postfix +'.h5'
model.save(filename)