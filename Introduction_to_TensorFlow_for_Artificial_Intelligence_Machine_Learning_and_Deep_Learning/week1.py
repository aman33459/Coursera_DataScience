import tensorflow as tf
import numpy as np
from tensorflow import keras
# GRADED FUNCTION: house_model
def house_model(y_new):
    k = 1.0
    xs=[]
    ys=[]
    for i in range(1,10000):
        xs.append(i)
        ys.append(k)
        k = k +0.5
    print(xs)
    print(ys)
    #xs = np.array([1.0,2.0,3.0,4.0 , 5.0 , 6.0],dtype=float) # Your Code Here#
    #ys = np.array([1.0 , 1.5 , 2.0 , 2.5 , 3.0 , 3.5] , dtype=float) # Your Code Here#
    model =  tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # Your Code Here#
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=500)
    return model.predict(y_new)[0]
prediction = house_model([7.0])
print(prediction)
