from flask import Flask ,redirect , url_for , render_template,request,jsonify
from flask_mysqldb import MySQL
import datetime
import os
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread
import tensorflow as tf
from keras.preprocessing import image
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# Connect MySQL Database
app.config['MYSQL_HOST'] = '127.0.0.1'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'abhi_123'
app.config['MYSQL_DB'] = 'plant_disease'

# initailizing the MySQL
mysql = MySQL(app)


img_height = 180
img_width = 180

model = tf.keras.models.load_model('model/Iris_Model.h5')

# Class names and mapping
class_names = ['iris_setosa', 'iris_versicolour', 'iris_virginica']

# Read the dataset
df = pd.read_csv('model/data_set.csv', encoding='latin-1')
df = df.replace('', np.nan)
df = df.dropna(axis="columns", how="any")


# routes 
@app.route('/', methods=['GET', 'POST'])
def index():
   filename = None
   if request.method == 'POST':
      file = request.files['image']
      if file:
         target_directory = 'static/uploads/'
         new_filename = 'test_image.jpg'
         absolute_path = os.path.join(target_directory, new_filename)
         file.save(absolute_path)

         
         path = absolute_path

         directory_path = "static/uploads/"
         file_name = 'test_image.jpg'

         path = os.path.join(directory_path, file_name)

         if os.path.exists(path):
            print(f"The file path is: {path}")
         else:
            print(f"The file {path} does not exist.")

         #something
         img = image.load_img(path,target_size=(img_height,img_width))
         img = image.img_to_array(img)
         img = np.expand_dims(img,axis=0)

         # Make predictions
         prediction = model.predict(img)
         predicted_class = np.argmax(prediction)

         print(f"Predicted class: {predicted_class}")
         print(f"Predict Disease: {class_names[predicted_class]}")

         # Extract information from the dataset
         Class_Name,Varieties,Petals,Sepals,Flowers,Leaves,Height,Habitat = df.loc[0,[ 'Class Name','Varieties','Petals','Sepals','Flowers','Leaves','Height','Habitat']]
         print(f"Class_Name :{Class_Name} \nVarieties :{Varieties}\nPetals : {Petals}\nSepals : {Sepals}\nFlowers : {Flowers}\nLeaves : {Leaves}\nHeight : {Height}\nHabitat : {Habitat}")

         path = os.path.abspath(path)

         # dataset = {
         #    "Disease_Type": Class_Name,
         #    "Severity": Varieties,
         #    "Description": Facts,
         #    "Symptoms": Uses,
         #    "Diagnosis": Petals,
         #    "Precautions": Sepals,
         #    "Flowers":Flowers,
         #    "Leaves":Leaves,
         #    "Height":Height,
         #    "Habitat":Habitat
         # }
         return render_template('display.html',Class_Name=Class_Name,Varieties=Varieties,Petals=Petals,Sepals=Sepals,Flowers=Flowers,Leaves=Leaves,Height=Height,Habitat=Habitat)

   return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True)