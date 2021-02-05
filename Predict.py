import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


#Loading saved model
model = keras.models.load_model('Yoga_Posture_model.h5')



test_datagen=ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('DATA/test/TEST',
                                                batch_size=1,
                                                class_mode='categorical',
                                                target_size=(150, 150),
                                                shuffle=False)


result = model.evaluate(test_generator, batch_size=1)


y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(test_generator.classes, y_pred))


cf_matrix = confusion_matrix(test_generator.classes, y_pred)
Labels = ['downdog', 'tree', 'warrior2', 'goddess', 'plank']
plt.figure(figsize=(20, 8))
heatmap = sns.heatmap(cf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d', color='blue')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()




