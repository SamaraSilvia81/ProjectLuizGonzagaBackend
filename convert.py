import tensorflow as tf
from keras.models import load_model

# Definir a função de perda personalizada
def sparse_cat_loss(y_true, y_pred):
    # Implemente sua função de perda aqui
    pass

# Registrar a função de perda personalizada
tf.keras.utils.get_custom_objects()['sparse_cat_loss'] = sparse_cat_loss

# Carregar o modelo salvo em formato H5
model = load_model('luizgonzaga_gen_mjr.h5')

# Salvar o modelo como JSON
model_json = model.to_json()
with open('./models/model.json', 'w') as json_file:
    json_file.write(model_json)
