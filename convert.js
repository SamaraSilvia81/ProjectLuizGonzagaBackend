const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const loadModel = async () => {
  const modelPath = './models/model.json';
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  return model;
};

// Carregar o modelo
const model = await loadModel();

// Realizar previsões com o modelo
const input = tf.tensor2d([[...seus_dados_de_entrada]]);
const predictions = model.predict(input);

// Converter as previsões para um array
const predictionsArray = await predictions.array();

console.log(predictionsArray);
