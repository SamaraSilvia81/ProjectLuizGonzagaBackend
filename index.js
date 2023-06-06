const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const cors = require('cors');

const app = express();
app.use(cors());
const port = 3001;

app.use(express.json());

// Load the trained model
const path = require('path');

const loadModel = async () => {
  const modelPath = path.join(__dirname, 'luizgonzaga_gen_mjr.h5');
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  return model;
};

// Make prediction based on the loaded model and input text
/*const makePrediction = async (model, text) => {
  const tensor = tf.tensor2d([text]);
  const tensorResult = model.predict(tensor);
  const response = tensorResult.arraySync()[0];
  return response;
};*/

const makePrediction = async (model, text) => {
  const tensor = tf.tensor2d([text]);
  const tensorResult = model.predict(tensor);
  const response = await tensorResult.array();
  return response[0];
};

// Route to receive text from the front-end and send it to the AI
app.post('/generate', async (req, res) => {
  const { text } = req.body;

  try {
    // Load the model
    const AIModel = await loadModel();

    // Make prediction based on the text
    const AIResponse = await makePrediction(AIModel, text);

    // Return the AI response to the front-end
    res.json({ response: AIResponse });
  } catch (error) {
    console.error('Error processing text:', error);
    res.status(500).json({ error: 'Error processing text' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});