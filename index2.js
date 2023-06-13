const express = require('express');
const { loadLayersModel, tensor2d } = require('@tensorflow/tfjs-node');
const cors = require('cors');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

const app = express();
const port = process.env.PORT || 3000;

const vocab = ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', '0', '2', '3', '5', '7', ':', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '}', 'À', 'É', 'Ê', 'Í', 'Ó', 'Ô', 'à', 'á', 'â', 'ã', 'ç', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ü', '’', '\ufeff'];

let char_to_ind = {};
let ind_to_char = {};

vocab.forEach((char, index) => {
  char_to_ind[char] = index;
  ind_to_char[index] = char;
});

app.use(cors());
app.use(express.json());

let AIModel;

const loadModel = async () => {
  try {
    const modelPath = path.resolve(__dirname, 'tmp/tfjs_mobilenetv2', 'model.json');
    AIModel = await loadLayersModel(`file://${modelPath}`);

    console.log('Modelo carregado com sucesso');
  } catch (error) {
    console.error('Erro ao carregar o modelo:', error);
  }
};

const generateText = async (startSeed, genSize) => {
  const numGenerate = genSize;
  const inputEval = Array.from(startSeed, (char) => char_to_ind[char]);
  let input_eval = tf.tensor2d([inputEval]);

  const textGenerated = [];

  for (let i = 0; i < numGenerate; i++) {
    const predictions = AIModel.predict(input_eval);
    const outputData = await predictions.array();
    const outputReal = [outputData[0][0]]
    const teste = outputReal[0]

    let maxIndex = 0;
    let maxValue = teste[0];
    
    for (let i = 1; i < teste.length; i++) {
      if (teste[i] > maxValue) {
        maxValue = teste[i];
        maxIndex = i;
      }
    }
    const predicted_id = maxIndex

    inputEval.push(predicted_id);
    input_eval = tf.tensor2d([[predicted_id]]);

    textGenerated.push(ind_to_char[predicted_id]);
  }

  return startSeed + ' ' + textGenerated.join('');
};


app.post('/generate', async (req, res) => {
  const { text, count } = req.body;

  try {
    if (!AIModel) {
      await loadModel();
    }
    const AIResponse = await generateText(text, count);

    res.json({
      response: AIResponse,
    });
  } catch (error) {
    console.error('Erro ao processar o texto:', error);
    res.status(500).json({
      error: 'Erro ao processar o texto',
    });
  }
});

app.listen(port, () => {
  console.log(`Servidor executando em http://localhost:${port}`);
});
