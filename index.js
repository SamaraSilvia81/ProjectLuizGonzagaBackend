const express = require('express');
const { loadLayersModel, tensor2d } = require('@tensorflow/tfjs-node');
const cors = require('cors');
const path = require('path');

const app = express();
const port = process.env.PORT || 3000;

const dictionary = {
  '\n': 0,
  ' ': 1,
  '!': 2,
  '"': 3,
  "'": 4,
  '(': 5,
  ')': 6,
  ',': 7,
  '-': 8,
  '.': 9,
  '0': 10,
  '2': 11,
  '3': 12,
  '5': 13,
  '7': 14,
  ':': 15,
  '?': 16,
  'A': 17,
  'B': 18,
  'C': 19,
  'D': 20,
  'E': 21,
  'F': 22,
  'G': 23,
  'H': 24,
  'I': 25,
  'J': 26,
  'K': 27,
  'L': 28,
  'M': 29,
  'N': 30,
  'O': 31,
  'P': 32,
  'Q': 33,
  'R': 34,
  'S': 35,
  'T': 36,
  'U': 37,
  'V': 38,
  'X': 39,
  'Z': 40,
  'a': 41,
  'b': 42,
  'c': 43,
  'd': 44,
  'e': 45,
  'f': 46,
  'g': 47,
  'h': 48,
  'i': 49,
  'j': 50,
  'k': 51,
  'l': 52,
  'm': 53,
  'n': 54,
  'o': 55,
  'p': 56,
  'q': 57,
  'r': 58,
  's': 59,
  't': 60,
  'u': 61,
  'v': 62,
  'w': 63,
  'x': 64,
  'y': 65,
  'z': 66,
  '}': 67,
  'À': 68,
  'É': 69,
  'Ê': 70,
  'Í': 71,
  'Ó': 72,
  'Ô': 73,
  'à': 74,
  'á': 75,
  'â': 76,
  'ã': 77,
  'ç': 78,
  'é': 79,
  'ê': 80,
  'í': 81,
  'ó': 82,
  'ô': 83,
  'õ': 84,
  'ú': 85,
  'ü': 86,
  '’': 87,
  '\ufeff': 88
};

const reverseDictionary = {};
for (const char in dictionary) {
  const numericValue = dictionary[char];
  reverseDictionary[numericValue] = char;
}

app.use(cors());
app.use(express.json());

let AIModel;

const loadModel = async () => {
  try {
    const modelPath = path.resolve(__dirname, 'tmp/tfjs_mobilenetv2', 'model2.json');
    AIModel = await loadLayersModel(`file://${modelPath}`);

    console.log('Modelo carregado com sucesso');
  } catch (error) {
    console.error('Erro ao carregar o modelo:', error);
  }
};

const preprocessText = (text, dictionary) => {
  const inputLength = text.length;
  const numericValues = [];

  for (let i = 0; i < inputLength; i++) {
    const char = text[i];
    const numericValue = dictionary[char];

    if (numericValue !== undefined) {
      numericValues.push(numericValue);
    } else {
      numericValues.push(-1);
    }
  }

  console.log('valor enviado para predict:', numericValues);
  return numericValues;
};

const makePrediction = async (inputData) => {
  try {
    const inputNumeric = preprocessText(inputData, dictionary);
    const inputTensor = tensor2d(inputNumeric, [1, inputNumeric.length]);
    const outputTensor = AIModel.predict(inputTensor);
    const outputData = await outputTensor.array();

    inputTensor.dispose();
    outputTensor.dispose();

    return outputData[0][0];
  } catch (error) {
    console.error('Erro ao fazer a previsão:', error);
    throw error;
  }
};

const decodeOutput = (outputData, reverseDictionary) => {
  const decodedOutput = [];
  const outputLength = outputData.length;

  for (let i = 0; i < outputLength; i++) {
    const numericValue = outputData[i];
    const char = reverseDictionary[numericValue];

    if (char !== undefined) {
      decodedOutput.push(char);
    } else {
      decodedOutput.push('a');
    }
  }

  const joinedOutput = decodedOutput.join(' ');
  return joinedOutput;
};

const findArgmax = (array) => {
  return array.reduce((maxIndex, currentValue, currentIndex, array) => {
    return currentValue > array[maxIndex] ? currentIndex : maxIndex;
  }, 0);
};

app.post('/generate', async (req, res) => {
  const { text } = req.body;
  const count = 100;

  try {
    if (!AIModel) {
      await loadModel();
    }

    const AIResponse = await makePrediction(text);
    const argmaxIndices = [];

    for (let i = 0; i < count; i++) {
      const argmaxIndex = findArgmax(AIResponse);
      console.log(argmaxIndex);
      argmaxIndices.push(argmaxIndex);
      AIResponse[argmaxIndex] = -Infinity;
    }

    const words = argmaxIndices.map((index) => reverseDictionary[index]);
    const joinedWords = words.join('');
    const formattedResponse = joinedWords.replace(/\\n/g, '\n');

    res.json({
      response: formattedResponse,
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
