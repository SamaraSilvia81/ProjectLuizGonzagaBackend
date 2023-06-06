const express = require('express');
const { loadLayersModel, tensor2d } = require('@tensorflow/tfjs-node');
const cors = require('cors');

const app = express();
app.use(cors());

const port = process.env.PORT || 3000; // Use a variável de ambiente fornecida pelo Railway para a porta

app.use(express.json());

let AIModel; // Variável para armazenar o modelo carregado

const loadModel = async () => {
  try {
    AIModel = await loadLayersModel('file://./models/model.json');
    console.log('Modelo carregado com sucesso');
  } catch (error) {
    console.error('Erro ao carregar o modelo:', error);
  }
};

const makePrediction = async (text) => {
    const tensor = tensor2d([text]);
    const tensorResult = AIModel.predict(tensor);
    const response = await tensorResult.array();
    return response[0];
};

// Rota para obter o texto da AI
app.get('/', async (req, res) => {
    res.send('Welcome to Project AI Luiz Gonzaga !!!');
});

// Rota para receber o texto do front-end e enviá-lo para a IA
app.post('/generate', async (req, res) => {
    const { text } = req.body;

    try {
        if (!AIModel) {
            // Carregar o modelo somente na primeira chamada à rota
            await loadModel();
        }

        // Fazer a previsão com base no texto
        const AIResponse = await makePrediction(text);

        // Retornar a resposta da IA para o front-end
        res.json({ response: AIResponse });
    } catch (error) {
        console.error('Erro ao processar o texto:', error);
        res.status(500).json({ error: 'Erro ao processar o texto' });
    }
});

// Rota para obter o texto da AI
app.get('/generate', async (req, res) => {
    const { text } = req.query;

    try {
        if (!AIModel) {
            // Carregar o modelo somente na primeira chamada à rota
            await loadModel();
        }

        // Fazer a previsão com base no texto
        const AIResponse = await makePrediction(text);

        // Retornar a resposta da AI
        res.json({ response: AIResponse });
    } catch (error) {
        console.error('Erro ao processar o texto:', error);
        res.status(500).json({ error: 'Erro ao processar o texto' });
    }
});

app.listen(port, () => {
    console.log(`Servidor executando em http://localhost:${port}`);
});
