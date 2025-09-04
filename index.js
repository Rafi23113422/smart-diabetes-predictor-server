const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.send('SmartDiabetes Node Server is running');
});

const ML_API_URL = 'http://localhost:5001/api'; // Updated to match Flask port

app.post('/api/full_details_prediction', async (req, res) => {
  try {
    const response = await axios.post(`${ML_API_URL}/full_details_prediction`, req.body, { timeout: 10000 });
    res.json(response.data);
  } catch (error) {
    console.error('Full prediction error:', error.message);
    res.status(500).json({ error: 'Failed to get full prediction' });
  }
});

app.post('/api/simple_details_prediction', async (req, res) => {
  try {
    const response = await axios.post(`${ML_API_URL}/simple_details_prediction`, req.body, { timeout: 10000 });
    res.json(response.data);
  } catch (error) {
    console.error('Simple prediction error:', error.message);
    res.status(500).json({ error: 'Failed to get simple prediction' }); 
  }
});

app.listen(port, () => {
  console.log(`Node.js server listening on port ${port}`);
});
