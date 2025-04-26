const ort = require('onnxruntime-node');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// Class names
const classNames = [
  'Tomato_Rot', 'Apple_Fresh', 'Banana_Fresh', 'Banana_Anthracnose', 'Mango_Fresh', 'Okra_Rot',
  'Peach_Scab', 'Tomato_Fresh', 'Mango_Black Mould Rot', 'Grape_Gray Mold', 'Papaya_Ring Spot',
  'Strawberry_Powdery Mildew', 'Orange_Pencilium Decay', 'Lime_Fresh', 'Mango_Alternaria',
  'Mango_Anthracnose', 'Papaya_Fresh', 'Cucumber_Fresh', 'Apple_Sooty Blotch', 'Capsicum_Fresh',
  'Papaya_Powdery Mildew', 'Lime_Rotten', 'Guava_Fresh', 'Peach_Brown Rot', 'Peach_Fresh',
  'Grape_Powdery Mildew', 'Mango_Stem End Rot', 'Strawberry_Fresh', 'Orange_Citrus Black Spot',
  'Guava_Rotten', 'Apple_Moldy Core', 'Banana_Cigar End Rot', 'Cucumber_Rot', 'Papaya_Anthracnose',
  'Papaya_Black Spot', 'Orange_Sour Rot', 'Strawberry_Gray Mold', 'Orange_Fresh', 'Apple_Scab',
  'Capsicum_Stale', 'Banana_Thrips Damage', 'Grape_Fresh', 'Okra_Fresh', 'Papaya_Phytophthora',
  'Apple_Black Rot'
];

// Softmax function
function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

// Normalize image
function normalize(image, mean, std) {
  const normalized = image.map((val, idx) => {
    const c = idx % 3;
    return (val / 255 - mean[c]) / std[c];
  });
  return normalized;
}

// Preprocess image to match model input shape: (1, 3, 224, 224)
async function preprocessImage(imagePath) {
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const { data, info } = await sharp(imagePath)
    .resize(224, 224)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const floatData = Float32Array.from(data);
  const normalizedData = normalize(floatData, mean, std);

  const chwData = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    chwData[i] = normalizedData[i * 3];       // R
    chwData[i + 224 * 224] = normalizedData[i * 3 + 1]; // G
    chwData[i + 2 * 224 * 224] = normalizedData[i * 3 + 2]; // B
  }

  return new ort.Tensor('float32', chwData, [1, 3, 224, 224]);
}

// Predict function
async function predict(imagePath, modelPath) {
  const session = await ort.InferenceSession.create(modelPath);

  const inputTensor = await preprocessImage(imagePath);
  const inputName = session.inputNames[0];
  const feeds = { [inputName]: inputTensor };

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  const probabilities = softmax(Array.from(output));
  const maxIdx = probabilities.indexOf(Math.max(...probabilities));
  const label = classNames[maxIdx];
  const confidence = (probabilities[maxIdx] * 100).toFixed(2);

  console.log(`Prediction: ${label}, Confidence: ${confidence}%`);
}

async function predict_multiple(imagePath, modelPath) {
    const session = await ort.InferenceSession.create(modelPath);
  
    const inputTensor = await preprocessImage(imagePath);
    const inputName = session.inputNames[0];
    const feeds = { [inputName]: inputTensor };
  
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data;
  
    const probabilities = softmax(Array.from(output));
  
    // Get top 5 predictions
    const top5 = probabilities
      .map((prob, index) => ({ class: classNames[index], confidence: +(prob * 100).toFixed(2), index }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);
  
    console.log(JSON.stringify(top5, null, 2));
    return top5;
}
  
// Example usage
const imagePath = './images.jpg';
const modelPath = './model.onnx';

predict(imagePath, modelPath);

predict_multiple(imagePath, modelPath);