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

// Softmax
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

// Normalize
function normalize(image, mean, std) {
  return image.map((val, idx) => {
    const c = idx % 3;
    return (val / 255 - mean[c]) / std[c];
  });
}

// Preprocess
async function preprocessImage(imagePath) {
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const { data } = await sharp(imagePath)
    .resize(224, 224)
    .raw()
    .toBuffer({ resolveWithObject: true });

  const floatData = Float32Array.from(data);
  const normalizedData = normalize(floatData, mean, std);

  const chwData = new Float32Array(3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    chwData[i] = normalizedData[i * 3];           // R
    chwData[i + 224 * 224] = normalizedData[i * 3 + 1]; // G
    chwData[i + 2 * 224 * 224] = normalizedData[i * 3 + 2]; // B
  }

  return new ort.Tensor('float32', chwData, [1, 3, 224, 224]);
}

// Predict
async function predict(modelPath, imagePath) {
  if (!fs.existsSync(modelPath)) {
    console.error(`❌ Model not found: ${modelPath}`);
    process.exit(1);
  }

  if (!fs.existsSync(imagePath)) {
    console.error(`❌ Image not found: ${imagePath}`);
    process.exit(1);
  }

  const session = await ort.InferenceSession.create(modelPath);
  const inputTensor = await preprocessImage(imagePath);
  const feeds = { [session.inputNames[0]]: inputTensor };

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;
  const probabilities = softmax(Array.from(output));

  const top5 = probabilities
    .map((prob, index) => ({
      class: classNames[index],
      confidence: +(prob * 100).toFixed(2),
      index
    }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);

  console.log(JSON.stringify(top5, null, 2));
}

// 🏃 Entry Point
const args = process.argv.slice(2);

if (args.length < 2) {
  console.error('⚠️  Please provide model path and image path.\nUsage: node index.js <model-path> <image-path>');
  process.exit(1);
}

const [modelPath, imagePath] = args;
predict(modelPath, imagePath);
