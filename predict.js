const { exec } = require('child_process');
const ort = require('onnxruntime-node');
const fs = require('fs');
const parse = require('numpy-parser');

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

function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

async function runInference(npyPath, modelPath) {
  const buffer = fs.readFileSync(npyPath);
  const tensorData = parse(buffer);

  const session = await ort.InferenceSession.create(modelPath);
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', tensorData.data, tensorData.shape);
  const feeds = { [inputName]: tensor };

  const results = await session.run(feeds);
  const output = results[session.outputNames[0]].data;

  const probabilities = softmax(Array.from(output));
  const top5 = probabilities
    .map((prob, index) => ({ class: classNames[index], confidence: +(prob * 100).toFixed(2) }))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, 5);

  console.log("Top 5 Predictions:");
  console.table(top5);
}

// Main pipeline
async function main() {
  const imagePath = 'input.jpg';
  const npyPath = 'output.npy';
  const modelPath = 'model.onnx';

  console.log('Preprocessing image with Python...');
  exec(`python3 preprocess_image.py ${imagePath} ${npyPath}`, async (err, stdout, stderr) => {
    if (err) {
      console.error('Python preprocessing failed:\n', stderr);
      return;
    }
    console.log('Running ONNX inference...');
    await runInference(npyPath, modelPath);
  });
}

main();
