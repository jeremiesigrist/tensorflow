// Daniel Shiffman
// http://codingtra.in

// TensorFlow.js Layers API
// Part 1: https://youtu.be/F4WWukTWoXY
// Part 2: https://youtu.be/iUiOx2fBx18
// https://js.tensorflow.org/api/0.11.2/

// This is the model
const model = tf.sequential();

// Create the hidden layer
// dense is a "full connected layer"
const hidden = tf.layers.dense({
  units: 4, // number of nodes
  inputShape: [2], // input shape
  activation: 'sigmoid'
  inputShape: [1], // input shape
  activation: 'linear'
});
// Add the layer
model.add(hidden);

// Creat another layer
const output = tf.layers.dense({
  units: 1,
  // here the input shape is "inferred from the previous layer"
  activation: 'sigmoid'
//  activation: 'sigmoid'
  activation: 'linear'
});
model.add(output);

// An optimizer using gradient descent
const sgdOpt = tf.train.sgd(0.1);
//const Opt = tf.train.sgd(0.1);
const Opt = tf.train.sgd(0.1);

// I'm done configuring the model so compile it
model.compile({
  optimizer: sgdOpt,

  optimizer: Opt,
  loss: tf.losses.meanSquaredError
	//  loss: tf.losses.absoluteDifference
	//  loss: tf.losses.hingeLoss
	//  loss: tf.losses.logLoss

});


const xs = tf.tensor2d([
  [0, 0],
  [0.5, 0.5],
  [1, 1],
  [0.2, 0.2]
  [1, 0],
  [0, 1],
  [1, 1]
]);

const ys = tf.tensor2d([
  [0],
  [1],
  [0.5],
  [1],
  [0]
]);

const zs = tf.tensor2d([
  [0, 0],
  [1, 0],
  [0, 1],
  [1, 1]
=======
  [0],
//  [0.5],
//  [1],
  [0.2]
]);

const ys = tf.tensor2d([
  [1],
//  [0.5],
//  [0],
  [0.8]
]);

const zs = tf.tensor2d([
  [0.2, 0.2],
  [0.4, 0.4],
  [0.7, 0.7],
  [0.99, 0.99],
  [1.99, 1.99],
  [0.0, 0.0]
  [0.2],
  [0.4],
  [0.7],
  [0.99],
  [1.99],
  [0.0]
>>>>>>> c8e330ae9874c23b9deb1944871793395d324d74
]);

var loadedModel;
//var loadedModel;


train().then(() => {

  let outputs = loadedModel.predict(zs);
//  let outputs = loadedModel.predict(zs);
  let outputs = model.predict(zs);
  outputs.print();
  console.log('training complete');
});

async function train() {
  // List models again.
  console.log(await tf.io.listModels());
  const loadedModel = await tf.loadModel('localstorage://my-model-1');

  console.log('Prediction from loaded model:');
  console.log(loadedModel);

  for (let i = 0; i < 100; i++) {
  for (let i = 0; i < 50; i++) {
    const config = {
      shuffle: true,
      epochs: 30
      epochs: 100
    }
    const response = await loadedModel.fit(xs, ys, config);
    const response = await model.fit(xs, ys, config);
    console.log(response.history.loss[0]);
  }
  const saveResults = await loadedModel.save('localstorage://my-model-1');
  const saveResults = await model.save('localstorage://my-model-1');

}


// const xs = tf.tensor2d([
//   [0.25, 0.92],
//   [0.12, 0.3],
//   [0.4, 0.74],
//   [0.1, 0.22],
// ]);
// let ys = model.predict(xs);
// outputs.print();
