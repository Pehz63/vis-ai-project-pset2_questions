import * as tf from '@tensorflow/tfjs';
import { IMAGE_H, IMAGE_W, NUM_CLASSES, SAMPLE_TRAIN_ELEMENTS, SAMPLE_TEST_ELEMENTS, MNIST_IMAGES_SPRITE_PATH, MNIST_LABELS_PATH, NUM_DATASET_ELEMENTS, IMAGE_SIZE, NUM_TRAIN_ELEMENTS, NUM_TEST_ELEMENTS } from './config';




/**
 * A class that fetches the sprited MNIST dataset and provide data as
 * tf.Tensors.
 */
export class MnistData {
  constructor() { }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true } );
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        img.width = img.naturalWidth;
        img.height = img.naturalHeight;

        const datasetBytesBuffer =
          new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array(
            datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
            IMAGE_SIZE * chunkSize);
          ctx.drawImage(
            img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
            chunkSize);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            datasetBytesView[j] = imageData.data[j * 4] / 255;
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] =
      await Promise.all([imgRequest, labelsRequest]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());

    // // Slice the the images and labels into train and test sets.
    // this.trainImages =
    //     this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    // this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    // this.trainLabels =
    //     this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    // this.testLabels =
    //     this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);


    // samples for training testing
    function getRandomSample(arr, sampleSize) {
      const shuffled = arr.map((value, index) => ({ value, index })).sort(() => 0.5 - Math.random());
      const selected = shuffled.slice(0, sampleSize).map(item => item.index);
      return selected;
    }
    const sampleTrainIndices = getRandomSample(Array.from({ length: NUM_TRAIN_ELEMENTS }, (_, i) => i), SAMPLE_TRAIN_ELEMENTS);
    const sampleTestIndices = getRandomSample(Array.from({ length: NUM_TEST_ELEMENTS }, (_, i) => i), SAMPLE_TEST_ELEMENTS);


    this.trainImages = new Float32Array(SAMPLE_TRAIN_ELEMENTS*IMAGE_SIZE)
    this.testImages = new Float32Array(SAMPLE_TEST_ELEMENTS*IMAGE_SIZE)
    this.trainLabels = new Uint8Array(SAMPLE_TRAIN_ELEMENTS*NUM_CLASSES)
    this.testLabels = new Uint8Array(SAMPLE_TEST_ELEMENTS*NUM_CLASSES);

    sampleTrainIndices.forEach((index, i) => {
     
      this.trainImages.set(this.datasetImages.slice(index * IMAGE_SIZE, (index + 1) * IMAGE_SIZE), i * IMAGE_SIZE);
      this.trainLabels.set(this.datasetLabels.slice(index * NUM_CLASSES, (index + 1) * NUM_CLASSES), i * NUM_CLASSES);
    });
    sampleTestIndices.forEach((index, i) => {
      this.testImages.set(this.datasetImages.slice((NUM_TRAIN_ELEMENTS + index) * IMAGE_SIZE, (NUM_TRAIN_ELEMENTS + index + 1) * IMAGE_SIZE), i * IMAGE_SIZE);
      this.testLabels.set(this.datasetLabels.slice((NUM_TRAIN_ELEMENTS + index) * NUM_CLASSES, (NUM_TRAIN_ELEMENTS + index + 1) * NUM_CLASSES), i * NUM_CLASSES);
    });


  }

  /**
   * Get all training data as a data tensor and a labels tensor.
   *
   * @returns
   *   xs: The data tensor, of shape `[numTrainExamples, 28, 28, 1]`.
   *   labels: The one-hot encoded labels tensor, of shape
   *     `[numTrainExamples, 10]`.
   */
  getTrainData(numExamples) {
    const xs = tf.tensor4d(
      this.trainImages,
      [this.trainImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
    const labels = tf.tensor2d(
      this.trainLabels, [this.trainLabels.length / NUM_CLASSES, NUM_CLASSES]);
    if (numExamples != null && numExamples < this.trainImages.length / IMAGE_SIZE) {
      return {
        xs: xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1]),
        labels: labels.slice([0, 0], [numExamples, NUM_CLASSES])
      };
    } else return { xs, labels };
  }

  /**
   * Get all test data as a data tensor and a labels tensor.
   *
   * @param {number} numExamples Optional number of examples to get. If not
   *     provided,
   *   all test examples will be returned.
   * @returns
   *   xs: The data tensor, of shape `[numTestExamples, 28, 28, 1]`.
   *   labels: The one-hot encoded labels tensor, of shape
   *     `[numTestExamples, 10]`.
   */
  getTestData(numExamples) {
    let xs = tf.tensor4d(
      this.testImages,
      [this.testImages.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);
    let labels = tf.tensor2d(
      this.testLabels, [this.testLabels.length / NUM_CLASSES, NUM_CLASSES]);

    if (numExamples != null) {
      xs = xs.slice([0, 0, 0, 0], [numExamples, IMAGE_H, IMAGE_W, 1]);
      labels = labels.slice([0, 0], [numExamples, NUM_CLASSES]);
    }
    return { xs, labels };
  }
}

export async function loadData() {
  const data = new MnistData();
  await data.load();
  return data;
}

// helper function that converts tensor to image URL
export function getURL(image) {
  const canvas = document.createElement('canvas');
  const [width, height] = [28, 28];
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = image.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    imageData.data[j + 0] = data[i] * 255;
    imageData.data[j + 1] = data[i] * 255;
    imageData.data[j + 2] = data[i] * 255;
    imageData.data[j + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
}

export async function base64ToTensor(base64) {
  return new Promise((resolve, reject) => {
      // Create an image element
      const img = new Image();
      img.crossOrigin = 'Anonymous'; // Use this if the image is served from a different domain
      img.src = base64;

      img.onload = () => {
          // Set up canvas
          const canvas = document.createElement('canvas');
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext('2d', { willReadFrequently: true });

          // Draw image on canvas
          ctx.drawImage(img, 0, 0);

          // Get image data from canvas
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          // Convert the image data to a tensor
          let tensor = tf.browser.fromPixels(imageData);

          // Resize the tensor to 28x28
          tensor = tf.image.resizeBilinear(tensor, [28, 28]);

          // Convert to grayscale
          tensor = tf.mean(tensor, 2);

          // Expand dimensions to match shape [1, 28, 28, 1]
          tensor = tensor.expandDims(0).expandDims(-1);
          // Normalize the tensor to have values between 0 and 1
          tensor = tensor.div(255.0);
        //   // Invert colors
        //   tensor = tf.scalar(1.0).sub(tensor);
          resolve(tensor);
      };

      img.onerror = (error) => {
          reject(error);
      };
  });
}
