export const IMAGE_H = 28;
export const IMAGE_W = 28;
export const IMAGE_SIZE = IMAGE_H * IMAGE_W;

export const NUM_CLASSES = 10;
export const NUM_DATASET_ELEMENTS = 65000;

export const NUM_TRAIN_ELEMENTS = 55000;
export const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

export const BATCH_SIZE = 64;

// export const SAMPLE_RATIO = 0.15
// export const TRAIN_EPOCHS = 3;
// export const BATCH_STEP = 10 // save the log every certain batches
// if the model is slow, you can reduce the number of training examples but need to increase the number of epochs and reduce the batch step
export const SAMPLE_RATIO = 0.01
export const TRAIN_EPOCHS = 40;
export const BATCH_STEP = 2;

export const SAMPLE_TRAIN_ELEMENTS = NUM_TRAIN_ELEMENTS * SAMPLE_RATIO
export const SAMPLE_TEST_ELEMENTS = NUM_TEST_ELEMENTS * SAMPLE_RATIO;

export const VALIDATION_SPLIT = 0.15; // 15% of the training data is used for validation at the end of epochs


export const TOTAL_BATCHS =
Math.ceil(SAMPLE_TRAIN_ELEMENTS * (1 - VALIDATION_SPLIT) / BATCH_SIZE) * TRAIN_EPOCHS 


export const MNIST_IMAGES_SPRITE_PATH =
'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
export const MNIST_LABELS_PATH =
'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';