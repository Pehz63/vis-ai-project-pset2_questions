import React from 'react';
import { getURL } from '../utils/data';
import { IMAGE_W } from '../utils/config';

export const NUM_IMAGES = 100; // number of images to display

/**
 * @param {Object} props
 * @param {class} props.data
 * @param {tf.Tensor} props.predictions
 * @returns 
 */
export default function Dataset({ data, predictions }) {
    const examples = data.getTestData(NUM_IMAGES);
    let imgSRC = [];
    for (let i = 0; i < examples.xs.shape[0]; i++) {
        const image = examples.xs.slice([i, 0], [1, examples.xs.shape[1]]);
        const url = getURL(image.flatten());
        imgSRC.push(url);
    }

    const labels = examples.labels.argMax(1).dataSync(); // ground truth labels
    const predLabels = predictions ? predictions.argMax(1).dataSync() : null; // predicted labels

    return <div id="Dataset">
        <h2>Dataset</h2>
        <p>Preview of {NUM_IMAGES} test images</p>
        {imgSRC.map((url, i) => {
            return < figure key={i} style={{ display: 'inline-block', position: 'relative', margin: '10px 5px' }}>
                <img src={url} alt={`Example ${i}`} />
                {predLabels &&
                    <figcaption style={{ position: 'absolute', width: IMAGE_W, textAlign: 'center', backgroundColor: labels[i] === predLabels[i] ? 'lightgreen' : 'pink' }} > {predLabels[i]} </figcaption>
                }
            </figure>
        })}
    </div>
}

