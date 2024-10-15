import { UMAP } from "umap-js"
import * as d3 from "d3"
import { TRAIN_EPOCHS } from '../utils/config';
import { createModelWithInnerOutputs } from '../utils/model'
import { useEffect, useState } from "react";
import Testing from './NewDataTest'
import './Inner.css'

export default function Inner({ data, model, predictions, epoch }) {

    const width = 450, height = 450, margin = 25;
    const NUM_IMAGES = 800;
    const groundTruth = data ? data.getTrainData(NUM_IMAGES).labels.argMax(1).dataSync() : [];

    const [similarExamples, setSimilarExamples] = useState([]); // indices of similar examples with the same label
    const [differentExamples, setDifferentExamples] = useState([]); // indices of similar examples with different labels

    // // embeddings is a 2D array of embeddings for each layer of each epoch
    // const [embeddings, setEmbeddings] = useState([...Array(TRAIN_EPOCHS)].fill(0).map(() => layerIndices.map(() => null)));
    const [embeddings, setEmbeddings] = useState([]);

    const modelWithInnerOutputs = createModelWithInnerOutputs(model);
    const LAYER_INDEX = 6
    const innerOutputs = modelWithInnerOutputs.predict(data.getTrainData(NUM_IMAGES).xs) // get embeddings from model inner layer
    const output = innerOutputs[LAYER_INDEX];
    const outputArray = output.reshape([output.shape[0], -1]).arraySync();


    const normalizeEmbedding = (embedding) => {
        // pset 2.1: normalize both x, y the range [-1, 1]
        // Find the min and max values for x and y coordinates
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;

        // Loop through the embedding to find the min/max for x and y
        for (let i = 0; i < embedding.length; i++) {
            const [x, y] = embedding[i];
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        // Normalize each point's x and y coordinates to [-1, 1]
        return embedding.map(([x, y]) => {
            const normalizedX = 2 * (x - minX) / (maxX - minX) - 1;
            const normalizedY = 2 * (y - minY) / (maxY - minY) - 1;
            return [normalizedX, normalizedY];
        });
    };


    const xScale = d3.scaleLinear().domain([-1, 1]).range([margin, width - margin]);
    const yScale = d3.scaleLinear().domain([-1, 1]).range([margin, height - margin]);
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    useEffect(() => {

        const updateEmbedding = async () => {

            // pset 2.1: generate UMAP embeddings, https://github.com/PAIR-code/umap-js

            const umap = new UMAP({ nComponents: 2, nNeighbors: 15, minDist: 0.1, nEpochs: TRAIN_EPOCHS });
            const embedding = await umap.fitAsync(outputArray);
            const normalizedEmbedding = normalizeEmbedding(embedding);
            setEmbeddings(normalizedEmbedding);
            // end of pset 2.1

        };

        updateEmbedding();

    }, [epoch]);

    return <div className="Row">
        <div className="Inner">
            <h2>UMAP of Inner Layer Outputs</h2>
            <svg width='50vw' height={height} id="Inner">
                {embeddings.length === 0 || epoch === -1 ? // only show embeddings once training is done
                    <text x={margin} y={0} textAnchor="start">Dimension Reduction will run at the end of each epoch</text>
                    : <g className="scatters">

                        <g key={`embedding`} transform={`translate(${(margin)}, 0)`}>
                            <rect x={0} y={0} width={width} height={height} fill='white' stroke="black" className="background"></rect>
                            {/* pset 2.2: draw embeddings */}{embeddings.map((point, i) => {
                                const isSimilar = similarExamples.includes(i);
                                const isDifferent = differentExamples.includes(i);

                                return (
                                    <g key={`point_${i}`}>
                                        <text
                                            x={xScale(point[0])} y={yScale(point[1])}
                                            fontSize={similarExamples.length === 0 ? 9 : (isSimilar || isDifferent) ? 14 : 9}
                                            opacity={similarExamples.length === 0 || isSimilar || isDifferent ? 1 : 0.5}
                                            fill={similarExamples.length === 0 ? colorScale(groundTruth[i]) : similarExamples.includes(i) ? 'red' : differentExamples.includes(i) ? 'blue' : 'lightgray'}
                                        >
                                            {groundTruth[i]}
                                        </text>
                                    </g>
                                );
                            })}

                            {/* end of pset 2.2 */}
                            <text x={width / 2} y={height - 2} textAnchor="middle"> Epoch {epoch} Layer {LAYER_INDEX}</text>

                        </g>

                        {/* color legend */}
                        <g className="legend" transform={`translate(${(margin)}, 0)`}>
                            {Array.from({ length: 10 }).map((_, i) => {
                                return <g key={i}>
                                    <circle cx={5} cy={i * 12 + 5} r={5} fill={colorScale(i)}></circle>
                                    <text x={15} y={i * 12 + 10} fontSize={10}>{i}</text>
                                </g>
                            })}
                        </g>
                    </g>
                }



            </svg>
        </div>

        <Testing
            model={model} data={data}
            innerOutputs={outputArray}
            LAYER_INDEX={LAYER_INDEX}
            groundTruth={groundTruth}
            similarExamples={similarExamples} differentExamples={differentExamples}
            setDifferentExamples={setDifferentExamples} setSimilarExamples={setSimilarExamples}
        />
    </div>
}