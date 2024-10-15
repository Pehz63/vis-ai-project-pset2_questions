import { ReactSketchCanvas } from "react-sketch-canvas";
import { useRef, useState } from "react";
import { base64ToTensor, getURL } from "../utils/data";
import * as d3 from 'd3';
import { NUM_CLASSES } from '../utils/config'
import { createModelWithInnerOutputs } from "../utils/model";

export default function Testing({ model, groundTruth, innerOutputs, LAYER_INDEX, data }) {

    const drawWidth = 150, drawHeight = 150, margin = 10;
    const barChartHeight = 180, barChartWidth = 100;
    const drawingRef = useRef(null);
    const [newPrediction, setNewPrediction] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [similarExamples, setSimilarExamples] = useState([]); // indices of similar examples with the same label
    const [differentExamples, setDifferentExamples] = useState([]); // indices of similar examples with different labels

    const clearCanvas = () => {
        drawingRef.current.clearCanvas();
        setNewPrediction(null);
        setDifferentExamples([]);
        setSimilarExamples([]);
    }
    const onPredict = async () => {
        setSimilarExamples([]);
        setDifferentExamples([]);
        const dataURL = await drawingRef.current.exportImage('png')
        const tensor = await base64ToTensor(dataURL, drawWidth, drawHeight);
        const prediction = await model.predict(tensor).dataSync();
        setNewPrediction(Array.from(prediction));
        setTimeout(() => findExamples(tensor, prediction.indexOf(Math.max(...prediction)), innerOutputs), 10); // wait for prediction to update

    }

    const findExamples = async (tensor, currentLabel, innerOutputs) => {
        let sameLabelExamples = []; // indices of similar examples with the same label
        let sameLabelDistances = [];
        let differentLabelExamples = []; // indices of similar examples with different labels
        let differentLabelDistances = [];

        const modelWithInnerOutputs = createModelWithInnerOutputs(model);
        const currentOutput = modelWithInnerOutputs.predict(tensor)[LAYER_INDEX].arraySync();

        // pset 2.3: find similar and counterfactual examples, update sameLabelExamples and differentLabelExamples

        const updateTopNExamples = (examples = [], distances = [], newExample, newDistance, topN = 3) => {
            let n = examples?.length;
        
            // Insert newDistance and newExample maintaining sorted distances
            for (let i = 0; i <= n; i++) {
                if (i === n || newDistance < distances[i]) {
                    distances.splice(i, 0, newDistance); // Insert newDistance at correct position
                    examples.splice(i, 0, newExample);   // Insert newExample at the same position
                    break;
                }
            }
        
            // Truncate arrays to the first topN elements
            return {
                examples: examples.slice(0, topN),
                distances: distances.slice(0, topN)
            };
        };        

        const calculateL2Distance = (a, b) => {
            return a.reduce((acc, curr, i) => acc + Math.pow(curr - b[i], 2), 0);
        };

        innerOutputs.forEach((example, index) => {
            const distance = calculateL2Distance(example, currentOutput); // (optional): try different distance metrics
            if (groundTruth[index] === currentLabel) {
                const updatedSameLabel = updateTopNExamples(sameLabelExamples, sameLabelDistances, index, distance);
                sameLabelExamples = updatedSameLabel.examples;
                sameLabelDistances = updatedSameLabel.distances;
            } else {
                const updatedDifferentLabel = updateTopNExamples(differentLabelExamples, differentLabelDistances, index, distance);
                differentLabelExamples = updatedDifferentLabel.examples;
                differentLabelDistances = updatedDifferentLabel.distances;
            }
        })
        // end of pset 2.3

        setDifferentExamples(differentLabelExamples);
        setSimilarExamples(sameLabelExamples);

    }

    const xScale = d3.scaleLinear().domain([0, 1]).range([margin, barChartWidth - 2 * margin]);
    const yScale = d3.scaleBand().domain([...Array(NUM_CLASSES).keys()]).range([margin, barChartHeight - margin]).paddingInner(0.1);

    return <div className="Test">
        <h2>Test</h2>
        <div style={{ display: 'inline-block' }}>
            <ReactSketchCanvas ref={drawingRef} width={drawWidth} height={drawHeight} strokeWidth={20} strokeColor="white" canvasColor="black" />
            <button onClick={clearCanvas}>Clear</button> <button onClick={onPredict}>Predict</button>
        </div>
        <div style={{ display: 'inline-block' }}>
            <svg width={barChartWidth} height={barChartHeight}>
                {newPrediction && newPrediction.map((d, i) => {
                    return <g key={i}>
                        <text x={margin} y={yScale(i) + yScale.bandwidth() / 2} textAnchor='end' alignmentBaseline='middle' fontSize='10'>{i}</text>
                        <rect key={i} x={margin} y={yScale(i)} width={xScale(d)} height={yScale.bandwidth()} fill='skyblue' />
                        <text x={xScale(d) + margin} y={yScale(i) + yScale.bandwidth() / 2} textAnchor='start' alignmentBaseline='middle' fontSize='10'>{d.toFixed(2)}</text>
                    </g>
                })}
            </svg>
        </div>
        {newPrediction && similarExamples.length > 0 ?
            <div >
                <h3>Similar Examples</h3>
                <div>
                    {similarExamples.map((index, i) => {
                        const example = data.getTrainData().xs.slice([index, 0], [1, data.getTrainData().xs.shape[1]]);
                        const url = getURL(example.flatten());
                        return <img key={i} src={url} alt={`Example ${index}`} />
                    })}
                </div>

                <h3>Counterfactual Examples</h3>
                <div>
                    {differentExamples.map((index, i) => {
                        const example = data.getTrainData().xs.slice([index, 0], [1, data.getTrainData().xs.shape[1]]);
                        const url = getURL(example.flatten());
                        return <img key={i} src={url} alt={`Example ${index}`} />
                    })}
                </div>

            </div>
            :
            <div style={{ display: 'inline-block', height: { drawHeight } }}>
                <h3>Fetching examples...</h3>
            </div>}
    </div>
}