import * as d3 from 'd3';
import { useEffect } from 'react';
export default function Results({ data, predictions }) {

    const groundTruth = data ? data.getTestData().labels.argMax(1).dataSync() : []; // dataSync returns the values of the tensor as a typed array
    const predLabels = predictions ? predictions.argMax(1).dataSync() : [];

    const confusionMatrix = generateConfusionMatrix(groundTruth, predLabels);

    const width = 250, height = 250, margin = 25;

    const xScale = d3.scaleBand().domain(d3.range(10)).range([margin, width - margin]),
        yScale = d3.scaleBand().domain(d3.range(10)).range([margin, height - margin]),
        // colorScale = d3.scaleLinear().domain([0, 1]).range(['white', 'blue']);
        colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, 1]);

    const drawAxis = () => {
        d3.select('#results').selectAll('g.axis').remove();
        d3.select('#results').append('g').attr('class', 'x-axis axis').attr('transform', `translate(0, ${height - margin})`).call(d3.axisBottom(xScale))

        d3.select('#results').append('g').attr('class', 'y-axis axis').attr('transform', `translate(${height - margin}, 0)`).call(d3.axisRight(yScale))
    }

    useEffect(() => {
        drawAxis();
    },)

    return <div className="Results">
        <h2>Evaluation Results</h2>
        <svg width='100vw' height={height} id="results">
            {confusionMatrix.map((row, i) => <g key={`row_${i}`}>
                {row.map((cell, j) =>
                    <g key={`cell_${i}_${j}`}
                    ><rect key={`cell_${i}_${j}`}
                        className='cell' id={`cell-${i}-${j}`}
                        stroke='black'
                        x={xScale(j)} y={yScale(i)} fill={colorScale(cell) || 'white'}
                        width={xScale.bandwidth()} height={yScale.bandwidth()}>
                        </rect>
                        <text x={xScale(j) + xScale.bandwidth() / 2} y={yScale(i) + yScale.bandwidth() / 2} fill={cell > 0.5 ? 'white' : 'black'} textAnchor='middle' fontSize={7}>{cell ? cell.toFixed(2) : ''}</text>
                    </g>)}
            </g>)}
            <text x={width / 2} y={margin / 2} fill='black' textAnchor='middle'>Predicted</text>
            <text x={margin / 2} y={height / 2} fill='black' transform={`rotate(-90, ${margin / 2}, ${height / 2})`} textAnchor='middle'>Ground Truth</text>
            <g className='legend' transform={`translate(${width + margin}, ${margin})`}>
                {[...Array(5).keys()].map((d, i) => {
                    return <g key={i}>
                        <rect x={0} y={i * 13} width={10} height={10} fill={colorScale(i * 0.25)} stroke='gray' />
                        <text x={15} y={i * 13 + 10} fontSize={10}>{`${i * 0.25 * 100}%`}</text>
                    </g>
                }
                )}
            </g>
        </svg>
    </div>
}

function generateConfusionMatrix(groundTruth, predLabels, numClasses = 10) {
    let confusionMatrix = Array.from({ length: numClasses }, () => Array.from({ length: numClasses }, () => 0)); // initialize confusion matrix as 2D array with zeros
    for (let i = 0; i < groundTruth.length; i++) {
        confusionMatrix[groundTruth[i]][predLabels[i]]++;
    }
    // normalize the confusion matrix
    for (let i = 0; i < numClasses; i++) {
        const sum = confusionMatrix[i].reduce((a, b) => a + b, 0);
        for (let j = 0; j < numClasses; j++) {
            confusionMatrix[i][j] /= sum;
        }
    }
    return confusionMatrix;
}