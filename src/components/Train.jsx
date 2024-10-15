
import { createModelWithInnerOutputs, train } from '../utils/model'
import React, { useEffect, useState } from 'react';
import { NUM_IMAGES } from './Dataset';
import { BATCH_STEP, TOTAL_BATCHS, TRAIN_EPOCHS } from '../utils/config';
import * as d3 from 'd3';

export default function Train({ model, data, setPredictions, setModel, epoch, setEpoch }) {
    const [trainLogs, setTrainLogs] = useState([]); // trainLogs[i] is the training logs for the i-th batch
    const [valLogs, setValLogs] = useState([]); // valLogs[i] is the validation logs for the i-th epoch, shorter than trainLogs
    const [batch, setBatch] = useState(0); // batch number [0, TOTAL_BATCHS

    const [status, setStatus] = useState('Model Loaded');

    const onIteration = (text, progress, logs) => {
        if (text === 'onBatchEnd') {
            setTrainLogs(prevTrainLogs => {
                const newTrainLogs = [...prevTrainLogs, logs];
                return newTrainLogs;
            })
            setBatch(progress);
            setPredictions(model.predict(data.getTestData().xs));

        }
        if (text === 'onEpochEnd') {
            console.log('Epoch End');
            setValLogs(prevValLogs => {
                const newValLogs = [...prevValLogs, logs];
                return newValLogs;
            })
            // setProgress(progress);
            setPredictions(model.predict(data.getTestData().xs));
            setModel(model);
            setEpoch(progress);

        }
    }

    const startTraining = async () => {
        setStatus('Training...')
        await train(model, data, onIteration);
        setStatus('Done!');

        setPredictions(model.predict(data.getTestData(NUM_IMAGES).xs));
        setBatch(TOTAL_BATCHS); // the last serval batches may not be logged as not enough for a batch step
        setModel(model);
    }

    const Logs = trainLogs.length > 0 ? <>
        <div>
            <b>Accuracy:</b>{trainLogs[trainLogs.length - 1]['acc'].toFixed(3)} {`  `}
            <b>Validation Accuracy:</b>{valLogs.length > 0 && valLogs[valLogs.length - 1]['val_acc'].toFixed(3)}
        </div>
        <div>
            <b>Loss:</b>{trainLogs[trainLogs.length - 1]['loss'].toFixed(3)} {`  `}
            <b>Validation Loss:</b>{valLogs.length > 0 && valLogs[valLogs.length - 1]['val_loss'].toFixed(3)}
        </div>
    </> :
        null

    const width = 200, height = 150, margin = 25;
    const xScale = d3.scaleLinear().domain([0, Math.max(1, trainLogs.length * BATCH_STEP)]).range([margin, width - margin]),
        accYScale = d3.scaleLinear().domain([0, 1]).range([height - margin, margin]),
        lossYScale = d3.scaleLinear().domain([0, Math.max(...trainLogs.map(d => d['loss']))]).range([height - margin, margin]);

    const accLineChart = trainLogs.map((log, i) => {
        if (i === 0) return null;
        const x1 = xScale((i - 1) * BATCH_STEP), x2 = xScale(i * BATCH_STEP),
            y1 = accYScale(trainLogs[i - 1]['acc']), y2 = accYScale(trainLogs[i]['acc'])
        return <line key={i} x1={x1} x2={x2} y1={y1} y2={y2} stroke='orange' strokeWidth={2} />
    })

    const accValLineChart = valLogs.map((log, i) => {
        // number of batches in each epoch
        const numBatches = TOTAL_BATCHS / TRAIN_EPOCHS;
        const x1 = xScale(i * numBatches), x2 = xScale((i + 1) * numBatches),
            y1 = accYScale(valLogs[i]['val_acc']), y2 = y1
        return <line key={i} x1={x1} x2={x2} y1={y1} y2={y2} stroke='blue' strokeWidth={2} />
    })

    const lossLineChart = trainLogs.map((log, i) => {
        if (i === 0) return null;
        const x1 = xScale((i - 1) * BATCH_STEP), x2 = xScale(i * BATCH_STEP),
            y1 = lossYScale(trainLogs[i - 1]['loss']), y2 = lossYScale(trainLogs[i]['loss'])
        return <line key={i} x1={x1} x2={x2} y1={y1} y2={y2} stroke='orange' strokeWidth={2} />
    })

    const lossValLineChart = valLogs.map((log, i) => {
        const numBatches = TOTAL_BATCHS / TRAIN_EPOCHS;
        const x1 = xScale(i * numBatches), x2 = xScale((i + 1) * numBatches),
            y1 = lossYScale(valLogs[i]['val_loss']), y2 = y1
        return <line key={i} x1={x1} x2={x2} y1={y1} y2={y2} stroke='blue' strokeWidth={2} />
    })

    const drawAxis = () => {
        d3.select('#training').selectAll('g.axis').remove();
        const tickNum = Math.min(trainLogs.length, 5);

        d3.select('#training')
            .select('g.acc')
            .append('g').attr('class', 'x-axis axis').attr('transform', `translate(0, ${height - margin})`)
            .call(d3.axisBottom(xScale).tickFormat(d3.format('d')).ticks(tickNum))

        d3.select('#training')
            .select('g.acc')
            .append('g').attr('class', 'y-axis axis').attr('transform', `translate(${margin}, 0)`).call(d3.axisLeft(accYScale).ticks(tickNum))


        d3.select('#training')
            .select('g.loss')
            .append('g').attr('class', 'x-axis axis').attr('transform', `translate(0, ${height - margin})`).call(d3.axisBottom(xScale).tickFormat(d3.format('d')).ticks(tickNum))

        d3.select('#training')
            .select('g.loss')
            .append('g').attr('class', 'y-axis axis').attr('transform', `translate(${margin}, 0)`).call(d3.axisLeft(lossYScale).ticks(tickNum))
    }

    useEffect(drawAxis, [trainLogs])

    return <div id="Train">
        <h2>Model Training</h2>
        {status != 'Training...' && <button onClick={startTraining}>
            Click to Start Training
        </button>}
        <span id='status'><b>Status:</b> {status}</span>
        <br />
        <svg width='50vw' height={height} id="training">
            <g className='acc'>
                <text className='acc' x={width / 2} y={margin * 0.8} fill='black' textAnchor='middle'>Accuracy</text>
                <g className='accChart chart'>
                    <g className='train'>{accLineChart}</g>
                    <g className='val'>{accValLineChart}</g>
                </g>
                <text x={width / 2} y={height} textAnchor='middle' fontSize={10}>#Batch</text>
            </g>

            <g className='loss' transform={`translate(${width + margin}, 0)`}>
                <text className='loss' x={width / 2} y={margin * 0.8} fill='black' textAnchor='middle'>Loss</text>
                <g className='lossChart chart'>
                    <g className='train'>{lossLineChart} </g>
                    <g className='val'>{lossValLineChart}</g>
                </g>
                <text x={width / 2} y={height} textAnchor='middle' fontSize={10}>#Batch</text>
            </g>
            <g className='legend' transform={`translate(${width * 2 + margin}, ${margin})`}>
                <rect x={0} y={0} width={10} height={10} fill='orange'></rect>
                <text x={15} y={10} fontSize={10}>Training</text>
                <rect x={0} y={15} width={10} height={10} fill='blue'></rect>
                <text x={15} y={25} fontSize={10}>Validation</text>
            </g>
        </svg>

        {Logs}
        <div id='progress'><b>Progress:</b> {(batch / TOTAL_BATCHS * 100).toFixed(1)}%</div>

    </div>
}