import './App.css';
import React, { useState, useEffect } from 'react';
import { loadData } from './utils/data';
import { createConvModel } from './utils/model';

import Dataset from './components/Dataset';
import Results from './components/Results';
import Train from './components/Train';
import Inner from './components/Inner';
import Testing from './components/NewDataTest';

function App() {
    const [data, setData] = useState(null);
    const [predictions, setPredictions] = useState(null);
    const [epoch, setEpoch] = useState(-1);

    const [model, setModel] = useState(createConvModel());

    useEffect(() => {
        loadData().then((data) => {
            setData(data);
        });
    }, []);


    return (
        <div className="App">
            <header className="App-header">
                <h1>Visualization with AI</h1>
            </header>

            {data ? <div className='Body'>
                {/* make the data and train the same row */}

                <Dataset data={data} predictions={predictions} />
                <div className='Row'>
                    <Train model={model} data={data} setPredictions={setPredictions} setModel={setModel} epoch={epoch} setEpoch={setEpoch} />
                    <Results data={data} predictions={predictions} />
                </div>
                <Inner data={data} model={model} predictions={predictions} epoch={epoch} />
            </div> : <div className='Body'><h2>Loading Data...</h2></div>}

        </div>
    );
};

export default App;
