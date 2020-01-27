import * as wasm from "diffeq-example-wasm";

document.getElementById("btnsolve").addEventListener("click", solve, false);


function getInit() {
    const x = parseFloat(document.getElementById("posx").value);
    const y = parseFloat(document.getElementById("posy").value);
    const z = parseFloat(document.getElementById("posz").value);
    return [x, y, z];
}

function getConfig() {
    return {
        ode: document.getElementById("solver-select").value.toLowerCase(),
        init: getInit(),
        start_time: parseFloat(document.getElementById("start-time").value),
        end_time: parseFloat(document.getElementById("end-time").value),
        num_times: parseInt(document.getElementById("steps").value),
    };
}


function plot(solution) {
    const x = [];
    const y = [];
    const z = [];

    for (let i = 0; i < solution.yout.length; i++) {
        const yout = solution.yout[i];
        x.push(yout[0]);
        y.push(yout[1]);
        z.push(yout[2]);
    }

    Plotly.newPlot('plotlydiv', [{
        type: 'scatter3d',
        mode: 'lines',
        x: x,
        y: y,
        z: z,
        opacity: 1,
        line: {
            width: 4,
            colorscale: 'Blues'
        }
    }], {
        height: 640
    }, {displayModeBar: false});

    Plotly.newPlot('x-y-cut', [{
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: y
    }], {
        title: "X-Y Cut",
        height: 640
    }, {displayModeBar: false});

    Plotly.newPlot('x-z-cut', [{
        type: 'scatter',
        mode: 'lines',
        x: x,
        y: z
    }], {
        title: "X-Z Cut",
        height: 640
    }, {displayModeBar: false});

    Plotly.newPlot('y-z-cut', [{
        type: 'scatter',
        mode: 'lines',
        x: y,
        y: z
    }], {
        title: "Y-Z Cut",
        height: 620
    }, {displayModeBar: false});
}

function solve() {
    const solution = wasm.solve_lorenz_attractor(getConfig());
    plot(solution)
}

solve();