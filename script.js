// --- DOM Elements ---
const algorithmSelect = document.getElementById('algorithm-select');
const epsilonControl = document.getElementById('epsilon-control');
const epsilonSlider = document.getElementById('epsilon-slider');
const epsilonValueSpan = document.getElementById('epsilon-value');
const ucbControl = document.getElementById('ucb-control');
const ucbSlider = document.getElementById('ucb-slider');
const ucbValueSpan = document.getElementById('ucb-value');
const thompsonControl = document.getElementById('thompson-control');
const startPauseBtn = document.getElementById('start-pause-btn');
const stepBtn = document.getElementById('step-btn');
const resetBtn = document.getElementById('reset-btn');
const speedSlider = document.getElementById('speed-slider');
const speedValueSpan = document.getElementById('speed-value');
const banditsContainer = document.getElementById('bandits-container');
const stepCounterSpan = document.getElementById('step-counter');
const totalRewardSpan = document.getElementById('total-reward');
const actionFeedbackP = document.getElementById('action-feedback');
const optimalRewardSpan = document.getElementById('optimal-reward');
const regretSpan = document.getElementById('regret');
const currentAlgorithmName = document.getElementById('current-algorithm-name');
const algorithmExplanation = document.getElementById('algorithm-explanation');
const algorithmFormula = document.getElementById('algorithm-formula');

// Config panel elements
const configToggleBtn = document.getElementById('config-toggle');
const configPanel = document.getElementById('config-panel');
const banditConfigContainer = document.getElementById('bandit-config-container');
const applyConfigBtn = document.getElementById('apply-config');

// Chart contexts
const rewardChartCtx = document.getElementById('reward-chart').getContext('2d');
const pullsChartCtx = document.getElementById('pulls-chart').getContext('2d');
const regretChartCtx = document.getElementById('regret-chart').getContext('2d');
const exploreExploitChartCtx = document.getElementById('explore-exploit-chart').getContext('2d');

// --- Simulation State ---
let bandits = [];
let NUM_BANDITS = 4; // Number of arms/bandits
let step = 0;
let totalReward = 0;
let optimalReward = 0;
let regret = 0;
let isRunning = false;
let simulationInterval;
let intervalTime = 500; // ms
let rewardHistory = []; // For reward chart
let optimalRewardHistory = []; // For optimal theoretical reward
let regretHistory = []; // For regret chart
let exploreCount = 0;
let exploitCount = 0;
let explorationHistory = []; // For exploration vs exploitation chart

let lastAction = {
    type: null, // 'explore' or 'exploit'
    reason: '',
    banditId: -1
};

// --- Algorithm Parameters ---
let epsilon = 0.1;
let ucbC = 2; // Exploration factor for UCB

// --- Chart Instances ---
let rewardChart;
let pullsChart;
let regretChart;
let exploreExploitChart;

// --- Bandit Configuration (True Probabilities) ---
let TRUE_PROBABILITIES = [0.2, 0.8, 0.5, 0.6]; // Default probabilities
let OPTIMAL_ARM_INDEX = TRUE_PROBABILITIES.indexOf(Math.max(...TRUE_PROBABILITIES));
let OPTIMAL_PROBABILITY = Math.max(...TRUE_PROBABILITIES);

// --- Algorithm Descriptions ---
const algorithmDescriptions = {
    'greedy': {
        name: 'Greedy (Ganancioso)',
        description: 'O algoritmo Greedy sempre escolhe o braço com a maior estimativa de recompensa, sem exploração. Pode ficar preso em máximos locais.',
        formula: 'a_t = argmax_a Q_t(a)'
    },
    'epsilon-greedy': {
        name: 'ε-Greedy (Epsilon-Ganancioso)',
        description: 'O algoritmo ε-Greedy escolhe uma ação aleatória com probabilidade ε, e a melhor ação conhecida com probabilidade 1-ε. Equilibra exploração e exploração.',
        formula: 'P(a) = ε/|A| + (1-ε) × 1{a = argmax_a Q_t(a)}'
    },
    'ucb': {
        name: 'UCB (Upper Confidence Bound)',
        description: 'UCB seleciona ações com base na incerteza. Cada ação recebe um bônus de exploração baseado na quantidade de vezes que foi escolhida.',
        formula: 'a_t = argmax_a [Q_t(a) + c × √(ln t / N_t(a))]'
    },
    'thompson': {
        name: 'Thompson Sampling',
        description: 'Thompson Sampling mantém uma distribuição de probabilidade sobre as recompensas de cada ação e escolhe baseado em amostras dessas distribuições.',
        formula: 'Para cada braço a, amostra θ_a ~ Beta(S_a+1, F_a+1)'
    }
};

// --- Beta Distribution for Thompson Sampling ---
function betaDistribution(alpha, beta) {
    // Simple approximation of beta distribution sampling
    // Using Gamma distribution approximation
    const u = Math.random();
    const v = Math.random();
    const x = Math.pow(u, 1/alpha);
    const y = Math.pow(v, 1/beta);
    return x / (x + y);
}

// --- Initialization ---
function initializeBandits() {
    banditsContainer.innerHTML = ''; // Clear previous bandits
    bandits = [];
    for (let i = 0; i < NUM_BANDITS; i++) {
        const bandit = {
            id: i,
            trueProbability: TRUE_PROBABILITIES[i],
            estimatedValue: 0, // Q-value estimate
            pullCount: 0,     // N(a)
            successCount: 0,  // For Thompson Sampling
            failureCount: 0,  // For Thompson Sampling
            element: createBanditElement(i), // Reference to the DOM element
            ucbValue: 0       // For UCB visualization
        };
        bandits.push(bandit);
        banditsContainer.appendChild(bandit.element);
    }
    
    // Update the optimal arm
    OPTIMAL_ARM_INDEX = TRUE_PROBABILITIES.indexOf(Math.max(...TRUE_PROBABILITIES));
    OPTIMAL_PROBABILITY = Math.max(...TRUE_PROBABILITIES);
    
    // Initialize configuration panel
    initBanditConfigPanel();
}

function createBanditElement(id) {
    const div = document.createElement('div');
    div.className = 'bandit';
    div.id = `bandit-${id}`;
    
    // Create extra info based on algorithm
    let extraInfo = '';
    if (algorithmSelect.value === 'thompson') {
        extraInfo = `<span class="ts-info">Beta(α=1, β=1)</span>`;
    } else if (algorithmSelect.value === 'ucb') {
        extraInfo = `<span class="ucb-info">Bônus UCB: 0</span>`;
    }
    
    div.innerHTML = `
        <h3>Braço ${id + 1}</h3>
        <div class="bandit-info">
            <span>Estimativa (Q): <strong class="estimate">0.00</strong></span>
            <span>Escolhas (N): <strong class="pulls">0</strong></span>
            ${extraInfo}
        </div>
        <div class="bandit-reward"></div>
    `;
    return div;
}

function initBanditConfigPanel() {
    banditConfigContainer.innerHTML = '';
    
    for (let i = 0; i < NUM_BANDITS; i++) {
        const configItem = document.createElement('div');
        configItem.className = 'bandit-config-item';
        configItem.innerHTML = `
            <label for="bandit-prob-${i}">Braço ${i+1} Probabilidade:</label>
            <input type="number" id="bandit-prob-${i}" min="0" max="1" step="0.01" value="${TRUE_PROBABILITIES[i]}">
        `;
        banditConfigContainer.appendChild(configItem);
    }
}

function initializeCharts() {
    const banditLabels = bandits.map(b => `Braço ${b.id + 1}`);

    // --- Reward Chart ---
    if (rewardChart) rewardChart.destroy(); // Clear previous chart
    rewardChart = new Chart(rewardChartCtx, {
        type: 'line',
        data: {
            labels: [], // Steps
            datasets: [{
                label: 'Recompensa Atual',
                data: [], // Cumulative rewards
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }, {
                label: 'Recompensa Ótima',
                data: [], // Optimal rewards
                borderColor: 'rgba(192, 75, 75, 0.5)',
                borderDash: [5, 5],
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            scales: { 
                y: { beginAtZero: true, title: { display: true, text: 'Recompensa Total' } },
                x: { title: { display: true, text: 'Passo' } } 
            },
            animation: false // Faster updates
        }
    });

    // --- Pulls Chart ---
    if (pullsChart) pullsChart.destroy(); // Clear previous chart
    pullsChart = new Chart(pullsChartCtx, {
        type: 'bar',
        data: {
            labels: banditLabels,
            datasets: [{
                label: 'Número de Escolhas',
                data: bandits.map(() => 0), // Initial pulls
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(54, 162, 235, 0.5)',
                    'rgba(255, 206, 86, 0.5)',
                    'rgba(75, 192, 192, 0.5)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: { y: { beginAtZero: true, title: { display: true, text: 'Contagem' } } },
            indexAxis: 'y', // Horizontal bars
            animation: false
        }
    });
    
    // --- Regret Chart ---
    if (regretChart) regretChart.destroy();
    regretChart = new Chart(regretChartCtx, {
        type: 'line',
        data: {
            labels: [], // Steps
            datasets: [{
                label: 'Regret Cumulativo',
                data: [], // Cumulative regret
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1,
                fill: false
            }]
        },
        options: {
            scales: { 
                y: { beginAtZero: true, title: { display: true, text: 'Regret Total' } },
                x: { title: { display: true, text: 'Passo' } } 
            },
            animation: false
        }
    });
    
    // --- Explore/Exploit Chart ---
    if (exploreExploitChart) exploreExploitChart.destroy();
    exploreExploitChart = new Chart(exploreExploitChartCtx, {
        type: 'line',
        data: {
            labels: [], // Steps
            datasets: [{
                label: '% Exploração',
                data: [], // Exploration percentage
                borderColor: 'rgb(255, 159, 64)',
                backgroundColor: 'rgba(255, 159, 64, 0.2)',
                tension: 0.1,
                fill: true
            }, {
                label: '% Exploração',
                data: [], // Exploitation percentage
                borderColor: 'rgb(54, 162, 235)',
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                tension: 0.1,
                fill: true
            }]
        },
        options: {
            scales: { 
                y: { 
                    beginAtZero: true, 
                    max: 100,
                    title: { display: true, text: 'Porcentagem (%)' } 
                },
                x: { title: { display: true, text: 'Passo' } } 
            },
            animation: false
        }
    });
}

function resetSimulation() {
    clearInterval(simulationInterval);
    isRunning = false;
    startPauseBtn.textContent = 'Iniciar';
    step = 0;
    totalReward = 0;
    optimalReward = 0;
    regret = 0;
    rewardHistory = [];
    optimalRewardHistory = [];
    regretHistory = [];
    exploreCount = 0;
    exploitCount = 0;
    explorationHistory = [];
    
    stepCounterSpan.textContent = step;
    totalRewardSpan.textContent = totalReward;
    optimalRewardSpan.textContent = optimalReward;
    regretSpan.textContent = regret;
    actionFeedbackP.textContent = 'Simulação resetada.';
    
    initializeBandits();
    initializeCharts();
    updateParameterControls(); // Ensure correct controls are visible
    updateAlgorithmInfo(); // Update algorithm description
}

// --- Core MAB Logic ---
function getReward(banditId) {
    // Simulate pulling the arm based on its true probability
    return Math.random() < bandits[banditId].trueProbability ? 1 : 0;
}

function updateEstimate(banditId, reward) {
    const bandit = bandits[banditId];
    bandit.pullCount++;
    
    if (algorithmSelect.value === 'thompson') {
        // For Thompson Sampling, track successes and failures
        if (reward === 1) {
            bandit.successCount++;
        } else {
            bandit.failureCount++;
        }
        // Update estimate for display purposes
        bandit.estimatedValue = bandit.successCount / bandit.pullCount;
    } else {
        // Incremental update formula: Q_new = Q_old + (1/N) * (R - Q_old)
        bandit.estimatedValue += (1 / bandit.pullCount) * (reward - bandit.estimatedValue);
    }
}

function chooseArm() {
    const algorithm = algorithmSelect.value;
    let chosenArmId = -1;
    let feedback = "";
    let actionType = "exploit"; // Default to exploit
    
    // --- Handle initial pulls for UCB (pull each arm once) ---
    if (algorithm === 'ucb') {
        const unpulled = bandits.find(b => b.pullCount === 0);
        if (unpulled) {
            chosenArmId = unpulled.id;
            feedback = `UCB: Primeira escolha do Braço ${chosenArmId + 1} (garantir exploração inicial).`;
            actionType = "explore";
            return { chosenArmId, feedback, actionType };
        }
    }
    
    // --- Also handle initial pulls for any algorithm if necessary ---
    if (step < NUM_BANDITS && bandits.some(b => b.pullCount === 0)) {
        const unpulled = bandits.find(b => b.pullCount === 0);
        if(unpulled) {
            chosenArmId = unpulled.id;
            feedback = `${algorithm.toUpperCase()}: Primeira escolha do Braço ${chosenArmId + 1} (inicialização).`;
            actionType = "explore";
            return { chosenArmId, feedback, actionType };
        }
    }

    // --- Algorithm Selection Logic ---
    if (algorithm === 'greedy') {
        let maxEstimate = -Infinity;
        let bestArms = [];
        
        bandits.forEach((bandit, id) => {
            if (bandit.estimatedValue > maxEstimate) {
                maxEstimate = bandit.estimatedValue;
                bestArms = [id];
            } else if (bandit.estimatedValue === maxEstimate) {
                bestArms.push(id);
            }
        });
        
        // Break ties randomly
        chosenArmId = bestArms[Math.floor(Math.random() * bestArms.length)];
        feedback = `Greedy: Escolheu Braço ${chosenArmId + 1} (Maior Q = ${maxEstimate.toFixed(2)}).`;
        actionType = "exploit";

    } else if (algorithm === 'epsilon-greedy') {
        if (Math.random() < epsilon) {
            // Explore: choose a random arm
            chosenArmId = Math.floor(Math.random() * NUM_BANDITS);
            feedback = `ε-Greedy: Explorando aleatoriamente (ε=${epsilon.toFixed(2)}), escolheu Braço ${chosenArmId + 1}.`;
            actionType = "explore";
        } else {
            // Exploit: choose the best arm (like greedy)
            let maxEstimate = -Infinity;
            let bestArms = [];
            
            bandits.forEach((bandit, id) => {
                if (bandit.estimatedValue > maxEstimate) {
                    maxEstimate = bandit.estimatedValue;
                    bestArms = [id];
                } else if (bandit.estimatedValue === maxEstimate) {
                    bestArms.push(id);
                }
            });
            
            chosenArmId = bestArms[Math.floor(Math.random() * bestArms.length)];
            feedback = `ε-Greedy: Exploração (1-ε), escolheu Braço ${chosenArmId + 1} (Maior Q = ${maxEstimate.toFixed(2)}).`;
            actionType = "exploit";
        }

    } else if (algorithm === 'ucb') {
        let maxUCB = -Infinity;
        let bestArmId = -1;
        const totalPulls = step + 1; // Use step+1 as total pulls for log

        bandits.forEach((bandit, id) => {
            let ucbScore;
            
            if (bandit.pullCount === 0) {
                // Should have been handled above, but as fallback, treat as infinite
                ucbScore = Infinity;
            } else {
                // UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
                const explorationBonus = ucbC * Math.sqrt(Math.log(totalPulls) / bandit.pullCount);
                ucbScore = bandit.estimatedValue + explorationBonus;

                // Store bonus for feedback
                bandit.ucbValue = ucbScore;
                bandit.lastExplorationBonus = explorationBonus;
            }

            if (ucbScore > maxUCB) {
                maxUCB = ucbScore;
                bestArmId = id;
            }
        });
        
        chosenArmId = bestArmId; // Arm with highest UCB score
        const chosenBandit = bandits[chosenArmId];
        
        feedback = `UCB: Escolheu Braço ${chosenArmId + 1} (Q=${chosenBandit.estimatedValue.toFixed(2)} + Bonus=${chosenBandit.lastExplorationBonus.toFixed(2)} = UCB ${maxUCB.toFixed(2)}).`;
        
        // Determine if this is exploration or exploitation
        const bestEstimateId = bandits.reduce((maxId, bandit, id, arr) => 
            bandit.estimatedValue > arr[maxId].estimatedValue ? id : maxId, 0);
            
        actionType = (chosenArmId !== bestEstimateId) ? "explore" : "exploit";

    } else if (algorithm === 'thompson') {
        let maxSample = -Infinity;
        let bestArmId = -1;
        
        bandits.forEach((bandit, id) => {
            // Sample from Beta distribution
            const alpha = bandit.successCount + 1;
            const beta = bandit.failureCount + 1;
            const sample = betaDistribution(alpha, beta);
            
            bandit.lastSample = sample; // Store for visualization
            
            if (sample > maxSample) {
                maxSample = sample;
                bestArmId = id;
            }
        });
        
        chosenArmId = bestArmId;
        const chosenBandit = bandits[chosenArmId];
        
        feedback = `Thompson: Escolheu Braço ${chosenArmId + 1} (Amostra Beta(${chosenBandit.successCount + 1}, ${chosenBandit.failureCount + 1}) = ${maxSample.toFixed(3)})`;
        
        // For Thompson, we consider it exploration if it chooses non-greedy action
        const bestEstimateId = bandits.reduce((maxId, bandit, id, arr) => 
            bandit.estimatedValue > arr[maxId].estimatedValue ? id : maxId, 0);
            
        actionType = (chosenArmId !== bestEstimateId) ? "explore" : "exploit";
    }

    if (chosenArmId === -1) {
        // Fallback: choose randomly if something went wrong
        chosenArmId = Math.floor(Math.random() * NUM_BANDITS);
        feedback = `Fallback: Escolhendo aleatoriamente Braço ${chosenArmId + 1}.`;
        actionType = "explore";
    }

    return { chosenArmId, feedback, actionType };
}

// --- Simulation Step ---
function simulationStep() {
    if (!isRunning && !lastAction.stepping) return;

    step++;
    stepCounterSpan.textContent = step;

    // 1. Choose Arm
    const { chosenArmId, feedback, actionType } = chooseArm();
    actionFeedbackP.textContent = feedback;

    // 2. Get Reward
    const reward = getReward(chosenArmId);

    // 3. Update Estimate and Counts
    updateEstimate(chosenArmId, reward);

    // 4. Track action types
    if (actionType === "explore") {
        exploreCount++;
    } else {
        exploitCount++;
    }
    
    // 5. Update Total Reward & Optimal Reward
    totalReward += reward;
    optimalReward += OPTIMAL_PROBABILITY; // Expected reward from optimal arm
    regret += OPTIMAL_PROBABILITY - bandits[chosenArmId].trueProbability; // Update regret
    
    totalRewardSpan.textContent = totalReward.toFixed(1);
    optimalRewardSpan.textContent = optimalReward.toFixed(1);
    regretSpan.textContent = regret.toFixed(1);
    
    // 6. Update histories for charts
    rewardHistory.push(totalReward);
    optimalRewardHistory.push(optimalReward);
    regretHistory.push(regret);
    
    // Calculate exploration/exploitation percentages
    explorationHistory.push({
        explore: (exploreCount / step) * 100,
        exploit: (exploitCount / step) * 100
    });

    // 7. Update UI
    updateUI(chosenArmId, reward, actionType);

    // 8. Store last action details
    lastAction = {
        type: actionType,
        reason: feedback,
        banditId: chosenArmId,
        stepping: lastAction.stepping && false // Reset stepping flag if it was set
    };
    
    // 9. If we're in stepping mode, stop after this step
    if (lastAction.stepping) {
        isRunning = false;
        startPauseBtn.textContent = 'Continuar';
    }
}

// --- UI Update ---
function updateUI(chosenArmId, reward, actionType) {
    // Update bandit displays
    bandits.forEach((bandit, id) => {
        const estimateEl = bandit.element.querySelector('.estimate');
        const pullsEl = bandit.element.querySelector('.pulls');
        const rewardEl = bandit.element.querySelector('.bandit-reward');
        
        // Update Thompson Sampling display if needed
        if (algorithmSelect.value === 'thompson' && bandit.element.querySelector('.ts-info')) {
            bandit.element.querySelector('.ts-info').textContent = 
                `Beta(α=${bandit.successCount + 1}, β=${bandit.failureCount + 1})`;
        }
        
        // Update UCB info if needed
        if (algorithmSelect.value === 'ucb' && bandit.element.querySelector('.ucb-info')) {
            const bonus = bandit.lastExplorationBonus !== undefined ? 
                bandit.lastExplorationBonus.toFixed(2) : '0';
            bandit.element.querySelector('.ucb-info').textContent = `Bônus UCB: ${bonus}`;
        }

        estimateEl.textContent = bandit.estimatedValue.toFixed(2);
        pullsEl.textContent = bandit.pullCount;

        // Reset all classes first
        bandit.element.classList.remove('selected', 'explore', 'exploit');
        
        if (id === chosenArmId) {
            bandit.element.classList.add('selected');
            bandit.element.classList.add(actionType); // Add explore/exploit class
            
            rewardEl.textContent = `Recebeu: ${reward}`;
            rewardEl.className = reward > 0 ? 
                `bandit-reward ${actionType}` : 
                `bandit-reward no-reward ${actionType}`;
        } else {
            rewardEl.textContent = ''; // Clear reward text for others
        }
    });

    // Update Charts
    // Reward Chart
    rewardChart.data.labels.push(step);
    rewardChart.data.datasets[0].data.push(totalReward);
    rewardChart.data.datasets[1].data.push(optimalReward);
    rewardChart.update();

    // Pulls Chart
    pullsChart.data.datasets[0].data = bandits.map(b => b.pullCount);
    pullsChart.update();
    
    // Regret Chart
    regretChart.data.labels.push(step);
    regretChart.data.datasets[0].data.push(regret);
    regretChart.update();
    
    // Explore/Exploit Chart
    const lastEntry = explorationHistory[explorationHistory.length - 1];
    exploreExploitChart.data.labels.push(step);
    exploreExploitChart.data.datasets[0].data.push(lastEntry.explore);
    exploreExploitChart.data.datasets[1].data.push(lastEntry.exploit);
    exploreExploitChart.update();
}

function updateParameterControls() {
    const algorithm = algorithmSelect.value;
    
    // Hide all controls first
    epsilonControl.style.display = 'none';
    ucbControl.style.display = 'none';
    thompsonControl.style.display = 'none';
    
    // Show relevant controls
    if (algorithm === 'epsilon-greedy') {
        epsilonControl.style.display = 'flex';
    } else if (algorithm === 'ucb') {
        ucbControl.style.display = 'flex';
    } else if (algorithm === 'thompson') {
        thompsonControl.style.display = 'flex';
    }
    
    // Update algorithm information
    updateAlgorithmInfo();
}

function updateAlgorithmInfo() {
    const algorithm = algorithmSelect.value;
    const info = algorithmDescriptions[algorithm];
    
    currentAlgorithmName.textContent = info.name;
    algorithmExplanation.textContent = info.description;
    algorithmFormula.textContent = info.formula;
}

// --- Event Listeners ---
startPauseBtn.addEventListener('click', () => {
    if (isRunning) {
        clearInterval(simulationInterval);
        startPauseBtn.textContent = 'Continuar';
    } else {
        simulationInterval = setInterval(simulationStep, intervalTime);
        startPauseBtn.textContent = 'Pausar';
    }
    isRunning = !isRunning;
});

resetBtn.addEventListener('click', resetSimulation);

stepBtn.addEventListener('click', () => {
    // If running, pause first
    if (isRunning) {
        clearInterval(simulationInterval);
        isRunning = false;
        startPauseBtn.textContent = 'Continuar';
    }
    
    // Execute a single step
    lastAction.stepping = true;
    isRunning = true;
    simulationStep();
});

algorithmSelect.addEventListener('change', () => {
    resetSimulation();
});

speedSlider.addEventListener('input', (e) => {
    // Invert the slider value: faster speed means smaller interval
    intervalTime = 1050 - parseInt(e.target.value);
    speedValueSpan.textContent = `${intervalTime}ms`;
    
    if (isRunning) {
        // Reset interval if running
        clearInterval(simulationInterval);
        simulationInterval = setInterval(simulationStep, intervalTime);
    }
});

epsilonSlider.addEventListener('input', (e) => {
    epsilon = parseFloat(e.target.value);
    epsilonValueSpan.textContent = epsilon.toFixed(2);
});

ucbSlider.addEventListener('input', (e) => {
    ucbC = parseFloat(e.target.value);
    ucbValueSpan.textContent = ucbC.toFixed(1);
});

// Bandit configuration panel
configToggleBtn.addEventListener('click', () => {
    if (configPanel.style.display === 'none') {
        configPanel.style.display = 'block';
        configToggleBtn.textContent = 'Configurar Bandits ▲';
    } else {
        configPanel.style.display = 'none';
        configToggleBtn.textContent = 'Configurar Bandits ▼';
    }
});

applyConfigBtn.addEventListener('click', () => {
    // Update bandit probabilities
    for (let i = 0; i < NUM_BANDITS; i++) {
        const input = document.getElementById(`bandit-prob-${i}`);
        const value = parseFloat(input.value);
        if (!isNaN(value) && value >= 0 && value <= 1) {
            TRUE_PROBABILITIES[i] = value;
        }
    }
    
    resetSimulation();
    configPanel.style.display = 'none';
    configToggleBtn.textContent = 'Configurar Bandits ▼';
});

// --- Setup tooltips ---
document.addEventListener('DOMContentLoaded', () => {
    // Initialize tooltips if tippy.js is loaded
    if (typeof tippy === 'function') {
        tippy('[data-tippy-content]');
    }
    
    resetSimulation(); // Initialize everything on load
});