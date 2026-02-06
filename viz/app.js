// QuantNet-Solver D3.js Visualization
// Polls JSON file for real-time solver progress (no WebSocket needed)

class SolverVisualization {
    constructor() {
        this.convergenceData = [];
        this.currentStrategy = null;
        this.currentActionEVs = null;
        this.maxDataPoints = 500;
        this.lastIterationCount = 0;
        this.isComplete = false;
        this.lastBeta = null;

        this.initChart();
        this.poll();
    }

    async poll() {
        if (this.isComplete) {
            this.updateStatus('complete', 'Complete');
            return;
        }

        try {
            // Add timestamp to prevent caching
            const response = await fetch('solver_output.json?' + Date.now());
            if (response.ok) {
                const data = await response.json();
                this.handleData(data);
                this.updateStatus('connected', 'Connected');
            } else {
                this.updateStatus('waiting', 'Waiting for solver...');
            }
        } catch (e) {
            this.updateStatus('waiting', 'Waiting for solver...');
        }

        // Poll every 200ms
        setTimeout(() => this.poll(), 200);
    }

    updateStatus(className, text) {
        const el = document.getElementById('connection-status');
        el.className = 'value ' + className;
        el.textContent = text;
    }

    handleData(data) {
        // Process any new iterations first (before checking completion)
        if (data.iteration_count > this.lastIterationCount) {
            const iterations = data.iterations || [];
            for (let i = this.lastIterationCount; i < iterations.length; i++) {
                this.handleIteration(iterations[i]);
            }
            this.lastIterationCount = data.iteration_count;
        }

        // Check if solver is complete (after processing all iterations)
        if (data.latest && data.latest.type === 'complete') {
            this.isComplete = true;
            document.getElementById('current-residual').textContent = 'Complete';
            document.getElementById('current-exploit').textContent =
                data.latest.final_exploitability.toExponential(3) + ' (final)';
        }
    }

    handleIteration(data) {
        // Update status bar
        document.getElementById('game-name').textContent = data.game || '-';
        document.getElementById('current-beta').textContent =
            data.beta !== undefined ? data.beta.toFixed(2) : '-';
        document.getElementById('current-iter').textContent =
            data.iteration !== undefined ? data.iteration : '-';
        document.getElementById('current-residual').textContent =
            data.residual_norm !== undefined ? data.residual_norm.toExponential(3) : '-';

        if (data.exploitability !== undefined) {
            document.getElementById('current-exploit').textContent =
                data.exploitability.toExponential(3);
        }

        // Update EV (P0 payoff) in status bar
        if (data.expected_value !== undefined) {
            const evEl = document.getElementById('current-ev');
            if (evEl) {
                evEl.textContent = data.expected_value.toFixed(4);
                evEl.style.color = data.expected_value >= 0 ? '#4CAF50' : '#f44336';
            }
        }

        // Track beta transitions for chart markers
        const isBetaChange = this.lastBeta !== null && data.beta !== this.lastBeta;
        this.lastBeta = data.beta;

        // Add to convergence data
        this.convergenceData.push({
            iteration: this.convergenceData.length,
            residual: data.residual_norm || 1e-10,
            beta: data.beta || 0,
            exploitability: data.exploitability || 0,
            isBetaChange: isBetaChange
        });

        // Limit data points
        if (this.convergenceData.length > this.maxDataPoints) {
            this.convergenceData.shift();
        }

        // Update chart
        this.updateChart();

        // Update strategy display
        if (data.strategy) {
            this.currentStrategy = data.strategy;
            this.currentActionEVs = data.action_evs || null;
            this.updateStrategyDisplay();
        }
    }

    // Parse info set ID like "P0:K:cb" into structured object
    parseInfoSetId(id) {
        const parts = id.split(':');
        const player = parseInt(parts[0][1]);
        const card = parts[1];
        const history = parts[2] || '';
        return { player, card, history, id };
    }

    // Convert history string to human-readable label
    historyLabel(history, player) {
        if (history === '') {
            return player === 0 ? 'Opening action' : 'Response to check';
        }
        const labels = {
            'c': 'After opponent checks',
            'b': 'Facing a bet',
            'cb': 'Facing bet after check'
        };
        return labels[history] || history;
    }

    // Get card display info
    cardInfo(card) {
        const info = {
            'J': { name: 'Jack', class: 'jack', symbol: 'J' },
            'Q': { name: 'Queen', class: 'queen', symbol: 'Q' },
            'K': { name: 'King', class: 'king', symbol: 'K' }
        };
        return info[card] || { name: card, class: 'unknown', symbol: card };
    }

    initChart() {
        const container = document.getElementById('convergence-panel');
        const svg = d3.select('#convergence-chart');

        // Get dimensions
        const width = container.clientWidth - 40;
        const height = 300;
        const margin = { top: 30, right: 60, bottom: 40, left: 60 };

        svg.attr('width', width)
           .attr('height', height);

        // Create scales
        this.chartWidth = width - margin.left - margin.right;
        this.chartHeight = height - margin.top - margin.bottom;

        this.xScale = d3.scaleLinear()
            .range([0, this.chartWidth]);

        this.yScaleResidual = d3.scaleLog()
            .range([this.chartHeight, 0]);

        this.yScaleExploit = d3.scaleLog()
            .range([this.chartHeight, 0]);

        // Create chart group
        this.chartGroup = svg.append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        // Add axes
        this.xAxis = this.chartGroup.append('g')
            .attr('class', 'axis x-axis')
            .attr('transform', `translate(0,${this.chartHeight})`);

        this.yAxisLeft = this.chartGroup.append('g')
            .attr('class', 'axis y-axis-left');

        this.yAxisRight = this.chartGroup.append('g')
            .attr('class', 'axis y-axis-right')
            .attr('transform', `translate(${this.chartWidth},0)`);

        // Add axis labels
        this.chartGroup.append('text')
            .attr('class', 'axis-label')
            .attr('x', this.chartWidth / 2)
            .attr('y', this.chartHeight + 35)
            .attr('text-anchor', 'middle')
            .text('Iteration');

        this.chartGroup.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -this.chartHeight / 2)
            .attr('y', -45)
            .attr('text-anchor', 'middle')
            .text('Residual');

        this.chartGroup.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(90)')
            .attr('x', this.chartHeight / 2)
            .attr('y', -this.chartWidth - 45)
            .attr('text-anchor', 'middle')
            .text('Exploitability');

        // Add line generators
        this.residualLine = d3.line()
            .x(d => this.xScale(d.iteration))
            .y(d => this.yScaleResidual(Math.max(d.residual, 1e-15)));

        this.exploitLine = d3.line()
            .x(d => this.xScale(d.iteration))
            .y(d => this.yScaleExploit(Math.max(d.exploitability || 1e-15, 1e-15)));

        // Add path for residual
        this.residualPath = this.chartGroup.append('path')
            .attr('class', 'line residual-line')
            .attr('fill', 'none')
            .attr('stroke', '#4CAF50')
            .attr('stroke-width', 2);

        // Add path for exploitability
        this.exploitPath = this.chartGroup.append('path')
            .attr('class', 'line exploit-line')
            .attr('fill', 'none')
            .attr('stroke', '#FF9800')
            .attr('stroke-width', 2);

        // Add legend
        const legend = this.chartGroup.append('g')
            .attr('class', 'legend')
            .attr('transform', `translate(${this.chartWidth - 120}, -15)`);

        legend.append('line')
            .attr('x1', 0).attr('y1', 0)
            .attr('x2', 20).attr('y2', 0)
            .attr('stroke', '#4CAF50')
            .attr('stroke-width', 2);
        legend.append('text')
            .attr('x', 25).attr('y', 4)
            .text('Residual');

        legend.append('line')
            .attr('x1', 0).attr('y1', 20)
            .attr('x2', 20).attr('y2', 20)
            .attr('stroke', '#FF9800')
            .attr('stroke-width', 2);
        legend.append('text')
            .attr('x', 25).attr('y', 24)
            .text('Exploitability');
    }

    updateChart() {
        if (this.convergenceData.length < 2) return;

        const data = this.convergenceData;

        // Update scales
        this.xScale.domain([0, data.length - 1]);

        const residualExtent = d3.extent(data, d => d.residual);
        const exploitExtent = d3.extent(data.filter(d => d.exploitability > 0), d => d.exploitability);

        this.yScaleResidual.domain([
            Math.max(residualExtent[0] * 0.1, 1e-15),
            residualExtent[1] * 10
        ]);

        if (exploitExtent[0] && exploitExtent[1]) {
            this.yScaleExploit.domain([
                Math.max(exploitExtent[0] * 0.1, 1e-15),
                exploitExtent[1] * 10
            ]);
        }

        // Update axes with smooth transition
        this.xAxis.transition().duration(100)
            .call(d3.axisBottom(this.xScale).ticks(5));

        this.yAxisLeft.transition().duration(100)
            .call(d3.axisLeft(this.yScaleResidual).ticks(5, '.0e'));

        this.yAxisRight.transition().duration(100)
            .call(d3.axisRight(this.yScaleExploit).ticks(5, '.0e'));

        // Update lines with smooth transition
        this.residualPath
            .datum(data)
            .transition().duration(100)
            .attr('d', this.residualLine);

        const dataWithExploit = data.filter(d => d.exploitability > 0);
        this.exploitPath
            .datum(dataWithExploit)
            .transition().duration(100)
            .attr('d', this.exploitLine);

        // Remove old beta markers
        this.chartGroup.selectAll('.beta-marker').remove();
        this.chartGroup.selectAll('.beta-label').remove();

        // Add beta transition markers (vertical dashed lines)
        const betaChanges = data.filter(d => d.isBetaChange);

        this.chartGroup.selectAll('.beta-marker')
            .data(betaChanges)
            .enter()
            .append('line')
            .attr('class', 'beta-marker')
            .attr('x1', d => this.xScale(d.iteration))
            .attr('x2', d => this.xScale(d.iteration))
            .attr('y1', 0)
            .attr('y2', this.chartHeight)
            .attr('stroke', '#4A90D9')
            .attr('stroke-width', 1)
            .attr('stroke-dasharray', '4,3')
            .attr('opacity', 0.6);

        // Add beta labels at top
        this.chartGroup.selectAll('.beta-label')
            .data(betaChanges)
            .enter()
            .append('text')
            .attr('class', 'beta-label')
            .attr('x', d => this.xScale(d.iteration) + 3)
            .attr('y', -8)
            .text(d => 'β=' + d.beta.toFixed(1))
            .attr('fill', '#4A90D9')
            .attr('font-size', '9px')
            .attr('font-family', "'Fira Code', monospace");
    }

    updateStrategyDisplay() {
        const p0Container = document.querySelector('#strategy-p0 .player-strategies');
        const p1Container = document.querySelector('#strategy-p1 .player-strategies');

        if (!p0Container || !p1Container) return;

        p0Container.innerHTML = '';
        p1Container.innerHTML = '';

        if (!this.currentStrategy) return;

        // Parse and group info sets by player and card
        const grouped = { 0: {}, 1: {} };
        const cardOrder = ['J', 'Q', 'K'];

        for (const isId of Object.keys(this.currentStrategy)) {
            const parsed = this.parseInfoSetId(isId);
            if (!grouped[parsed.player][parsed.card]) {
                grouped[parsed.player][parsed.card] = [];
            }
            grouped[parsed.player][parsed.card].push({
                ...parsed,
                actions: this.currentStrategy[isId],
                evs: this.currentActionEVs ? this.currentActionEVs[isId] : null
            });
        }

        // Render each player's strategies
        for (const player of [0, 1]) {
            const container = player === 0 ? p0Container : p1Container;

            for (const card of cardOrder) {
                if (!grouped[player][card]) continue;

                const cardData = this.cardInfo(card);
                const cardGroup = document.createElement('div');
                cardGroup.className = 'card-group';

                // Card badge
                const badge = document.createElement('div');
                badge.className = `card-badge ${cardData.class}`;
                badge.innerHTML = `<span class="card-symbol">${cardData.symbol}</span> ${cardData.name}`;
                cardGroup.appendChild(badge);

                // Each decision point for this card
                for (const is of grouped[player][card]) {
                    const decisionDiv = document.createElement('div');
                    decisionDiv.className = 'decision-point';

                    const contextLabel = document.createElement('div');
                    contextLabel.className = 'decision-context';
                    contextLabel.textContent = this.historyLabel(is.history, player);
                    decisionDiv.appendChild(contextLabel);

                    const actionsDiv = document.createElement('div');
                    actionsDiv.className = 'decision-actions';

                    for (const [action, prob] of Object.entries(is.actions)) {
                        const actionDiv = document.createElement('div');
                        actionDiv.className = 'action';

                        const label = document.createElement('span');
                        label.className = 'action-label';
                        label.textContent = action.charAt(0).toUpperCase() + action.slice(1);

                        const barContainer = document.createElement('div');
                        barContainer.className = 'bar-container';

                        const bar = document.createElement('div');
                        bar.className = 'bar';
                        bar.style.width = `${prob * 100}%`;

                        // Color based on probability
                        if (prob > 0.7) {
                            bar.style.backgroundColor = '#4CAF50';
                        } else if (prob > 0.3) {
                            bar.style.backgroundColor = '#FFC107';
                        } else {
                            bar.style.backgroundColor = '#f44336';
                        }

                        const probValue = document.createElement('span');
                        probValue.className = 'action-prob';
                        probValue.textContent = prob.toFixed(3);

                        // Add EV value if available
                        const evValue = document.createElement('span');
                        evValue.className = 'action-ev';
                        if (is.evs && is.evs[action] !== undefined) {
                            const ev = is.evs[action];
                            evValue.textContent = `EV: ${ev >= 0 ? '+' : ''}${ev.toFixed(3)}`;
                            evValue.style.color = ev >= 0 ? '#4CAF50' : '#f44336';
                        }

                        barContainer.appendChild(bar);
                        actionDiv.appendChild(label);
                        actionDiv.appendChild(barContainer);
                        actionDiv.appendChild(probValue);
                        actionDiv.appendChild(evValue);
                        actionsDiv.appendChild(actionDiv);
                    }

                    decisionDiv.appendChild(actionsDiv);
                    cardGroup.appendChild(decisionDiv);
                }

                container.appendChild(cardGroup);
            }
        }
    }
}

// Initialize visualization when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.viz = new SolverVisualization();
});
